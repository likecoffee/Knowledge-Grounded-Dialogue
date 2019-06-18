# -*- coding: utf-8 -*-
# file: decoder.py
# author: JiachenDu <jacobvan199165@gmail.com>
# Copyright (C) 2018, 2019. All right Reserved

import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import rnn_cell
import attention 
import output
from utils import PAD_TOKEN, generate_mask_by_length, cos_sim

class DecoderRNN(nn.Module):
    def __init__(self, word_embedding, src_word_size, src_context_size, KG_word_size,
                       rnn_size,  num_layers, num_softmax, dropout = 0):
        super(DecoderRNN, self).__init__()
        self.num_vocab, self.emb_size = word_embedding.weight.size()
        self.src_word_size = src_word_size
        self.src_context_size = src_context_size
        self.KG_word_size = KG_word_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.num_softmax = num_softmax
        self.dropout = dropout

        self.word_embedding = word_embedding
        self.rnn_cell = rnn_cell.StackedGRUCell(self.emb_size + self.src_word_size + self.KG_word_size, 
                                                self.rnn_size,
                                                self.num_layers,
                                                self.dropout)
        self.context2hidden = nn.Linear(self.src_context_size, self.rnn_size * self.num_layers)
        self.attention = attention.Attention(self.src_word_size, rnn_size, type = 'dot') 
        self.KG_attention = attention.Attention(self.KG_word_size, rnn_size + src_word_size, type = "dot")
        self.gen_dist_trans = nn.Linear(self.KG_word_size + self.rnn_size + self.emb_size, 1)
        self.output = output.OutputLayer(self.word_embedding, self.rnn_size,
                                         self.num_softmax, self.dropout, padding_idx = PAD_TOKEN) 
    
    def forward_step(self, src_context_output, src_word_output, src_word_mask, 
                     KG_word_output, KG_word_seq, KG_word_mask, hidden, tgt_word, combine_knowledge = False):
        """
            Args:
                src_context_output (FloatTensor) : (batch_size, context_size)
                src_word_output (FloatTensor)    : (batch_size, src_word_len, word_size)
                src_word_mask (FloatTensor)      : (batch_size, src_word_len)
                KG_word_output (FloatTensor)     : (batch_size, KG_max_word_len, KG_word_size)
                KG_word_seq (LongTensor)         : (batch_size, src_max_word_len)
                KG_word_mask (FloatTensor)       : (batch_size, KG_word_len)
                last_hidden (FloatTensor)        : (batch_size, rnn_size)
                tgt_word (LongTensor)            : (batch_size)
            Regurns:
                hidden (FloatTensor)             : (num_layer, batch_size, rnn_size)
                logit_word (FloatTensor)         : (batch_size, num_vocab)
        """
        batch_size = tgt_word.size(0)
        # last_hidden : [batch_size, rnn_size]
        last_hidden = hidden[-1]
        # attn_src_word : [batch_size, src_word_size] 
        attn_src_word,_ = self.attention(last_hidden, src_word_output, src_word_output, src_word_mask)
        # attn_KG_word : [batch_size, src_KG_size] 
        KG_attention_query = last_hidden + attn_src_word
        attn_KG_word, attn_KG_scores = self.attention(KG_attention_query, KG_word_output, KG_word_output, KG_word_mask)
        # tgt_word_emb = [batch_size, emb_size]
        tgt_word_emb = self.word_embedding(tgt_word) 
        # rnn_inputs : [barch_size, emb_size + src_word_size]
        rnn_inputs = torch.cat([tgt_word_emb, attn_src_word, attn_KG_word], dim = 1)
        # rnn_output : [batch_size, rnn_size] ; hidden : [num_layer, batch_size, rnn_size]
        rnn_output, hidden = self.rnn_cell(rnn_inputs, hidden)
        # prob_word : [batch_size, num_vocab]
        prob_word = self.output(rnn_output)
        if combine_knowledge == True:
            # copy_dist : [batch_size, num_vocab]
            copy_prob = torch.zeros(batch_size, self.num_vocab, device = src_context_output.device)
            copy_prob = torch.scatter_add(input = copy_prob, 
                                          dim = 1, 
                                          index = KG_word_seq, src = attn_KG_scores.permute(1, 0))
            # gen_dist_trans_input : [batch_size, emb_size + KG_rnn_size + rnn_size]
            gen_dist_trans_input = torch.cat([tgt_word_emb, attn_KG_word, rnn_output], dim = 1)
            gen_dist = self.gen_dist_trans(gen_dist_trans_input)
            gen_dist = torch.sigmoid(gen_dist)
            # combined_prob_word: [batch_size, num_vocab]
            combined_prob_word = prob_word * gen_dist + (1 - gen_dist) * copy_prob
        else:
            combined_prob_word = prob_word
        # logit_word : [batch_size, num_vocab]
        logit_word = combined_prob_word.log()

        return hidden, logit_word

    def greedy_generate(self, src_context_output, src_word_output, src_word_len, KG_word_output, 
                        KG_word_len, KG_word_seq, init_tgt_word_input, max_tgt_word_len, combine_knowledge = False,
                        temperature = None, topk = None, topp = None):
        """Generate the sequence given source sequence.
        Args:
            src_context_output (FloatTensor) :  (batch_size, context_size)
            src_word_output (FloatTensor)    :  (batch_size, src_word_len, word_size)
            KG_word_output (FloatTensor)     :  (batch_size, KG_max_word_len, KG_word_size)
            KG_word_len (LongTensor)         :  (batch_size)
            KG_word_seq (LongTensor)         :  (batch_size, src_max_word_len)
            init_tgt_word_input (LongTensor) :  (batch_size)
            max_tgt_word_len (int)           :  the maximum lenght of target sequence
        Regurns:
            logit (FloatTensor) :   (batch_size, num_vocab)
        """
        assert src_context_output.size(0) == src_word_output.size(0)
        batch_size = src_context_output.size(0)
        max_src_len = src_word_output.size(1)
        max_KG_len = KG_word_output.size(1)
        # src_word_output : [src_word_len, batch_size, src_word_size]
        src_word_output = src_word_output.permute(1, 0, 2)
        # src_word_mask : [max_src_len, batch_size]
        src_word_mask = generate_mask_by_length(src_word_len, max_src_len) 
        # KG_word_output : [KG_word_len, batch_size, KG_word_size]
        KG_word_output = KG_word_output.permute(1, 0, 2)
        # KG_word_mask : [max_KG_len, batch_size]
        KG_word_mask = generate_mask_by_length(KG_word_len, max_KG_len) 
        # hidden : [batch_size, num_layer, rnn_size]
        hidden = self.context2hidden(src_context_output).view(batch_size, self.num_layers, self.rnn_size)
        # hidden : [num_layer, batch_size, rnn_size]
        hidden = hidden.permute(1, 0, 2)

        tgt_word = init_tgt_word_input
        tgt_word_list = []
        for word_index in range(max_tgt_word_len):
            # hidden : [num_layer, batch_size, rnn_size]
            # logit_word : [batch_size, num_vocab]
            hidden, logit_word = self.forward_step(src_context_output, src_word_output, src_word_mask,
                                           KG_word_output, KG_word_seq, KG_word_mask, hidden, tgt_word, combine_knowledge)
            # original_prob : [batch_size, num_vocab]
            original_prob = F.softmax(logit_word, dim = 1)
            if temperature is not None:
                # temperature_prob : [batch_size, num_vocab]
                scaled_prob = torch.pow(original_prob, 1 / temperature)
                temperature_prob = scaled_prob / scaled_prob.sum(dim = 1, keepdim = True)
                # tgt_word : [batch_size]
                tgt_word = temperature_prob.multinomial(1).squeeze(1)
            elif topk is not None:
                # topk_prob : [batch_size, topk]
                # topk_indices : [batch_size, topk]
                topk_prob, topk_indices = original_prob.topk(topk, dim = 1)  
                topk_prob = topk_prob / topk_prob.sum(dim = 1, keepdim = True)
                # tgt_word_indices : [batch_size, 1]
                tgt_word_indices = topk_prob.multinomial(1)
                # tgt_word : [batch_size]
                tgt_word =  torch.gather(input = topk_indices,
                                         dim = 1,
                                         index = tgt_word_indices)
                tgt_word = tgt_word.squeeze(1)
            elif topp is not None:
                # sorted_prob : [batch_size, num_vocab]
                # sorted_indices : [batch_size, num_vocab]
                sorted_prob, sorted_indices = original_prob.sort(dim = 1, descending = True)
                # cumsum_prob : [batch_size, num_vocab]
                cumsum_prob = sorted_prob.cumsum(dim = 1)
                cumsum_prob[cumsum_prob > topp] = 0
                cumsum_prob = cumsum_prob / cumsum_prob.sum(dim = 1, keepdim = True)
                # tgt_word_indices : [batch_size, 1]
                tgt_word_indices = cumsum_prob.multinomial(1)
                tgt_word = torch.gather(input = sorted_indices,
                                        dim = 1,
                                        index = tgt_word_indices)
                tgt_word = tgt_word.squeeze(1)
            else:
                _, tgt_word = logit_word.max(dim = 1)
            tgt_word_list.append(tgt_word)
        # tgt_word_sequence : [batch_size, max_tgt_word_len] 
        tgt_word_sequence = torch.stack(tgt_word_list, dim = 1)

        return tgt_word_sequence

    def forward(self, src_context_output, src_word_output, src_word_len, 
                KG_word_output, KG_word_len, KG_word_seq, tgt_word_input, combine_knowledge = False):
        """Compute decoder scores from context_output.
        Args:
            src_context_output (FloatTensor) : (batch_size, context_size)
            src_word_output (FloatTensor)    : (batch_size, src_max_word_len, word_size)
            src_word_len (LongTensor)        : (batch_size)
            KG_word_output (FloatTensor)     : (batch_size, KG_max_word_len, KG_word_size)
            KG_word_len (LongTensor)         : (batch_size)
            KG_word_seq (LongTensor)         : (batch_size, src_max_word_len)
            tgt_word_input (LongTensor)      : (batch_size, tgt_word_len)
        Regurns:
            logit (FloatTensor)              : (batch_size, tgt_word_len, num_vocab)
            converage (FloatTensor)          : (batch_size, tgt_word_len)
        """
        assert src_context_output.size(0) == src_word_output.size(0)
        assert src_context_output.size(0) == tgt_word_input.size(0)
        assert src_context_output.size(0) == KG_word_output.size(0)  
        batch_size = src_context_output.size(0)
        max_src_len = src_word_output.size(1)
        max_KG_len = KG_word_output.size(1)
        max_tgt_len = tgt_word_input.size(1)

        # prepare the source word and KG word outputs
        # src_word_output : [src_word_len, batch_size, src_word_size]
        src_word_output = src_word_output.permute(1, 0, 2)
        # src_word_mask : [max_src_len, batch_size]
        src_word_mask = generate_mask_by_length(src_word_len, max_src_len) 
        # KG_word_output : [KG_word_len, batch_size, KG_word_size]
        KG_word_output = KG_word_output.permute(1, 0, 2)
        # KG_word_mask : [max_KG_len, batch_size]
        KG_word_mask = generate_mask_by_length(KG_word_len, max_KG_len) 

        # obtain word embedding and initial hidden states
        # tgt_word_emb : [batch_size, tgt_word_len, emb_size]
        tgt_word_emb = self.word_embedding(tgt_word_input)
        # hidden : [batch_size, num_layer, rnn_size]
        hidden = self.context2hidden(src_context_output).view(batch_size, self.num_layers, self.rnn_size)
        # hidden : [num_layer, batch_size, rnn_size]
        hidden = hidden.permute(1, 0, 2)
         
        logit_word_list = []
        coverage_list = []
        for word_index in range(max_tgt_len):
            # recurrence
            # last_hidden : [batch_size, rnn_size]
            last_hidden = hidden[-1]
            # attn_src_word : [batch_size, src_word_size] 
            attn_src_word,_ = self.attention(last_hidden, src_word_output, src_word_output, src_word_mask)
            # attn_KG_word : [batch_size, src_KG_size] 
            # attn_KG_scores : [max_KG_len ,batch_size]
            KG_attention_query = last_hidden + attn_src_word
            attn_KG_word, attn_KG_scores = self.attention(KG_attention_query, KG_word_output, KG_word_output, KG_word_mask)
            # rnn_inputs : [barch_size, emb_size + src_word_size + KG_word_size]
            rnn_inputs = torch.cat([tgt_word_emb[:,word_index], attn_src_word, attn_KG_word], dim = 1)
            # rnn_output : [batch_size, rnn_size] ; hidden : [num_layer, batch_size, rnn_size]
            rnn_output, hidden = self.rnn_cell(rnn_inputs, hidden)
            # prob_word : [batch_size, num_vocab]
            prob_word = self.output(rnn_output)
            if combine_knowledge == True:
                # copy_dist : [batch_size, num_vocab]
                copy_prob = torch.zeros(batch_size, self.num_vocab, device = src_context_output.device)
                copy_prob = torch.scatter_add(input = copy_prob, 
                                              dim = 1, 
                                              index = KG_word_seq, src = attn_KG_scores.permute(1, 0))
                # gen_dist_trans_input : [batch_size, emb_size + KG_rnn_size + rnn_size]
                gen_dist_trans_input = torch.cat([tgt_word_emb[:, word_index], attn_KG_word, rnn_output], dim = 1)
                gen_dist = self.gen_dist_trans(gen_dist_trans_input)
                gen_dist = torch.sigmoid(gen_dist)
                # combined_prob_word: [batch_size, num_vocab]
                combined_prob_word = prob_word * gen_dist + (1 - gen_dist) * copy_prob
            else:
                combined_prob_word = prob_word
            # logit_word : [batch_size, num_vocab]
            logit_word = combined_prob_word.log()
            # coverage_score : [batch_size]
            coverage_score = cos_sim(last_hidden, hidden[-1])
            coverage_score = F.relu(coverage_score)
            
            logit_word_list.append(logit_word)
        # logit : [batch_size, max_tgt_len, num_vocab]
        logit = torch.stack(logit_word_list, dim = 1)

        return logit
