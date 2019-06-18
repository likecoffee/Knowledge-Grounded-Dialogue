# -*- coding: utf-8 -*-
# file: HRED.py
# author: JiachenDu <jacobvan199165@gmail.com>
# Copyright (C) 2018, 2019. All right Reserved

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from encoder import WordEncoderRNN,ContextEncoderRNN
from decoder import DecoderRNN
from utils import PAD_TOKEN

class HRED(nn.Module):
    def __init__(self, num_vocab, emb_size, enc_word_rnn_size, enc_word_num_layers, enc_context_rnn_size, 
                enc_context_num_layers, KG_word_rnn_size, KG_word_num_layers, dec_rnn_size, dec_num_layers,
                 dec_num_softmax, dropout, pre_embedding = None):
        super(HRED, self).__init__()
        self.word_embedding = nn.Embedding(num_vocab, emb_size)
        if pre_embedding is not None:
            self.word_embedding.weight = nn.Parameter(self.word_embedding.weight.data.new(pre_embedding))
        self.context_encoder = ContextEncoderRNN(word_embedding = self.word_embedding,
                                                 word_rnn_size = enc_word_rnn_size,
                                                 word_num_layers = enc_word_num_layers,
                                                 context_rnn_size = enc_context_rnn_size,
                                                 context_num_layers = enc_context_num_layers,
                                                 dropout = dropout)
        self.KG_encoder = WordEncoderRNN(word_embedding = self.word_embedding,
                                         rnn_size = KG_word_rnn_size,
                                         num_layers = KG_word_num_layers,
                                         dropout = dropout)
        self.decoder = DecoderRNN(word_embedding = self.word_embedding,
                                  src_word_size = enc_word_rnn_size,
                                  src_context_size = enc_context_rnn_size,
                                  KG_word_size = KG_word_rnn_size,
                                  rnn_size = dec_rnn_size,
                                  num_layers = dec_num_layers,
                                  num_softmax = dec_num_softmax,
                                  dropout = dropout)

    def forward(self, src_sents, src_word_len, src_utterance_len, KG_sents, KG_word_len, tgt_word_input, combine_knowledge):
        """
            src_sents (LongTensor)          : [src_num_sent, src_word_len]
            src_word_len (LongTensor)       : [src_num_sent]
            src_utteracen_len (LongTensor)  : [batch_size]
            KG_sents (LongTensor)          :  [KG_num_sent, KG_word_len]
            KG_word_len (LongTensor)       :  [KG_num_sent]
            tgt_word_input (LongTensor)     : [tgt_num_sent, tgt_word_len]
        """
        # src_context_outputs : [tgt_num_sents, src_context_rnn_size]
        # src_word_outputs : [tgt_num_sents, max_src_word_len, src_word_rnn_size]
        src_context_outputs, src_word_outputs, src_word_len = self.context_encoder(src_sents, src_word_len, src_utterance_len)
        # KG_outputs : [KG_num_sent, max_KG_word_len, KG_word_rnn_size]
        KG_word_output, _ = self.KG_encoder(KG_sents, KG_word_len)
        # logit : [batch_size, tgt_word_len, num_vocab] 
        # converage : [batch_size, tgt_word_len]
        logit = self.decoder(src_context_outputs, src_word_outputs, src_word_len, 
                             KG_word_output, KG_word_len, KG_sents, tgt_word_input, combine_knowledge) 

        return logit
    
    def greedy_generate(self, src_sents, src_word_len, src_utterance_len, KG_sents, 
                        KG_word_len, max_tgt_word_len, initial_word_idx, combine_knowledge,
                        temperature, topk, topp):
        """
            src_sents (LongTensor)          : [src_num_sent, src_word_len]
            src_word_len (LongTensor)       : [src_num_sent]
            src_utteracen_len (LongTensor)  : [batch_size]
            max_tgt_len (int)               : The maxium length of target sequence
        """
        # src_context_outputs : [tgt_num_sents, src_context_rnn_size]
        # src_word_outputs : [tgt_num_sents, src_word_len, src_word_rnn_size]
        src_context_output, src_word_output, src_word_len = self.context_encoder(src_sents, src_word_len, src_utterance_len)
        KG_word_output, _ = self.KG_encoder(KG_sents, KG_word_len)
        num_sents = src_context_output.size(0)
        # init_tgt_word_input : [tgt_num_sents]
        init_tgt_word_input = torch.ones(num_sents, device = src_sents.device).long() * initial_word_idx
        generated_tgt_word = self.decoder.greedy_generate(src_context_output, src_word_output, src_word_len, 
                                                          KG_word_output, KG_word_len, KG_sents, init_tgt_word_input, 
                                                          max_tgt_word_len, combine_knowledge, temperature, topk, topp)

        return generated_tgt_word
