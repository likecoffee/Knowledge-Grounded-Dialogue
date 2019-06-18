# -*- coding: utf-8 -*-
# file: attention.py
# author: JiachenDu <jacobvan199165@gmail.com>
# Copyright (C) 2018. All right Reserved

import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, context_dim, hidden_dim, type="mlp"):
        super(Attention,self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.type = type
        if type== "mlp":
            self.attn = nn.Linear(self.hidden_dim + self.context_dim, hidden_dim)
            self.v = nn.Linear(self.hidden_dim, 1, bias = False)
        elif type == "bilinear":
            self.bilinear = nn.Bilinear(self.context_dim, self.hidden_dim, 1, bias=False)
        elif type == 'dot':
            None
        else:
            raise  Exception("Wrong Atten Type")
        self.init_weight()

    def __repr__(self):
        s = "type = {}, context_dim= {}, hidden_dim= {}".format(self.type, self.context_dim, self.hidden_dim)
        return s

    def init_weight(self):
        if self.type == "mlp":
            nn.init.xavier_normal_(self.attn.weight)
            nn.init.uniform_(self.attn.bias,-0.1,0.1)
            nn.init.xavier_normal_(self.v.weight)
        elif self.type == "bilinear":
            nn.init.xavier_normal_(self.bilinear.weight)
        else:
            None

    def bert_score(self, target, context):
        None

    def mlp_score(self, target, context, attention_mask):
        """
        target: FloatTensor [seq_len * batch_size * hidden_size]
        contex: FloatTensor [seq_len * batch_size * hidden_size]
        """
        seq_len, batch_size ,_ = context.size()
        attn_input= torch.cat([target,context],dim=2)   # [seq_len * batch_size * 2 hidden_size]
        energy = torch.tanh(self.attn(attn_input))      # [seq_len * batch_size * hidden_size]
        attn_score = self.v(energy)                     # [seq_len * batch_size * 1]
        attn_score = attn_score.squeeze(2)              # [seq_len * batch_size]
        attn_score = attn_score + attention_mask
        attn_score = F.softmax(attn_score, dim = 0)     # [seq_len * batch]
        return attn_score
    
    def bilinear_score(self, target, context, attention_mask):
        """
        target: FloatTensor [seq_len * batch_size * hidden_size]
        contex: FloatTensor [seq_len * batch_size * hidden_size]
        """
        attention_score_list = []
        seq_len, batch_size,_ = context.size()
        for seq_i in range(seq_len):
            context_i = context[seq_i]
            target_i = target[seq_i]
            attention_score_i = self.bilinear(context_i, target_i)
            attention_score_i = attention_score_i.squeeze(1)
            attention_score_list.append(attention_score_i)
        attention_score = torch.stack(attention_score_list, dim = 0)
        attention_score = attention_score + attention_mask
        attention_score = F.softmax(attention_score, dim=0)

        return attention_score    

    def dot_score(self, target, context, attention_mask):
        """
        target: FloatTensor [seq_len * batch_size * hidden_size]
        contex: FloatTensor [seq_len * batch_size * hidden_size]
        """
        attention_score_list = []
        seq_len, batch_size,_ = context.size()
        for seq_i in range(seq_len):
            context_i = context[seq_i]
            target_i = target[seq_i]
            attention_score_i = torch.mm(context_i,target_i.transpose(0,1)).diag(0)
            attention_score_list.append(attention_score_i)
        attention_score = torch.stack(attention_score_list, dim = 0)
        attention_score = attention_score + attention_mask
        attention_score = F.softmax(attention_score, dim=0)

        return attention_score
 

    def forward(self, target, context_key, context_value, attention_mask=None):
        """
        Args:
            target : FloatTensor        :[batch_size, hidden_size]
            context_key : FloatTensor   :[seq_len, batch_size, hidden_size]
            context_value : FloatTensor :[seq_len, batch_size, hidden_size]
        Returns:
            attn_context : FloatTensor  :[batch_size, hidden_size]
            attn_score : FloatTensor    :[seq_len, batch_size]
        """
        batch_size, hidden_size = target.size()
        seq_len, batch_size_, hidden_size_ = context_key.size()
        assert batch_size == batch_size_
        assert hidden_size == hidden_size_
        
        if attention_mask is None:
            attention_mask = target.new(seq_len, batch_size).zero_()

        if self.type == 'bilinear' or self.type == "dot":
            target = target.unsqueeze(0).repeat(seq_len, 1, 1)                          # [seq_len, batch_size , hidden_size]
            if self.type == 'bilinear':
                attn_score = self.bilinear_score(target, context_key, attention_mask)   # [seq_len , batch_size]
            else:
                attn_score = self.dot_score(target, context_key, attention_mask)
            attn_context = attn_score.unsqueeze(2) * context_value                      # [seq_len , batch_size , hidden_size]
        elif self.type == "mlp":
            target = target.unsqueeze(0).repeat(seq_len, 1, 1)                          # [seq_len , batch_size , hidden_size]
            attn_score = self.mlp_score(target, context_key, attention_mask)            # [seq_len , batch_size]
            attn_context = attn_score.unsqueeze(2) * context_value                      # [seq_len , batch_size , hidden_size]
        else:
            raise Exception("Wrong Atten Type")
        # attn_context : [batch_size, hidden_size]
        attn_context = attn_context.sum(dim = 0)

        return attn_context, attn_score
