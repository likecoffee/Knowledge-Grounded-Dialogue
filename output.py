import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Identity, NEAR_INF

class OutputLayer(nn.Module):
    """ Takes in final states and returns distributions over vocabulary candidates."""
    def __init__(self, embedding, input_dim, num_softmax = 1, dropout = 0, padding_idx = -1):
        """Initialize output layer.
        Args:
            embedding: (nn.Module)  : the word embedding module
            input_dim: (int)        : input dimension of OutputLayer
            num_softmax: (int)      : (default 1) number of softmaxes to calculate.
                                      see arxiv.org/abs/1711.03953 for more info
                                      increasing this can add more expressiveness
                                      of prediction.
            dropout: (float)        : (defaul 0.0) dropout ratio
            padding_idx (int)       : (default -1) model should output a large negative 
                                      number for score at this index, if set to -1 ,
                                      it is disabled. if >= 0, always outputs -1e20 at 
                                      this index.
        """
        super().__init__()
        self.embedding = embedding
        self.num_vocab, self.emb_size = embedding.weight.size()
        self.input_dim = input_dim
        self.num_softmax = num_softmax
        self.dropout = dropout
        self.padding_idx = padding_idx

        if self.num_softmax > 1:
            self.prior_trans = nn.Linear(self.input_dim, self.num_softmax, bias = False)
            self.latent_trans = nn.Sequential(
                                nn.Linear(self.input_dim, self.num_softmax * self.emb_size),
                                nn.Tanh(),
                                nn.Dropout(dropout)
                                )
        else:
            if self.input_dim != self.emb_size:
                self.output_trans = nn.Sequential( 
                                    nn.Linear(self.input_dim, self.emb_size, biase = True),
                                    nn.Dropout(dropout)
                                    )
            else:
                self.output_trans = nn.Sequential(
                                    Identity(),
                                    nn.Dropout(dropout)
                                    )

    def forward(self, input):
        """Compute scores from input.
        Args:
            input (FloatTensor) : (batch_size, input_dim)
        Regurns:
            scores (FloatTensor): (batch_size, num_vocab)
        """
        #assert the dimension of input is input_dim
        assert input.size(-1) == self.input_dim
        # mixture of softmax
        if self.num_softmax > 1:
            batch_size = input.size(0)
            # [batch_size, num_softmax * emb_size]
            latent = self.latent_trans(input)
            # [batch_size * num_softmax, emb_size]
            latent = latent.view(-1, self.emb_size)
            # [batch_size * num_softmax, num_vocab]
            logit = F.linear(latent, self.embedding.weight)
            # [batch_size * num_softmax, num_vocab]
            prob = F.softmax(logit, dim = 1)
            # [batch_size, num_softmax, num_vocab]
            prob = prob.view(batch_size, self.num_softmax, -1)
            
            # [batch_size, num_softmax]
            prior_logit = self.prior_trans(input)
            # [batch_size, num_softmax]
            prior_prob = F.softmax(prior_logit, dim = 1)
            # [batch_size, num_softmax, 1]
            prior_prob = prior_prob.unsqueeze(2)
            
            # [batch_size, num_vocab] 
            prob = torch.mul(prob, prior_prob).sum(1)
        # regular softmax
        else:
            # [batch_size, emb_size]
            output = self.output_trans(input)
            # [batch_size, num_vocab]
            scores = F.linear(output, self.embedding.weight)
            # [batch_size, num_vocab]
            prob = F.softmax(scores)
        
        return prob
