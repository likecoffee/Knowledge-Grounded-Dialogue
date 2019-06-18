import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import PAD_TOKEN, pad_tensor_by_length, Identity

class WordEncoderRNN(nn.Module):
    def __init__(self, word_embedding, rnn_size, num_layers, dropout = 0):
        assert rnn_size % 2 == 0 # assert the rnn_size is even
        super(WordEncoderRNN, self).__init__()
        self.word_embedding = word_embedding
        self.num_vocab, self.emb_size = word_embedding.weight.size()
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.word_rnn = nn.GRU(input_size = self.emb_size,
                       hidden_size = self.rnn_size // 2,
                       num_layers = self.num_layers,
                       batch_first = True,
                       dropout = self.dropout,
                       bidirectional = True)

    def forward(self, src_sents, src_len):
        """
        Args:
            src_sents (LongTensor) : [num_src_sent, max_src_len]
            src_len (LongTensor) : [num_src_sent]
        Returns:
            outputs (LongTensor) : [num_src_sent, max_src_len, rnn_size]
            hidden (LongTensor) : [num_layers*num_directions, rnn_size]
        """
        num_src_sent, max_src_len = src_sents.size()

        # sort the sents by descreasing order, borrowed from https://github.com/ctr4si/
        src_len_sorted, src_sorted_indices = src_len.sort(descending = True)
        # [num_src_sent, max_src_len]
        src_sents_sorted = src_sents.index_select(0, src_sorted_indices)

        embedded = self.word_embedding(src_sents_sorted)
        rnn_input = pack_padded_sequence(embedded, src_len_sorted, batch_first=True)
        # initialize the rnn hidden
        param = next(self.parameters())
        hidden = param.new(self.num_layers*2, num_src_sent, self.rnn_size//2).zero_()
        # accelerate the RNN model by using cuDNN
        self.word_rnn.flatten_parameters()
        # outputs : [num_src_sent, max_src_len, rnn_size]
        # hidden : [num_layers * 2, num_src_sent, rnn_size // 2]
        outputs, hidden = self.word_rnn(rnn_input, hidden)
        outputs, outputs_len = pad_packed_sequence(outputs, batch_first = True)

        #reorder the outputs
        _, inverse_src_sorted_indices = src_sorted_indices.sort()
        outputs = outputs.index_select(0, inverse_src_sorted_indices)
        hidden = hidden.index_select(1, inverse_src_sorted_indices)
        
        return outputs, hidden 

class ContextEncoderRNN(nn.Module):
    def __init__(self, word_embedding, word_rnn_size, word_num_layers,
                       context_rnn_size, context_num_layers, dropout = 0.0):
        super(ContextEncoderRNN, self).__init__()
        self.word_encoder = WordEncoderRNN(word_embedding = word_embedding, 
                                           rnn_size = word_rnn_size, 
                                           num_layers = word_num_layers, 
                                           dropout = dropout)
        self.word_rnn_size = word_rnn_size
        self.context_rnn_size = context_rnn_size
        self.context_num_layers = context_num_layers
        self.dropout = dropout

        self.context_rnn = nn.GRU(input_size = self.word_rnn_size,
                                 hidden_size = self.context_rnn_size,
                                 num_layers = self.context_num_layers,
                                 batch_first = True,
                                 dropout = self.dropout)

    def recover_utterance_form(self, flattened_tensor, context_start, 
                                utterance_len, dim):
        max_utterance_len = max(utterance_len)
        utterance_list = [flattened_tensor.narrow(dim, s, l)
                               for s, l in zip(context_start, utterance_len)]    
        padded_utterance_list = [pad_tensor_by_length(utterance, max_utterance_len, dim)
                                for utterance in utterance_list]
        padded_utterance_outputs = torch.stack(padded_utterance_list, dim)

        return padded_utterance_outputs


    def forward(self, sents, word_len, utterance_len):
        """
        Args:
            sents (LongTensor) : [num_sent, max_len]
            word_len (LongTensor) : [num_sent]
            utteracen_len (LongTensor) : [batch_size]
        Returns:
            context_outputs (FloatTensor) : [num_sent, context_rnn_size]
            word_outputs (FloatTensor) :[num_sent, max_sent_len, word_rnn_size]
        """
        batch_size = utterance_len.size(0)
        utterance_len = utterance_len * 2
        #word_outputs : [num_sent, max_word_len, word_rnn_size]
        word_outputs,_ = self.word_encoder(sents, word_len)
        #context_inputs : [num_sent, word_rnn_size] : Aggregation of word outputs
        context_inputs = word_outputs.sum(dim = 1) / word_len.unsqueeze(1).float()
        # mannually calculate the context start point in pack_packed way
        # context_start: [batch_size]
        utterance_pos = torch.cat([utterance_len.new(1).zero_(),
                                 utterance_len[:-1]], dim = 0)
        utterance_start = torch.cumsum(utterance_pos, dim = 0)
        # padded_word_outputs : [batch_size , max_num_utterance, word_rnn_size]
        padded_context_inputs = self.recover_utterance_form(context_inputs, 
                                                            utterance_start, 
                                                            utterance_len, 
                                                            dim = 0)

        # utterance_len_sorted : [batch_size],  context_sorted_indices : [batch_size]
        utterance_len_sorted, context_sorted_indices = utterance_len.sort(descending = True)
        # word_outputs_sorted : [batch_size, max_num_sent, word_rnn_size]
        padded_context_inputs_sorted = padded_context_inputs.index_select(0, context_sorted_indices)
        # pack_padded_sequence for context RNN
        context_rnn_input = pack_padded_sequence(padded_context_inputs_sorted, utterance_len_sorted, batch_first=True)
        # initialize the rnn hidden
        param = next(self.parameters())
        hidden = param.new(self.context_num_layers, batch_size, self.context_rnn_size).zero_()
        # accelerate the RNN model by using cuDNN
        self.context_rnn.flatten_parameters()
        # outputs : [batch_size, max_num_utterance, context_rnn_size]
        # hidden : [num_layers * 2, max_num_sent, rnn_size]
        context_outputs, hidden = self.context_rnn(context_rnn_input, hidden)
        context_outputs, outputs_len = pad_packed_sequence(context_outputs, batch_first = True)

        # reorder the outputs
        _, inverse_context_indices = context_sorted_indices.sort()
        # context_outputs : [batch_size, max_num_utterance, context_rnn_size]
        context_outputs = context_outputs.index_select(0, inverse_context_indices)
        flatened_context_output_list = [context_outputs[batch_index, :l]
                                          for batch_index, l in enumerate(utterance_len)]
        # flattened_context_outputs : [num_sent, context_rnn_size]
        flatened_context_outputs = torch.cat(flatened_context_output_list, dim=0)
        assert flatened_context_outputs.size(0) == sents.size(0)
        assert word_outputs.size(0) == sents.size(0)

        return flatened_context_outputs[0::2], word_outputs[0::2], word_len[0::2]
