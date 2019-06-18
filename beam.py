    def beam_decode(self, topk_words_array, topk_beam_indices_array, last_index):
        """
            topk_words_array (Long-typed Numpy array)       : [max_tgt_word_len, batch_size, beam_size]
            topk_beam_indices_array (Long-typed Numpy array): [max_tgt_word_len, batch_size, beam_size]
        """
        assert topk_words_array.size(0) == topk_beam_indices_array.size(0)
        max_tgt_word_len = topk_words_array.size(0)
        selected_word_array_list = []
        for index in range(max_tgt_word_len-1, -1, -1):
            # selected_word_array : [batch_size, 1]
            selected_word_array = torch.gather(input = topk_words_array[index], 
                                               dim = 1, 
                                               index = last_index)
            selected_word_array_list.append(selected_word_array.squeeze(1))
            # last_index : [batch_size, 1]
            last_index = torch.gather(input = topk_beam_indices_array[index], 
                                      dim = 1, 
                                      index = last_index)
        # whole_selected_word_array : [batch_size, max_tgt_word_len] 
        whole_selected_word_array  = torch.stack(selected_word_array_list, dim = 1)
        whole_selected_word_array = torch.flip(whole_selected_word_array, dims = (1,))

        return whole_selected_word_array

    def beam_generate(self, beam_size, src_context_output, src_word_output, src_word_len, KG_word_output, 
                            KG_word_len, KG_word_seq, init_tgt_word_input, max_tgt_word_len):
        """Generate the sequence given source sequence.
        Args:
            beam_size (int)                  :  The size of beam search
            src_context_output (FloatTensor) :  (batch_size, context_size)
            src_word_output (FloatTensor)    :  (batch_size, src_word_len, word_size)
            src_word_len (LongTensor)        :  (batch_size)
            KG_word_output (FloatTensor)     :  (batch_size, KG_max_word_len, KG_word_size)
            KG_word_len (LongTensor)         :  (batch_size)
            KG_word_seq (LongTensor)         :  (batch_size, src_max_word_len)
            init_tgt_word_input (LongTensor) :  (batch_size)
            max_tgt_word_len (int)           :  the maximum lenght of target sequence
        Regurns:
            logit (FloatTensor) :   (batch_size, num_vocab)
        """

        batch_size = init_tgt_word_input.size(0)
        src_context_output, src_word_output, src_word_mask, KG_word_output,\
            KG_word_mask, hidden = self.prepare_generate(src_context_output, src_word_output,
                                                         src_word_len, KG_word_output, KG_word_len, beam_size)
        # tgt_word : [batch_size * beam_size]
        tgt_word = init_tgt_word_input.unsqueeze(1).expand(batch_size, beam_size)
        tgt_word = tgt_word.contiguous()
        tgt_word = tgt_word.view(batch_size * beam_size)
        # total_logit : [batch_size, beam_size]
        total_logit = torch.zeros(batch_size, beam_size, device = tgt_word.device)
        topk_words_list = []
        topk_beam_indices_list = []
        for word_index in range(max_tgt_word_len):
            # hidden : [num_layer, batch_size * beam_size, rnn_size]
            # logit_word : [batch_size * beam_size, num_vocab]
            hidden, logit_word = self.forward_step(src_context_output, src_word_output, src_word_mask,
                                           KG_word_output, KG_word_seq, KG_word_mask, hidden, tgt_word)
            # logit_word : [batch_size, beam_size, num_vocab]
            logit_word = logit_word.view(batch_size, beam_size, logit_word.size(-1))
            # current_logit_summed : [batch_size, beam_size, num_vocab]
            current_logit_summed = total_logit.unsqueeze(2) +  logit_word
            # current_logit_summed : [batch_size, beam_size * num_vocab]
            current_logit_summed = current_logit_summed.view(batch_size, beam_size * current_logit_summed.size(-1))
            # total_logit: [batch_size, beam_size]
            # topk_words_indices : [batch_size, beam_size]
            total_logit, topk_words_indices = current_logit_summed.topk(beam_size, dim = 1)
            # topk_words : [batch_size, beam_size]
            # topk_beam_indices : [batch_size, beam_size]
            topk_words = topk_words_indices % self.num_vocab
            topk_beam_indices = topk_words_indices // self.num_vocab
            topk_words_list.append(topk_words.detach())
            topk_beam_indices_list.append(topk_beam_indices.detach())

        # whole_topk_words : [max_tgt_word_len, batch_size, beam_size]
        # whole_topk_beam_indices: [max_tgt_word_len, batch_size, beam_size]
        whole_topk_words = torch.stack(topk_words_list)
        whole_topk_beam_indices = torch.stack(topk_beam_indices_list)
        # last_index : [batch_size, 1]
        _,last_index = total_logit.topk(1, dim = 1)
        last_index = last_index 

        tgt_word_sequence = self.beam_decode(topk_words_array = whole_topk_words,
                                             topk_beam_indices_array = whole_topk_beam_indices,
                                             last_index = last_index)
        return tgt_word_sequence
