import json
import pickle
import numpy as np
from functools import reduce
from torch.utils.data import Dataset, DataLoader

class DialogDataset(Dataset):
    def __init__(self, data_file_name, pad_token, max_text_num_word= 40, 
                max_checkedsent_num_word = 40, max_num_utterance_pair=6):
        with open(data_file_name, "rb") as f:
            self.raw_dialog_list = pickle.load(f)
        self.pad_token = pad_token
        self.max_text_num_word = max_text_num_word
        self.max_checkedsent_num_word = max_checkedsent_num_word
        self.max_num_utterance_pair = max_num_utterance_pair

    def __len__(self):
        return len(self.raw_dialog_list)

    def _pad_utterance(self, utterance_idx, max_text_num_word, max_checkedsent_num_word):
        text_idx = utterance_idx['text']
        text_num_word = len(text_idx)
        padded_text_idx = text_idx[:max_text_num_word] + [self.pad_token]*(max_text_num_word - text_num_word)
        if utterance_idx.get('checked_sentence') is not None:
            checkedsent_idx = utterance_idx['checked_sentence']
            checkedsent_num_word = len(checkedsent_idx)
            padded_length = (max_checkedsent_num_word - checkedsent_num_word)
            padded_checkedsent_idx = checkedsent_idx[:max_checkedsent_num_word] + [self.pad_token] * padded_length 
        else:
            padded_checkedsent_idx = None
            checkedsent_num_word = 0

        if text_num_word > max_text_num_word:
            text_num_word = max_text_num_word
        if checkedsent_num_word > max_checkedsent_num_word:
            checkedsent_num_word = max_checkedsent_num_word

        return padded_text_idx, padded_checkedsent_idx, text_num_word, checkedsent_num_word
    
    def __getitem__(self, idx):
        dialog_item = self.raw_dialog_list[idx]
        chosen_topic_idx = dialog_item["chosen_topic_word_id"]
        # padding utterances
        utterance_idx_list = dialog_item["utterance_word_id_list"][:2*self.max_num_utterance_pair]
        padded_text_idx_list = []
        padded_checkedsent_idx_list = []
        text_num_word_list = []
        checkedsent_num_word_list = []
        for utterance_idx in utterance_idx_list:
            padded_text_idx, padded_checkedsent_idx, text_num_word, checkedsent_num_word = self._pad_utterance(utterance_idx,
                                                                                                         self.max_text_num_word,
                                                                                                         self.max_checkedsent_num_word)
            padded_text_idx_list.append(padded_text_idx)
            text_num_word_list.append(text_num_word)
            if padded_checkedsent_idx is not None:
                padded_checkedsent_idx_list.append(padded_checkedsent_idx)
                checkedsent_num_word_list.append(checkedsent_num_word)
        # generating target
        num_utterance_pair = len(utterance_idx_list) // 2
        source_text_idx_list = padded_text_idx_list
        source_text_idx_length =  text_num_word_list
        input_target_text_idx_list =  padded_text_idx_list[1::2] # only the even line is the response
        target_text_idx_list = [input_target_text_idx[1:] + [self.pad_token] 
                                for input_target_text_idx in input_target_text_idx_list]
        target_text_idx_length = text_num_word_list[1::2]
        checkedsent_idx_list =  padded_checkedsent_idx_list
        checkedsent_idx_length = checkedsent_num_word_list
        sample = (num_utterance_pair, source_text_idx_list, source_text_idx_length, 
                  target_text_idx_list, input_target_text_idx_list, target_text_idx_length,
                  checkedsent_idx_list, checkedsent_idx_length)

        return sample
    

def get_loader(data_file_name, pad_token, max_text_num_word= 40, 
        max_checkedsent_num_word = 40,  max_num_utterance_pair=6, batch_size=32, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    def collate_fn(data):
        """
        Collate list of data in to batch

        Args:
            data: list of tuple(target_text, input_text, checked_text, input_length, checked_length, utterance_length)
        Return:
            Batch of each feature
            - source (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - target (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - conversation_length (np.array): [batch_size]
            - source_length (LongTensor): [batch_size, max_conversation_length]
        """
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[0], reverse=True)

        # Separate
        num_utterance_pair, source_text_idx_list, source_text_idx_length,\
                target_text_idx_list, input_target_text_idx_list, target_text_idx_length,\
                checkedsent_idx_list, checkedsent_idx_length = zip(*data)

        batch_dict = dict(num_utterance_pair = num_utterance_pair, source = source_text_idx_list, 
                      source_len = source_text_idx_length, target = target_text_idx_list, 
                      input_target = input_target_text_idx_list, target_len = target_text_idx_length,
                      checked_sent = checkedsent_idx_list, checked_sent_len = checkedsent_idx_length)

        for key in batch_dict.keys():
            if key != "num_utterance_pair":
                batch_dict[key] = reduce(lambda x,y : x+y, batch_dict[key])

        return batch_dict

    dataset = DialogDataset(data_file_name, pad_token, max_text_num_word, 
                            max_checkedsent_num_word, max_num_utterance_pair)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last = True)

    return data_loader
