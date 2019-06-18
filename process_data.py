import os
import json
import spacy
import pickle
import logging
import numpy as np
from os import path
from ipdb import launch_ipdb_on_exception, set_trace

from vocabulary import Vocabulary, load_embedding

logging.basicConfig(format="%(asctime)s:\t%(message)s", level=logging.INFO)
nlp_processor = spacy.load("en", disable=["parser", "tagger", "ner"])
spacy_stopwords = set([word.lower() for word in spacy.lang.en.stop_words.STOP_WORDS] + ["'s","'m"])

def to_pickle(obj, pickle_name):
    with open(pickle_name, "wb") as wf:
        pickle.dump(obj, wf)

def tokenize_text(text, prefix, vocab = None, construct_vocab = False):
    word_list = [word.text.lower() for word in nlp_processor(text)]
    word_list = [prefix] + word_list + ["<EOS>"]
    if construct_vocab:
        vocab.add_word_list(word_list)
    return word_list

def process_dialog_text(dialog_dict, vocab = None, construct_vocab = False):
    dialog_chosen_topic = dialog_dict['chosen_topic']
    dialog_chosen_topic_word_list = tokenize_text(dialog_chosen_topic, "<TOPIC>",vocab, construct_vocab)
    # process utterances
    dialog_utterances_list = dialog_dict['dialog']
    processed_utterances_list = []

    if dialog_utterances_list[0]['speaker'].endswith("Wizard"):
        starting_point = 0
        message_word_list = dialog_chosen_topic_word_list
        if len(dialog_utterances_list) % 2 == 0:
            num_utterance = len(dialog_utterances_list) -1 
        else:
            num_utterance = len(dialog_utterances_list) 
        #####################
        #   A   A   topic
        #0  W   W   response
        #1  A   A   message
        #2  W   W   response
        #----------
        #3  A       message
        #   4   3   length
        #######################
    else:
        starting_point = 1
        message_word_list = tokenize_text(dialog_utterances_list[0]["text"],"<MESSAGE>",vocab, construct_vocab)
        message_word_list = dialog_chosen_topic_word_list + message_word_list
        num_utterance = len(dialog_utterances_list)  // 2 * 2
        
    processed_utterances_list.append(dict(utterance_type="message", text=message_word_list))
    # get the number of utterance

    for utterance_index in range(starting_point, num_utterance):
        utterance = dialog_utterances_list[utterance_index]
        text = utterance['text']
        if utterance['speaker'].endswith("Wizard"):
            text_word_list = tokenize_text(text, "<RESPONSE>" ,vocab, construct_vocab)
            if utterance.get("checked_sentence"):
                checked_sentence = list(utterance['checked_sentence'].values())[0]
                if checked_sentence != "no_passages_used" and len(checked_sentence) != 0:
                    checked_sentence_word_list = tokenize_text(checked_sentence, "<CHECKED_SENT>", vocab, construct_vocab)
                else:
                    checked_sentence_word_list = ["<CHECKED_SENT>", "<EOS>"]
            else:
                checked_sentence_word_list = ["<CHECKED_SENT>", "<EOS>"]
            processed_utterances_list.append(dict(utterance_type = "response", text = text_word_list,
                                                     checked_sentence = checked_sentence_word_list))
        else:
            text_word_list = tokenize_text(text, "<MESSAGE>" ,vocab, construct_vocab)
            processed_utterances_list.append(dict(utterance_type="message", text=text_word_list))
    assert len(processed_utterances_list) % 2 == 0 
    num_utterance_pair = len(processed_utterances_list) // 2
    dialog_item = dict()
    dialog_item['utterance_list'] = processed_utterances_list
    dialog_item['chosen_topic'] = dialog_chosen_topic_word_list
    dialog_item['num_utterance_pair'] = num_utterance_pair

    return dialog_item
        
def process_dataset_text(original_dialog_list, construct_vocab = False):
    if construct_vocab:
        vocab  = Vocabulary()
    else:
        vocab = None

    dialog_text_list = []
    for dialog_dict in original_dialog_list:
        utterance_text_list = process_dialog_text(dialog_dict, vocab, construct_vocab)
        dialog_text_list.append(utterance_text_list)
    
    if construct_vocab:
        return dialog_text_list, vocab
    else:
        return dialog_text_list

def convert_to_id(dialog_list, vocab):
    for dialog in dialog_list:
        dialog_word_id_list = []
        for utterance in dialog["utterance_list"]:
            text_id = vocab.convert_word_list(utterance["text"], mode="truncated")
            if utterance['utterance_type'] == "response":
                checked_sentence_id = vocab.convert_word_list(utterance["checked_sentence"], mode="truncated")
                dialog_word_id_list.append(dict(utterance_type="response", text=text_id, 
                                                checked_sentence = checked_sentence_id))
            else:
                dialog_word_id_list.append(dict(utterance_type="message", text=text_id))
        chosen_topic_word_id = vocab.convert_word_list(dialog["chosen_topic"], mode="truncated")
        dialog["chosen_topic_word_id"] = chosen_topic_word_id
        dialog["utterance_word_id_list"] = dialog_word_id_list
        dialog["utterance_length_list"] = [len(utterance_word_id['text']) for utterance_word_id in dialog_word_id_list]
        dialog["max_utterance_length"] = np.max(dialog['utterance_length_list'])

    return dialog_list

def process_wow_dataset(json_file_name, pickle_file_name, vocab, construct_vocab = False, truncated_vocab_number = 20000):
    with open(json_file_name) as f: 
        original_dialog_list = json.load(f)

    if construct_vocab:
        # when training, we use training data to construct vocabulary
        dialog_text_list, vocab = process_dataset_text(original_dialog_list, construct_vocab)
        vocab.truncate_dictionary(truncated_vocab_number)
        logging.info("Vocabulary constructed done")
    else:
        # valid set and test are not involeved in constructing vocabulary
        dialog_text_list = process_dataset_text(original_dialog_list, construct_vocab)

    # convert word list to id list
    whole_dataset_list = convert_to_id(dialog_text_list, vocab) 
    
    # save the raw data
    to_pickle(whole_dataset_list, pickle_file_name)

    return whole_dataset_list, vocab

def main():
    logging.info("Processing training data")
    if not path.exists("./processed_data"):
        os.mkdir("./processed_data")
    train_word_id_list, vocab = process_wow_dataset(json_file_name = "./data/train.json", 
                                                    pickle_file_name = "./processed_data/train_data.pkl", 
                                                    vocab = None, 
                                                    construct_vocab = True, 
                                                    truncated_vocab_number = 20000)
    logging.info("Processing validation data")
    valid_word_id_list,_ = process_wow_dataset(json_file_name = "./data/valid_random_split.json", 
                                               pickle_file_name = "./processed_data/valid_data.pkl", 
                                               vocab = vocab, 
                                               construct_vocab = False)
    logging.info("Processing test data")
    test_word_id_list,_ = process_wow_dataset(json_file_name = "./data/test_random_split.json", 
                                              pickle_file_name = "./processed_data/test_data.pkl", 
                                              vocab = vocab, 
                                              construct_vocab = False)
    logging.info("Saving Vocabulary and Loading Embedding")
    to_pickle(vocab, "./processed_data/vocab.pkl")
    embedding = load_embedding(vocab.truncated_word_id, 
                               embedding_file_name = "/home/dujiachen/local/embeddings/glove.840B.300d.txt",
                               embedding_size = 300)
    with open("./processed_data/embedding.pkl", "wb") as wf:
        pickle.dump(embedding, wf)

if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
