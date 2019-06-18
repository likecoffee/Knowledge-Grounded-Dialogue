import pickle
import logging
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from ipdb import launch_ipdb_on_exception

from HRED import HRED
from HRED_VAE import HRED_VAE
from data_loader import get_loader
from train import train_model, eval_model
from vocabulary import Vocabulary

def main():
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s\t%(message)s")
    with open("./processed_data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("./processed_data/embedding.pkl", "rb") as f:
        embedding = pickle.load(f)
    model_opt = dict(num_vocab = vocab.truncated_length,
                     emb_size = 300,
                     enc_word_rnn_size = 512,
                     enc_word_num_layers = 2,
                     enc_context_rnn_size = 512,
                     enc_context_num_layers = 2,
                     KG_word_rnn_size = 512,
                     KG_word_num_layers = 1,
                     dec_rnn_size = 512,
                     dec_num_layers = 2,
                     dec_num_softmax = 4,
                     latent_size = 300,
                     dropout = 0.3,
                     pre_embedding = embedding)
    model = HRED_VAE(**model_opt)
    pad_token = vocab.truncated_word_id["<EOS>"]
    train_data_loader = get_loader("./processed_data/train_data.pkl", pad_token = pad_token, batch_size = 20)
    valid_data_loader = get_loader("./processed_data/valid_data.pkl", pad_token = pad_token, batch_size = 20)
    test_data_loader = get_loader("./processed_data/test_data.pkl", pad_token = pad_token, batch_size = 20)
    optimizer = optim.Adam(model.parameters(), lr = 2e-4)
    num_epoch = 40
    clip_norm = 2
    valid_ratio = 0.2
    temperature = None
    topk = None
    topp = None
    metrics_df = pd.DataFrame(columns = ["train_ppl","valid_ppl",
                                         "train_context_kld", "train_KG_kld",
                                         "valid_context_kld", "valid_KG_kld",
                                         "emb_avg", "emb_ext", "emb_gre",
                                         "dist-1", "dist-2", "novel"])
    train_model(model, optimizer, train_data_loader, test_data_loader, test_data_loader,
                vocab, embedding, num_epoch, clip_norm, valid_ratio, metrics_df, 
                combine_knowledge = False, temperature = temperature, topk = topk, topp = topp, cuda = 3,
                promt = "Latent Variable InterPlotation")

if __name__=="__main__":
    with launch_ipdb_on_exception():
        main()
