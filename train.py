import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from utils import generate_mask_by_length
from metrics import embedding_metrics, diversity_metrics, evaluate_response

def print_metrics(metrics_df, idx, promt = ""):
    report_str = "\n--------------------------------- \n {}% Training Set\n".format(idx * 100)
    if len(promt) != 0:
        report_str += promt + "\n"
    ppl_str = "\n".join(["Training PPL : {0:.6f}".format(metrics_df.loc[idx, "train_ppl"]),
                         "Valid PPL : {0:.6f}".format(metrics_df.loc[idx, "valid_ppl"])])
    kld_str = "\n".join(["Training Context KLD : {0:.6f}".format(metrics_df.loc[idx, "train_context_kld"]),
                         "Training KG KLD : {0:.6f}".format(metrics_df.loc[idx, "train_KG_kld"]),
                         "Valiad Context KLD : {0:.6f}".format(metrics_df.loc[idx, "valid_context_kld"]),
                         "Valid KG KLD : {0:.6f}".format(metrics_df.loc[idx, "valid_KG_kld"])])
    emb_metrics_str = "\n".join(["Embedding Average: {0:.6f}".format(metrics_df.loc[idx, "emb_avg"]),
                                "Embedding Extreme: {0:.6f}".format(metrics_df.loc[idx, "emb_ext"]),
                                "Embedding Greedy : {0:.6f}".format(metrics_df.loc[idx, "emb_gre"])])
    diversity_metrics_str = "\n".join(["Dist-1 : {0:.6f}".format(metrics_df.loc[idx, "dist-1"]),
                                      "Dist-2 : {0:.6f}".format(metrics_df.loc[idx, "dist-2"]),
                                      "Novelty: {0:.6f}".format(metrics_df.loc[idx, "novel"])])

    result_str = "\n".join([report_str, ppl_str, kld_str, emb_metrics_str, diversity_metrics_str])
    result_str = result_str + "\n---------------------------------\n"
    return result_str

def CE_loss_length_masked(logit, tgt_sents, tgt_mask, tgt_len):
    """
    Args:
        logit (FloatTensor)       : [tgt_num_sents, max_tgt_len, num_vocab]
        tgt_sents (LongTensor)    : [tgt_num_sents, max_tgt_len]
        tgt_len (LongTensor)      : [tgt_num_sents, max_tgt_len]
    """
    # logit_flattened : [tgt_num_sents * max_tgt_len, num_vocab]
    logit_flattened = logit.view(-1, logit.size(2))
    # log_probs_flattened : [tgt_num_sents * max_tgt_len, num_vocab]
    log_probs_flattened = F.log_softmax(logit_flattened, dim = 1)
    # tgt_sents_flattened : [tgt_num_sents * max_tgt_len, 1]
    tgt_sents_flattened = tgt_sents.view(-1, 1)
    # losses_flattened : [tgt_num_sents * max_tgt_len, 1]
    losses_flattened = -torch.gather(log_probs_flattened, dim = 1, index = tgt_sents_flattened)
    # losses : [tgt_num_sents, max_tgt_len]
    losses = losses_flattened.view(*tgt_sents.size())
    
    losses = losses * tgt_mask
    loss = losses.sum() / tgt_len.float().sum()
    return loss

def compuate_loss(logit, tgt_sents, tgt_len, context_kld, KG_kld, kld_ratio):
    """
    Args:
        logit (FloatTensor)       : [tgt_num_sents, max_tgt_len, num_vocab]
        tgt_sents (LongTensor)    : [tgt_num_sents, max_tgt_len]
        tgt_len (LongTensor)      : [tgt_num_sents]
        converage_loss (FloatTensor)     : [tgt_num_sents, max_tgt_len]
    """
    max_tgt_len = tgt_sents.size(1)
    # tgt_mask : [max_tgt_len, tgt_num_sents]
    tgt_mask = generate_mask_by_length(tgt_len, max_tgt_len, masked_mode = "add")
    # tgt_mask : [tgt_num_sents, max_tgt_len]
    tgt_mask = tgt_mask.permute(1, 0)
    CE_loss = CE_loss_length_masked(logit, tgt_sents, tgt_mask, tgt_len)
    loss = CE_loss + (context_kld + KG_kld) * kld_ratio
    return loss, CE_loss, context_kld, KG_kld

def sample_dialogue(src_sents_list, src_len_list, KG_sents_list, KG_len_list,
                    num_utterance_pair_list, generated_sents_list, vocab):
    result_str_list = ["\n---------------------------------"]
    num_utterance = num_utterance_pair_list[0] * 2
    sampled_src_sents = src_sents_list[:num_utterance]
    sampled_src_len = src_len_list[:num_utterance]
    sampled_KG_sents = KG_sents_list[:num_utterance]
    sampled_KG_len = KG_len_list[:num_utterance]
    sampled_generated_sents = generated_sents_list[:num_utterance]
    for index in range(num_utterance):
        src = sampled_src_sents[index]
        KG = sampled_KG_sents[index // 2]
        length = sampled_src_len[index]
        KG_length = sampled_KG_len[index // 2]
        if index % 2 == 0:
            utterance_str = "Question : " + " ".join(vocab.convert_id_list(src[:length], mode="truncated"))
        else:
            generated = sampled_generated_sents[index // 2]
            utterance_str = "Answer : " + " ".join(vocab.convert_id_list(src[:length], mode="truncated"))
            utterance_str = utterance_str + "\n Generated: " + " ".join(vocab.convert_id_list(generated, mode="truncated"))
            utterance_str = utterance_str + "\n Checked: " + " ".join(vocab.convert_id_list(KG[:KG_length], mode="truncated"))
            utterance_str = utterance_str + "\n"

        result_str_list.append(utterance_str)
    result_str_list.append("---------------------------------") 
    result_str = "\n".join(result_str_list)
    return result_str

def eval_model(model, valid_data_loader, vocab, embedding, 
               combine_knowledge, temperature, topk, topp):
    with torch.no_grad():
        CE_loss_list = []
        context_kld_list, KG_kld_list = [], []
        hypothesis_list = []
        reference_list = []
        src_sents_list = []
        src_len_list = []
        KG_sents_list = []
        KG_len_list = []
        num_utterance_pair_list = []
        for batch_index, batch in enumerate(valid_data_loader):
            model.train(False)
            # load the data to Tensor
            valid_src_sents = torch.LongTensor(batch["source"]).cuda()
            valid_src_word_len = torch.LongTensor(batch["source_len"]).cuda()
            valid_num_utterance = torch.LongTensor(batch["num_utterance_pair"]).cuda()
            valid_KG_sents = torch.LongTensor(batch["checked_sent"]).cuda()
            valid_KG_word_len = torch.LongTensor(batch["checked_sent_len"]).cuda()
            valid_tgt_sents = torch.LongTensor(batch["target"]).cuda()
            valid_tgt_input_sents = torch.LongTensor(batch["input_target"]).cuda()
            valid_tgt_len = torch.LongTensor(batch["target_len"]).cuda()
            # compute the loss
            valid_logit,context_kld,KG_kld = model.forward(src_sents = valid_src_sents, 
                                                           src_word_len = valid_src_word_len,
                                                           src_utterance_len = valid_num_utterance, 
                                                           KG_sents = valid_KG_sents,
                                                           KG_word_len = valid_KG_word_len, 
                                                           tgt_word_input = valid_tgt_input_sents,
                                                           tgt_word_len = valid_tgt_len,
                                                           combine_knowledge = combine_knowledge)
            loss, CE_loss, context_kld, KG_kld = compuate_loss(valid_logit, valid_tgt_sents, valid_tgt_len,
                                                               context_kld, KG_kld, 0)
            CE_loss_list.append(CE_loss.detach().item())
            context_kld_list.append(context_kld.detach().item())
            KG_kld_list.append(KG_kld.detach().item())
            # store the generated dialogues
            valid_generated = model.greedy_generate(src_sents = valid_src_sents, 
                                                    src_word_len = valid_src_word_len,
                                                    src_utterance_len = valid_num_utterance,
                                                    KG_sents = valid_KG_sents,
                                                    KG_word_len = valid_KG_word_len,
                                                    max_tgt_word_len = valid_tgt_input_sents.size(1),
                                                    initial_word_idx = vocab.truncated_word_id["<RESPONSE>"],
                                                    combine_knowledge = False,
                                                    temperature = temperature, topk = topk, topp = topp)
            hypothesis_batch_list = valid_generated.detach().cpu().tolist()
            hypothesis_list.extend(hypothesis_batch_list)
            reference_list.extend(batch["target"])
            src_sents_list.extend(batch["source"])
            src_len_list.extend(batch["source_len"])
            KG_sents_list.extend(batch["checked_sent"])
            KG_len_list.extend(batch["checked_sent_len"])
            num_utterance_pair_list.extend(batch["num_utterance_pair"])
    # evaluate the metrics
    none_set = set([vocab.truncated_word_id['<EOS>'],
                    vocab.unknown_id])
    metrics_dict = evaluate_response(hypothesis_list, reference_list, embedding, none_set) 
    # sample the dialogue
    sampled_dialogue_str = sample_dialogue(src_sents_list, src_len_list,
                                           KG_sents_list, KG_len_list, 
                                           num_utterance_pair_list, hypothesis_list,
                                           vocab)
    metrics_dict["valid_ppl"] = np.exp(np.mean(CE_loss_list))
    metrics_dict["valid_context_kld"] = np.mean(context_kld_list)
    metrics_dict["valid_KG_kld"] = np.mean(KG_kld_list)

    return metrics_dict, sampled_dialogue_str

def train_model(model, optimizer, train_data_loader, valid_data_loader, test_data_loader, 
                vocab, embedding, num_epoch, clip_norm, valid_ratio, metrics_df, 
                combine_knowledge, temperature, topk, topp, cuda, promt = ""):
    num_batch = len(train_data_loader)
    valid_num_batch = int(valid_ratio * num_batch)
    logging.info("Number of Batches is {}\n Valid Number of Batches {}".format(num_batch, valid_num_batch))
    with torch.cuda.device(cuda):
        model.cuda()
        train_CE_loss_list = []
        train_context_kld_list, train_KG_kld_list = [], []
        num_validation = 0
        logging.info("Starting Training")
        for epoch_i in range(1, num_epoch+1):
            logging.info("Epoch {}".format(epoch_i))
            for batch_index, batch in enumerate(train_data_loader):
                model.train(True)
                optimizer.zero_grad()
                # load the data to Tensor
                train_src_sents = torch.LongTensor(batch["source"]).cuda()
                train_src_word_len = torch.LongTensor(batch["source_len"]).cuda()
                train_num_utterance = torch.LongTensor(batch["num_utterance_pair"]).cuda()
                train_KG_sents = torch.LongTensor(batch["checked_sent"]).cuda()
                train_KG_word_len = torch.LongTensor(batch["checked_sent_len"]).cuda()
                train_tgt_sents = torch.LongTensor(batch["target"]).cuda()
                train_tgt_input_sents = torch.LongTensor(batch["input_target"]).cuda()
                train_tgt_len = torch.LongTensor(batch["target_len"]).cuda()
                # forward pass of the model and compute the loss
                train_logit,context_kld,KG_kld = model.forward( src_sents = train_src_sents, 
                                                                src_word_len = train_src_word_len,
                                                                src_utterance_len = train_num_utterance, 
                                                                KG_sents = train_KG_sents,
                                                                KG_word_len = train_KG_word_len, 
                                                                tgt_word_input = train_tgt_input_sents,
                                                                tgt_word_len = train_tgt_len,
                                                                combine_knowledge = combine_knowledge)
                total_batch_index = (epoch_i - 1) * num_batch + batch_index
                kld_ratio = float((total_batch_index+100)//100) / 500
                kld_ratio = kld_ratio if kld_ratio <= 1 else 1
                loss, CE_loss, context_kld, KG_kld = compuate_loss(train_logit, train_tgt_sents, train_tgt_len,
                                                                   context_kld, KG_kld, kld_ratio)
                loss.backward()
                train_CE_loss_list.append(CE_loss.detach().item())
                train_context_kld_list.append(context_kld.detach().item())
                train_KG_kld_list.append(KG_kld.detach().item())
                # update the parameters by the loss
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                # evaluate the model and save it to result DataFrame
                if (batch_index + 1) % valid_num_batch == 0:
                    num_validation += 1
                    metrics_dict,sampled_dialog_str = eval_model(model, valid_data_loader, vocab, embedding,
                                                                combine_knowledge, temperature, topk, topp)
                    metrics_dict["train_ppl"] = np.exp(np.mean(train_CE_loss_list))
                    metrics_dict["train_context_kld"] = np.mean(train_context_kld_list)
                    metrics_dict["train_KG_kld"] = np.mean(train_KG_kld_list)
                    metrics_df.loc[valid_ratio * num_validation] = metrics_dict
                    report_str = print_metrics(metrics_df, valid_ratio * num_validation, promt)
                    logging.info(report_str)
                    logging.info(sampled_dialog_str)
