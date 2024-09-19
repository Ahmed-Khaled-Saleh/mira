import numpy as np
import torch
from transformers import AutoTokenizer

from utils.utils_data.default_tokens import DefaultToken


def subset(args, raw_datasets):

    raw_datasets, _ = torch.utils.data.dataset.random_split(raw_datasets, [int(len(raw_datasets) * args.dataset_subsample), len(raw_datasets) - int(len(raw_datasets) * args.dataset_subsample)])
    y_all = np.array([item['categories'] for item in raw_datasets])
    index_eval = np.where(y_all == args.zerotask)[0]
    # delete the indices of eval samples from the all set
    index_train = np.delete(np.arange(len(y_all)), index_eval)
    raw_datasets = np.array(raw_datasets)
    train_set = raw_datasets[index_train]
    eval_set = raw_datasets[index_eval]
    y_train = np.array([item['categories'] for item in train_set])

    return train_set, eval_set, y_train


def get_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if args.model in ['openai-community/gpt2']:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer