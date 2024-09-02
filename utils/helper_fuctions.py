import random
import importlib
import yaml
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from data.utils_data.default_tokens import DefaultToken


def dict_to_df(dictionary):
    """
    Convert a dictionary with tuple keys (client_index, task) to a Pandas DataFrame.
    
    Parameters:
    - dictionary: dict, the dictionary to be converted
    
    Returns:
    - df: pandas DataFrame, the resulting DataFrame
    """
    # Convert dictionary to a list of tuples (client_index, task, value)
    data = [(key[0], key[1], value) for key, value in dictionary.items()]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Client Index', 'Task', 'Value'])
    
    return df

def lst_dict_to_df(lst_dict):
    """
    Convert a list of dictionaries with tuple keys (client_index, task) to a Pandas DataFrame.
    
    Parameters:
    - lst_dict: list of dict, the list of dictionaries to be converted
    
    Returns:
    - df: pandas DataFrame, the resulting DataFrame
    """
    # Convert list of dictionaries to a list of tuples (client_index, task, value)
    data = [(key[0], key[1], value) for dictionary in lst_dict for key, value in dictionary.items()]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Client Index', 'Task', 'Value'])
    
    return df



def softmax(vec):
    vec = vec - np.max(vec)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def min_max_norm(vec):
    min_val = np.min(vec)
    return (vec - min_val) / (np.max(vec) + 1e-10 - min_val)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def get_client_indices_rounds(args):
    client_indices_rounds = []
    for _ in range(args.rounds):
        client_indices_rounds.append(np.random.choice(np.arange(args.num_clients), size=int(args.num_clients * args.m), replace=False))
    return client_indices_rounds

def get_client_list(list_train_ds, list_eval_ds, model, criterion, optimizer, list_train_ds_genr, list_eval_ds_genr, tokenizer, datacollator, args, candidate_seeds):
    Client = get_class('clients.client_' + args.name, f'Client_{args.name}')
    client_list = []

    for idx in range(args.num_clients):
        client_list.append(Client(list_train_ds[idx], list_eval_ds[idx], deepcopy(model), criterion, deepcopy(optimizer), list_train_ds_genr[idx], list_eval_ds_genr[idx], tokenizer, datacollator, idx, args, candidate_seeds))
        # client_list.append(Client(idx, args, candidate_seeds, list_train_loader[idx], list_eval_loader))
    return client_list


def get_server(args, candidate_seeds, log_dir, **kwargs):
    Server = get_class('servers.server_' + args.name, 'Server')
    return Server(args, candidate_seeds=candidate_seeds, log_dir=log_dir, **kwargs)


def add_vocab(tokenizer):
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
    return tokenizer