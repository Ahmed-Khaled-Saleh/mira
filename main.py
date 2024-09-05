import os
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from copy import deepcopy
from utils.utils_data.load_data import get_datasets
from trainers.trainer import Trainer
from optimizers.mezo_torch import MeZOOptimizer
from models.model import GPT2
from trainers.callbacks import empty_cach, log_memory
from utils.helper_fuctions import (setup_seed,  
                                   load_config,
                                   get_server,
                                   get_client_list,
                                   get_client_indices_rounds
                                   )


def federated_training(server,
                       client_indices_rounds,
                       args,
                       run,
                       memory_record_dic):
    print("...Starting the Federated Training...")
    lst_global_metrics_dfs = server.train(client_indices_rounds, args, run, memory_record_dic)
    
    return lst_global_metrics_dfs

def process_main(args_config_fname):
    fname = args_config_fname.fname
    config = load_config(fname)
    experiment = config[0]  # Assuming single experiment per config file

    args = argparse.Namespace(**experiment)
    run = wandb.init(project=args.project, name= args.name, config=args)
    client_indices_rounds = get_client_indices_rounds(args)
    print("client indices rounds generated", client_indices_rounds)
    
    time_stamp = str(time.time())
    memory_record_dic = {}
    
    previous_metric = args.eval_metric
    args.eval_metric = 'loss'
    # set CUDA visibility to targeted cuda device, to avoid the several hundred MB memory consumption of device 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    setup_seed(args.seed)
    print("Generating the datasets...")
    loss_ds, gener_ds = get_datasets(args)
    list_train_ds, list_eval_ds, tokenizer, datacollator = loss_ds
    list_train_ds_genr, list_eval_ds_genr = gener_ds
    print("Datasets generated successfully.")

    if args.dataset == 'instruct':
        args.iid = 'meta'
    log_dir = time_stamp

    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir)
    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')

    # since only CUDA device is available, load all models on device 0
    args.device = 0
    
    # sample `K` candidate seeds
    candidate_seeds = np.random.randint(1, 100000000000, args.K)
    
    def criterion(out):
        return out.loss

    kwargs = {"list_train_ds": list_train_ds, 
              "list_eval_ds": list_eval_ds, 
              "criterion": criterion, 
              "list_train_ds_genr": list_train_ds_genr, 
              "list_eval_ds_genr": list_eval_ds_genr, 
              "datacollator": datacollator}
    
    server = get_server(args, tokenizer, candidate_seeds, log_dir, **kwargs)
    print("server initialized")    

    if args.log:
        with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
            json.dump(memory_record_dic, writer)


    lst_global_metrics_dfs = federated_training(server, client_indices_rounds, args, run, memory_record_dic)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for i in range(len(lst_global_metrics_dfs)):
        print(log_dir)
        lst_global_metrics_dfs[i].to_csv(os.path.join(log_dir, f'global_metrics_{i}.csv'))

    run.finish()



if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_dotenv()
    key = os.getenv("WANDB_API_KEY")
    wandb.login(key=key, verify=False)

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--fname', type=str,
        help='name of config file to load',
        default='configs.yaml')
    
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of nodes')
    
    args_config_fname = parser.parse_args()
    process_main(args_config_fname)