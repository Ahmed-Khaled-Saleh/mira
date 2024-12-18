import numpy as np
import pandas as pd
import os
import math
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, Trainer
from peft import (
    set_peft_model_state_dict,
)
from torch.optim import Adam
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import BitsAndBytesConfig
from torch.nn.functional import normalize
import wandb
from utils.validation import *  # noqa: F403
from utils.helper_fuctions import *  # noqa: F403
from trainers.trainer import Trainer
from trainers.callbacks import empty_cach, log_memory
from torch.optim import AdamW, Adam
from servers.base_server import BaseServer
from optimizers.mezo_torch import MeZOOptimizer
from optimizers.mezo_optimizer import MeZOFramework
import datetime
class Server_fedit(BaseServer):
    def __init__(self, args, tokenizer, candidate_seeds, log_dir, **kwargs):
        self.args = args
        self.candidate_seeds = candidate_seeds
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.output_dir = self.args.output_dir
        self.output_dir += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print("secret ==: ", self.args.hf_secret)
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model, 
                                                          trust_remote_code=True,
                                                          device_map='cpu',
                                                          token="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")
        
        if self.args.model in ['openai-community/gpt2-large']:
            target_modules = ['c_attn','c_proj']
        else:
            target_modules = ['q_proj',]

        self.config = LoraConfig(
                    r=self.args.r,
                    target_modules=target_modules,
                    lora_alpha=8,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
        
        self.model = get_peft_model(self.model, self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.seed_pool = {seed: 0.0 for seed in self.candidate_seeds}
        
        self.device = torch.device(f'cuda:{self.args.device}')

        if self.args.bias_sampling:
            self.gradient_history = {seed: [self.args.grad_initial] for seed in self.candidate_seeds}
            self.probabilities = [1.0 / float(len(self.candidate_seeds)) for _ in range(len(self.candidate_seeds))]
        else:
            self.gradient_history = None
            self.probabilities = None

        
        for key, value in kwargs.items():
            setattr(self, key, value)
        


    def train(self,
              client_indices_rounds,
              args,
              run,
              memory_record_dic
              ):


        self.get_clients(args)
        num_clients = len(self.client_list)

        lst_global_metrics_dfs = []        
        previously_selected_clients_set = set()
        last_client_id = None
        local_dataset_len_dict = dict()
        
        print("Finished initializing the clients")
        
        for t in range(1, args.rounds + 1):
            print("length of client list: ", len(self.client_list))
            print("length of client indices rounds: ", len(client_indices_rounds[t-1]))
            selected_client = [self.client_list[i] for i in client_indices_rounds[t-1]]
            
            lst_global_metrics = []
            print("Starting round ", t)
            print("****************************************")
            for client in selected_client:
                print("Client ", client.idx, " is training")

                with torch.no_grad():
                    client.model = deepcopy(self.model)
                client.model = client.model.to(self.device)

                client.initiate_local_training(self.output_dir)
                
                client.optimizer = AdamW(client.model.parameters(),
                                        lr= float(self.args.lr),
                                        weight_decay= float(self.args.weight_decay))
                
                trainer = Trainer(client)
            
                local_iters = client.args.local_step
                epochs = client.args.local_step
                
                metrics = {}
                train_loss, val_loss = trainer.train(fed= True,
                                                    epochs= epochs,
                                                    local_iters= local_iters,
                                                    memory_record_dic= memory_record_dic,
                                                    callbacks=[])
                
                train_loss = np.array(train_loss).mean()
                val_loss = np.array(val_loss).mean()
                task = client.task if isinstance(client.task, str) else client.task[0]

                metrics['train_loss'], metrics['val_loss'], metrics['task'], metrics['idx'] = train_loss, val_loss, task, client.idx
                print("Client ", client.idx, " finished training")
                print("****************************************")
                print(f"Round Sats for client {client.idx}: {metrics}")

                lst_global_metrics.append(metrics)
                
                client.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = \
                    client.terminate_local_training(t, 
                                                    local_dataset_len_dict,
                                                    previously_selected_clients_set)
                
                client.clear_model()
                del trainer
                del client.optimizer
                del client
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
            print("Collecting the weights of clients and performing aggregation")
            self.model = self.model.to(self.device)
            self.model = self.aggregate(
                                        self.model,
                                        client_indices_rounds[t-1],
                                        local_dataset_len_dict,
                                        t,
                                        )
            
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, str(t), "pytorch_model.bin"))
            
            round_train_loss = np.array([metric['train_loss'] for metric in lst_global_metrics]).mean()
            round_val_loss = np.array([metric['val_loss'] for metric in lst_global_metrics]).mean()

            round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))

            run.log({"Train Loss": round_train_loss,
                     "Val Loss": round_val_loss,
                     f"round {t} (GM) Metrics": round_global_metrics})
            
            lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

            for client in selected_client:
                to_del = os.path.join(self.output_dir, str(t), "local_output_{}".format(client.idx),
                                            "pytorch_model.bin")
                if os.path.exists(to_del):
                    os.remove(to_del)
                
        train_acc, eval_acc = self.eval_clients(self.client_list)
        run.log({"Train Accuracy": train_acc,
                 "Eval Accuracy": eval_acc})
            
        return lst_global_metrics_dfs
    
    
    def aggregate(self, model, selected_clients_set, local_dataset_len_dict, epoch):
        weights_array = normalize(
            torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                        dtype=torch.float32),
            p=1, dim=0)

        for k, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(client_id),
                                            "pytorch_model.bin")
            single_weights = torch.load(single_output_dir)
            if k == 0:
                weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                        single_weights.keys()}
            else:
                weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                        for key in
                                        single_weights.keys()}

        set_peft_model_state_dict(model, weighted_single_weights, "default")

        return model

    def eval_clients(self, clients_list):
        clients_metrics = []
        train_acc = 0
        eval_acc = 0
        
        for client in clients_list:
            metrics = {}
            
            if not client.model:
                client.model = deepcopy(self.model)

            trainer = Trainer(client)

            client_train_acc = trainer.train_generate()
            client_eval_acc = trainer.eval_generate()

            task = client.task
            metrics['task'] = task
            metrics['train_acc'] = client_train_acc
            metrics['eval_acc'] = client_eval_acc
            

            clients_metrics.append(metrics)
            client.clear_model()

        for client_metric in clients_metrics:
            train_acc += client_metric['train_acc']
            eval_acc += client_metric['eval_acc']
        
        train_acc /= len(clients_metrics)
        eval_acc /= len(clients_metrics)


        return train_acc, eval_acc

    
    
    
    
    