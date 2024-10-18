import numpy as np
import pandas as pd
import os
import math
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.optim import Adam
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    LoHaConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.nn.functional import normalize
import wandb
from utils.validation import *  # noqa: F403
from utils.helper_fuctions import *  # noqa: F403
from trainers.trainer import Trainer
from trainers.callbacks import empty_cach, log_memory
from torch.optim import AdamW, Adam, SGD
from servers.base_server import BaseServer
from optimizers.mezo_torch import MeZOOptimizer
from optimizers.mezo_optimizer import MeZOFramework

class Server_mira_plus(BaseServer):
    def __init__(self, args, tokenizer, candidate_seeds, log_dir, **kwargs):
        self.args = args
        self.candidate_seeds = candidate_seeds
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.output_dir = self.args.output_dir
        self.output_dir += datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       
        N = self.args.num_clients
        b = np.random.uniform(0,1,size=(N,N))
        b_symm = (b + b.T)/2
        b_symm[b_symm < 0.25] = 0
        self.alk_connection = b_symm
        self.L_k = self.args.lambda_
        self.beta = 1

        if self.args.model in ['openai-community/gpt2-large']:
            target_modules = ['c_attn','c_proj']
            hf_cache = self.args.hf_cache
        else:
            target_modules = ['q_proj', ]
            hf_cache = self.args.hf_cache

        self.model = AutoModelForCausalLM.from_pretrained(self.args.model, 
                                                          trust_remote_code=True,
                                                          device_map='cpu',
                                                          token=self.args.hf_secret,
                                                          )

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
        latest_model_iter = dict()

        for t in range(1, args.rounds + 1):
            print("length of client list: ", len(self.client_list))
            print("length of client indices rounds: ", len(client_indices_rounds[t-1]))
            selected_client = [self.client_list[i] for i in client_indices_rounds[t-1]]
            
            lst_global_metrics = []
            print("Starting round ", t)
            print("****************************************")

            for client in selected_client:
                print("Client ", client.idx, " is training")

                if client.idx in latest_model_iter:
                    comm_round = latest_model_iter[client.idx]
                    model_path = os.path.join(self.output_dir, str(comm_round), "local_output_{}".format(client.idx),
                                            "pytorch_model.bin")
                else:
                    model_path = ''

                with torch.no_grad():
                    client.model = deepcopy(self.model)

                client.model = client.model.to(self.device)
                if os.path.exists(model_path):
                    set_peft_model_state_dict(client.model,
                                              torch.load(model_path, map_location=self.device),
                                              "default")
                
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
                
                torch.save(client.embedding.state_dict(), os.path.join(self.output_dir, str(t), "embedding.bin"))
                
                client.clear_model()
                del trainer
                del client.optimizer
                del client
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                
            print("Updating the connection matrix")
            self.alk_connection = self.cosine_similarity_per_layer(
                                                                    client_indices_rounds[t-1], 
                                                                    t
                                                                   )
            

            print("Collecting the weights of clients and performing aggregation")
            self.aggregate(
                            client_indices_rounds[t-1],
                            t,
                            )
            
            round_train_loss = np.array([metric['train_loss'] for metric in lst_global_metrics]).mean()
            round_val_loss = np.array([metric['val_loss'] for metric in lst_global_metrics]).mean()

            round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))

            run.log({"Train Loss": round_train_loss,
                     "Val Loss": round_val_loss,
                     f"round {t} (GM) Metrics": round_global_metrics})
            
            lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

            for client in selected_client:
                latest_model_iter[client.idx] = t

        models_paths = wandb.Table(dataframe=pd.DataFrame([latest_model_iter]))
        run.log({"models_paths": models_paths})

        train_acc, eval_acc = self.eval_clients(self.client_list)
        run.log({"Train Accuracy": train_acc,
                 "Eval Accuracy": eval_acc})
            
        return lst_global_metrics_dfs
    
    
    def cosine_similarity_per_layer(self, selected_clients_set, epoch):

        for i, client_id in enumerate(selected_clients_set):
            client_path = os.path.join(self.output_dir, str(epoch), "embedding.bin")

            client_state_dict = torch.load(client_path, map_location=self.device)

            for j, other_client_id in enumerate(selected_clients_set):
                similarities = []
                if i != j:
                    other_client_path = os.path.join(self.output_dir, str(epoch), "embedding.bin")
                    other_client_state_dict = torch.load(other_client_path, map_location=self.device)

                    for (key1, param1), (key2, param2) in zip(client_state_dict.items(), other_client_state_dict.items()):

                        assert key1 == key2, "Keys must match in both state_dicts"
                        assert param1.shape == param2.shape, f"Parameters for {key1} must have the same shape"
                        
                        # Compute cosine similarity for the corresponding tensors
                        sim = F.cosine_similarity(param1.view(-1), param2.view(-1), dim=0).item()
                        similarities.append(sim)
                else:
                    continue
                avg_sim = sum(similarities) / len(similarities)
                self.alk_connection[int(client_id)][int(other_client_id)] = F.softplus(torch.tensor(avg_sim)).item()
        # save memory
        del client_state_dict
        del other_client_state_dict
        del similarities
        del avg_sim
        torch.cuda.empty_cache()
        return self.alk_connection
    
    def aggregate(self, selected_clients_set, epoch):
        global_lr = float(self.args.lr) * float(self.args.local_step)
        reg_param = self.L_k

        for i, client_id in enumerate(selected_clients_set):
            client_path = os.path.join(self.output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin")
            client_state_dict = torch.load(client_path, map_location=self.device)

            client_diff = defaultdict(lambda: torch.tensor(0.0).to(self.device))

            for key in client_state_dict.keys():
                client_diff[key] = torch.zeros_like(client_state_dict[key]).to(self.device)

            for j, other_client_id in enumerate(selected_clients_set):
                if i != j:
                    other_client_path = os.path.join(self.output_dir, str(epoch), f"local_output_{other_client_id}", "pytorch_model.bin")
                    other_client_state_dict = torch.load(other_client_path, map_location=self.device)#.state_dict()

                    weight = self.alk_connection[int(client_id)][int(other_client_id)]
                    for key in client_state_dict.keys():
                        client_diff[key].data += weight * (client_state_dict[key].data.clone() - other_client_state_dict[key].data.clone())

            for key in client_state_dict:
                client_state_dict[key].data -=  global_lr * reg_param * client_diff[key].data


            # save the updated model
            save_dir = os.path.join(self.output_dir, str(epoch + 1), f"local_output_{client_id}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "pytorch_model.bin")
            torch.save(client_state_dict, save_path)
            set_peft_model_state_dict(self.model, client_state_dict, "default")
            # self.model.save_pretrained(save_dir)
            # self.config.save_pretrained(save_dir)


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

    
    
    
    def get_alk(self):
        # temporary fix value of akl, all client has same value of akl
        #akl = 0.25 # can set any value but need to modify eta accordingly
        akl = 0.5
        #akl = 1
        return akl
    