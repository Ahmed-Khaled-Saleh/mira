import numpy as np
import pandas as pd
import os
import math
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import torch
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
    get_peft_model,
    prepare_model_for_kbit_training,
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

class Server_mira(BaseServer):
    def __init__(self, args, tokenizer, candidate_seeds, log_dir, **kwargs):
        self.args = args
        self.candidate_seeds = candidate_seeds
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.quant_config = BitsAndBytesConfig(
            load_in_8bit=True,

        )
        N = self.args.num_clients
        b = np.random.uniform(0,1,size=(N,N))
        b_symm = (b + b.T)/2
        b_symm[b_symm < 0.25] = 0
        self.alk_connection = b_symm
        self.L_k = 0.01
        self.beta = 1
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model, 
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True,
                                                          device_map='cpu',
                                                          token=self.args.hf_secret)

        self.config = LoraConfig(
                    r=self.args.r,
                    target_modules=['q_proj',],
                    lora_alpha=16,
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
              memory_record_dic,
              output_dir= "./lora-shepherd/"):


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

                client.initiate_local_training()
                
                # client.optimizer = MeZOOptimizer(client.model.parameters(),
                #                             lr= float(self.args.lr),
                #                             zo_eps= self.args.zo_eps,
                #                             candidate_seeds= self.candidate_seeds,
                #                             weight_decay= float(self.args.weight_decay))
                # client.optimizer = MeZOFramework(
                #     client.model,
                #     self.args,
                #     float(self.args.lr),
                #     self.candidate_seeds
                # )
                client.optimizer = AdamW(client.model.parameters(),
                                        lr= float(self.args.lr),
                                        weight_decay= float(self.args.weight_decay))
                
                trainer = Trainer(client)
            
                local_iters = client.args.local_step
                epochs = 1
                
                metrics = {}
                train_loss, val_loss = trainer.train(fed= True,
                                                    epochs= epochs,
                                                    local_iters= local_iters,
                                                    memory_record_dic= memory_record_dic,
                                                    callbacks=[empty_cach, log_memory])
                
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
                                        output_dir,
                                        local_dataset_len_dict,
                                        t,
                                        )
                        
            round_train_loss = np.array([metric['train_loss'] for metric in lst_global_metrics]).mean()
            round_val_loss = np.array([metric['val_loss'] for metric in lst_global_metrics]).mean()

            round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))

            run.log({"Train Loss": round_train_loss,
                     "Val Loss": round_val_loss,
                     f"round {t} (GM) Metrics": round_global_metrics})
            
            lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

            # torch.save(self.model.state_dict(), os.path.join(output_dir, str(t), "adapter_model.bin"))
            # self.config.save_pretrained(output_dir)
            
            for client in selected_client:
                to_del = os.path.join(output_dir, str(t), "local_output_{}".format(client.idx),
                                            "pytorch_model.bin")
                if os.path.exists(to_del):
                    os.remove(to_del)
                
        train_acc, eval_acc = self.eval_clients(self.client_list)
        run.log({"Train Accuracy": train_acc,
                 "Eval Accuracy": eval_acc})
            
        return lst_global_metrics_dfs
    
    
    def aggregate(self, model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, akl):
        
        lora_params_dict = defaultdict(list)

        # Collect LoRA parameters from all clients
        for client_id in selected_clients_set:
            client_model_path = os.path.join(output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin")
            client_state_dict = torch.load(client_model_path, map_location=self.device)
            
            for name, param in client_state_dict.items():
                if 'lora_A' in name or 'lora_B' in name:
                    lora_params_dict[name].append(param)

        # Aggregate LoRA parameters
        for name, params_list in lora_params_dict.items():
            aggregated_param = torch.zeros_like(params_list[0])
            client_count = len(params_list)

            for i, client_param in enumerate(params_list):
                client_id = selected_clients_set[i]
                for j, other_client_param in enumerate(params_list):
                    if i != j:
                        other_client_id = selected_clients_set[j]
                        weight = akl[int(client_id)][int(other_client_id)]
                        aggregated_param += weight * (client_param - other_client_param)

            # Apply update
            update_factor = 0.5 * self.args.lr * self.args.L_k * self.args.beta * self.args.local_epochs / client_count
            aggregated_param *= update_factor

            # Update model's LoRA parameters
            current_param = model.state_dict()[name]
            model.state_dict()[name].copy_(current_param - aggregated_param)

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

    
    
    
    def get_alk(self, k):
        # temporary fix value of akl, all client has same value of akl
        #akl = 0.25 # can set any value but need to modify eta accordingly
        akl = 0.5
        #akl = 1
        return akl
    