import numpy as np
import pandas as pd
import os
import math
from copy import deepcopy
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from torch.nn.functional import normalize
import wandb
from utils.validation import *  # noqa: F403
from utils.helper_fuctions import *  # noqa: F403
from trainers.trainer import Trainer
from trainers.callbacks import empty_cach, log_memory
from torch.optim import AdamW
from servers.base_server import BaseServer


class Server_fedit(BaseServer):
    def __init__(self, args, tokenizer, candidate_seeds, log_dir, **kwargs):
        self.args = args
        self.candidate_seeds = candidate_seeds
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        
        self.model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cpu', load_in_8bit=True, torch_dtype=torch.float16, trust_remote_code=True)

        self.model_w0 = deepcopy(self.model)
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
        output_dir = os.path.join(output_dir, str(num_clients))

        
        print("Finished initializing the clients")
        
        for t in range(1, args.rounds + 1):
            print("length of client list: ", len(self.client_list))
            print("length of client indices rounds: ", len(client_indices_rounds[t-1]))
            selected_client = [self.client_list[i] for i in client_indices_rounds[t-1]]
            
            lst_global_metrics = []
            print("Starting round ", t)
            print("****************************************")
            for client in selected_client:
                
                client.initiate_local_training()

                self.model = prepare_model_for_int8_training(model)
                config = LoraConfig(
                    r=self.args.r,
                    lora_alpha=16,
                    target_modules=["q_proj",],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                self.model = get_peft_model(self.model, config)

                client.model = deepcopy(self.model)
                client.optimizer = deepcopy(AdamW(client.model.parameters(),
                                            lr= float(self.args.lr),
                                            weight_decay= self.args.weight_decay))

                trainer = Trainer(client)
            
                local_iters = client.args.local_step
                epochs = 1
                
                metrics = {}
                train_loss, val_loss, train_acc, val_acc = trainer.train(fed= True,
                                                                         epochs= epochs,
                                                                         local_iters= local_iters,
                                                                         memory_record_dic= memory_record_dic,
                                                                         callbacks=None)
                
                train_loss = np.array(train_loss).mean()
                task = client.task if isinstance(client.task, str) else client.task[0]

                metrics['train_loss'], metrics['val_loss'], metrics['task'], metrics['train_acc'], metrics['val_acc'] = \
                    train_loss, val_loss, task, train_acc, val_acc
                
                model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = \
                    client.terminate_local_training(t, 
                                                    local_dataset_len_dict,
                                                    previously_selected_clients_set)
                

                print("Client ", client.idx, " finished training")
                print("****************************************")
                print(f"Round Sats for client {self.client.idx}: {metrics}")

                lst_global_metrics.append(metrics)

                del client

        
            print("Collecting the weights of clients and performing aggregation")
            self.model = self.aggregate(
                                        model,
                                        selected_client,
                                        output_dir,
                                        local_dataset_len_dict,
                                        t,
                                        )
            
            torch.save(self.model.state_dict(), os.path.join(output_dir, str(t), "adapter_model.bin"))
            config.save_pretrained(output_dir)

            for metric in lst_global_metrics:
                run.log({"Train loss": metric['train_loss']})
                run.log({"Val loss": metric['val_loss']})
                run.log({"Train acc": metric['train_acc']})
                run.log({"Val acc": metric['val_acc']})



            round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))
            run.log({f"round {t} (GM) Metrics": round_global_metrics})
            
            lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

            
        return lst_global_metrics_dfs
    
    



    def aggregate(self, model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
        weights_array = normalize(
            torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                        dtype=torch.float32),
            p=1, dim=0)

        for k, client_id in enumerate(selected_clients_set):
            single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
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


    
    
    
    
    