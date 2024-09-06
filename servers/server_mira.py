import numpy as np
import pandas as pd
import os
import math
from copy import deepcopy

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
import wandb
from utils.validation import *  # noqa: F403
from utils.helper_fuctions import *  # noqa: F403
from trainers.trainer import Trainer
from trainers.callbacks import empty_cach, log_memory
from optimizers.mezo_torch import MeZOOptimizer
from servers.base_server import BaseServer


class Server_mira(BaseServer):
    def __init__(self, args, tokenizer, candidate_seeds, log_dir, **kwargs):
        self.args = args
        self.candidate_seeds = candidate_seeds
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        
        self.model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True)

        self.model_w0 = deepcopy(self.model)
        self.seed_pool = {seed: 0.0 for seed in self.candidate_seeds}
        
        self.device = torch.device(f'cuda:{self.args.device}')

        if self.args.bias_sampling:
            # initialize the probabilities of seeds
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
              memory_record_dic):

        lst_global_metrics_dfs = []        

        self.get_clients(args)
        print("Finished initializing the clients")
        
        for t in range(1, args.rounds + 1):
            print("length of client list: ", len(self.client_list))
            print("length of client indices rounds: ", len(client_indices_rounds[t-1]))
            selected_client = [self.client_list[i] for i in client_indices_rounds[t-1]]
            
            lst_global_metrics = []
            print("Starting round ", t)
            print("****************************************")
            for client in selected_client:
                client.model = deepcopy(self.model)
                
                client.optimizer = deepcopy(MeZOOptimizer(client.model.parameters(),
                                            lr= float(self.args.lr),
                                            zo_eps= self.args.zo_eps,
                                            candidate_seeds= self.candidate_seeds,
                                            weight_decay= self.args.weight_decay))
                
                trainer = Trainer(client)
            
                local_iters = client.args.local_step
                epochs = 1
                
                metrics = {}
                train_loss, val_loss, train_acc, val_acc = trainer.train(fed= True,
                                                                         epochs= epochs,
                                                                         local_iters= local_iters,
                                                                         memory_record_dic= memory_record_dic,
                                                                         callbacks=[empty_cach, log_memory])
                
                train_loss = np.array(train_loss).mean()
                task = client.task if isinstance(client.task, str) else client.task[0]

                

                metrics['train_loss'], metrics['val_loss'], metrics['task'], metrics['train_acc'], metrics['val_acc'] = \
                    train_loss, val_loss, task, train_acc, val_acc
                
                print("Client ", client.idx, " finished training")
                print("****************************************")
                print(f"Round Sats for client {self.client.idx}: {metrics}")

                lst_global_metrics.append(metrics)
            

            for metric in lst_global_metrics:
                run.log({"Train loss": metric['train_loss']})
                run.log({"Val loss": metric['val_loss']})
                run.log({"Train acc": metric['train_acc']})
                run.log({"Val acc": metric['val_acc']})



            round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))
            run.log({f"round {t} (GM) Metrics": round_global_metrics})
            
            lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

            self.aggregate_seed_pool(selected_client)
            self.update_global_model_by_seed_pool()

        return lst_global_metrics_dfs

    def update_global_model_by_seed_pool(self):
        self.model = deepcopy(self.model_w0)
        
        optimizer = MeZOOptimizer(self.model.parameters(), lr=float(self.args.lr), zo_eps=self.args.zo_eps, 
                                  candidate_seeds= self.candidate_seeds)  # noqa: F405
        
        progress_bar = tqdm(range(len(self.seed_pool))) 

        # pull the latest model via accumulated {seed, grad} pairs on the server
        for seed, grad in self.seed_pool.items():
            if grad != 0.0:
                optimizer._sgd_step(seed=seed, grad=grad)
            progress_bar.update(1)
            progress_bar.set_description('server update global model')

    def aggregate_seed_pool(self, selected_client_list):
        # step 7 in the FedK algorithm
        if self.args.equal_weight:
            weight_array = np.array([1.0 for _ in selected_client_list], dtype=np.float64)
            weight_array /= float(len(selected_client_list))
        else:
            weight_array = np.array([len(client.train_loader) for client in selected_client_list], dtype=np.float64)
            weight_array /= float(np.sum(weight_array))
        
        for client_idx in range(len(selected_client_list)):
            local_seed_pool = selected_client_list[client_idx].local_seed_pool
            for seed, grad in local_seed_pool.items():
                self.seed_pool[seed] += grad * weight_array[client_idx]

        for client in selected_client_list:
            client.clear_model()

    def calculate_probabilities(self):
        history_list = [self.gradient_history[seed] for seed in self.candidate_seeds]
        mean_grad_history = np.array([np.mean(np.abs(np.clip(history_cur_seed, -self.args.bias_loss_clip, self.args.bias_loss_clip))) for history_cur_seed in history_list])
        self.probabilities = softmax(min_max_norm(mean_grad_history))  # noqa: F405
        sum_prob = np.sum(self.probabilities)
        if sum_prob != 1.0:
            self.probabilities /= sum_prob
        return self.probabilities


    def eval_clients(self, clients_list ,cur_round, include_eval= False):
        clients_metrics = []

        for client in clients_list:
            metrics = {}

            task = client.task
            metrics['task'] = task

            train_acc, train_loss = client.train_error_and_loss(deepcopy(self.model))
            metrics['train_loss'] = train_loss
            metrics['train_acc'] = train_acc

            if include_eval:
                eval_acc, eval_loss = client.eval_error_and_loss()             
                metrics['eval_loss'] = eval_loss
                metrics['eval_acc'] = eval_acc

            clients_metrics.append(metrics)
        
        return clients_metrics
    
    
    