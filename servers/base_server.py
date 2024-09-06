import os
import math
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from clients.client_fedk import Client_fedk
from utils.validation import rouge_score
from utils.helper_fuctions import get_class
class BaseServer():

    def __init__(self) -> None:
        pass

    
    def get_clients(self, args):

        Server = get_class('clients.client' + args.name, 'Client_' + args.name)
        client_list = []
        for idx in range(args.num_clients):
            client_list.append(Server(self.list_train_ds[idx],
                                           self.list_eval_ds[idx],
                                           None,
                                           self.criterion,
                                           None,
                                           self.list_train_ds_genr[idx],
                                           self.list_eval_ds_genr[idx],
                                           self.tokenizer,
                                           self.datacollator, 
                                           idx, 
                                           self.args,
                                           candidate_seeds= self.candidate_seeds)
                                           )

        
        self.client_list = client_list 
        

    def eval(self, cur_round, eval_avg_acc):
        if self.args.eval_metric == 'loss':
            eval_metric = self.eval_loss(cur_round)
        else:
            eval_metric =  self.eval_generate(cur_round)
            
        if self.args.save and cur_round > 0:
            save_dir = self.log_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if (self.args.eval_metric == 'loss' and eval_metric < np.min(eval_avg_acc)) or (self.args.eval_metric != 'none' and eval_metric > np.max(eval_avg_acc)):
                for file_name in os.listdir(save_dir):
                    if 'best' in file_name:
                        os.remove(os.path.join(save_dir, file_name))  
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_best_round{cur_round}.bin'))
            for file_name in os.listdir(save_dir):
                if 'final' in file_name:
                    os.remove(os.path.join(save_dir, file_name)) 
            torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_final_round{cur_round}.bin'))
        return eval_metric

    
    def eval_loss(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()
        
        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        loss_total_eval = 0.0
        num_eval = 0
        
        loss_per_task = {}
        with torch.no_grad():
            for batch in self.eval_loader:
                task = batch['task'][0]
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                outputs = self.model(**batch)
                loss = outputs.loss
                loss_per_task[task] = loss if task not in loss_per_task else loss_per_task[task] + loss
                progress_bar_eval.update(1)
                if torch.isnan(loss):
                    continue
                loss_total_eval += loss
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, loss: {loss_total_eval / num_eval}')
        print()
        print()
        self.model = self.model.cpu()
        return (loss_total_eval / num_eval).item(), loss_per_task

    def eval_generate(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()
        
        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        acc_total_eval = 0.0
        num_eval = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                label_ids = batch['labels'].to(self.device)
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_eval += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.tokenizer)  # noqa: F405
                progress_bar_eval.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, metric: {acc_total_eval / num_eval}')
        print()
        print()
        self.model = self.model.cpu()
        return acc_total_eval / num_eval