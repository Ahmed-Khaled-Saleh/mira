
from tqdm import tqdm
import os
import torch
from copy import deepcopy
from collections import OrderedDict
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from utils.validation import *  # noqa: F403
from optimizers.mezo_torch import MeZOOptimizer
from trainers.trainer import Trainer
from clients.base_client import BaseClient


class Client_fedit(BaseClient):
    def __init__(self,
                 train_ds,
                 eval_ds,
                 model,
                 criterion,
                 optimizer,
                 train_ds_genr,
                 eval_ds_genr,
                 tokenizer,
                 datacollator,
                 idx,
                 args,
                 candidate_seeds,
                 K= 0):

        '''
        A client is defined as an object that contains :

        1- **Essentials**:
            dataseet (train, eval), model, loss function (criterion), and an optimizer.
            
        2- **Extra information**:
            Task-dependent and algrithm-specifics
        '''
        
        super().__init__(train_ds, eval_ds, model, criterion, optimizer)
        
        self.train_ds_genr = train_ds_genr
        self.eval_ds_genr = eval_ds_genr
        self.tokenizer = tokenizer
        self.data_collator = datacollator
        self.idx = idx
        self.args = args
        self.device = torch.device(f'cuda:{self.args.device}')
        self.candidate_seeds = candidate_seeds
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        self.task = self.train_ds[0]['task']
        self.task = self.task if isinstance(self.task, str) else self.task[0]
        self.train_stat = {}
        self.test_stats = {}
        


    
    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))




    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id

    def clear_model(self):
        self.model = None

    def _add_seed_pole(self, zo_random_seed, projected_grad):
        if self.local_seed_pool is not None:
            self.local_seed_pool[zo_random_seed] += projected_grad
    
    def migrate(self, device):
        """
        migrate a client to a new device
        """
        self.device = device

    def pull(self, forked_global_model):
        """
        pull model from the server
        """
        self.model = forked_global_model