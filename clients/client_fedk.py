from optimizers.mezo_optimizer import *  # noqa: F403
from optimizers.mezo_bias_optimizer import *  # noqa: F403
from tqdm import tqdm
import os
import torch
from copy import deepcopy
from utils.validation import *  # noqa: F403
from optimizers.mezo_torch import MeZOOptimizer
from trainers.trainer import Trainer
from clients.base_client import BaseClient


class Client_fedk(BaseClient):
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