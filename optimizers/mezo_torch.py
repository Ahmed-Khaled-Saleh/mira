import numpy as np
import torch
from torch.optim import Optimizer

class MeZOOptimizer(Optimizer):
    def __init__(self, 
                 params, 
                 lr= 0.000003,
                 zo_eps=0.000005,
                 candidate_seeds= None,
                 weight_decay=0.0):
        
        defaults = dict(lr=lr, 
                        zo_eps=zo_eps,
                        candidate_seeds= candidate_seeds,
                        weight_decay=weight_decay)
        
        super(MeZOOptimizer, self).__init__(params, defaults)
        self.candidate_seeds = candidate_seeds
        self.zo_eps = zo_eps


    def get_candidate_seeds(self):
        # Method to retrieve candidate_seeds from the first param group
        if len(self.param_groups) > 0:
            return self.param_groups[0]['candidate_seeds']
        return None
    
    def get_zoeps(self):
        # Method to retrieve zo_eps from the first param group
        if len(self.param_groups) > 0:
            return self.param_groups[0]['zo_eps']
        return
        

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError("Closure is required for MeZOOptimizer")
        self.candidate_seeds = self.get_candidate_seeds()
        self.zo_random_seed = np.random.choice(self.candidate_seeds, 1)[0]
        self.zo_eps = self.get_zoeps()

        # Positive perturbation
        self._perturb_parameters(scaling_factor=1)
        loss_pos = closure()
        print(f"loss_pos shape: {loss_pos.shape if hasattr(loss_pos, 'shape') else 'scalar'}")
        
        # Negative perturbation
        self._perturb_parameters(scaling_factor=-2)
        loss_neg = closure()

        # Restore original parameters
        self._perturb_parameters(scaling_factor=1)
        
        if torch.isnan(loss_pos) or torch.isnan(loss_neg):
            print("Warning: NaN loss detected in optimizer step")
            return loss_pos, self.zo_random_seed, torch.zeros_like(self.projected_grad)
    
        self.projected_grad = (loss_pos - loss_neg) / (2 * self.zo_eps)
    
        if torch.isnan(self.projected_grad).any():
            print("Warning: NaN detected in projected gradient")
            self.projected_grad = torch.zeros_like(self.projected_grad)

        self._sgd_step()
        return loss_pos, self.zo_random_seed, self.projected_grad
    
    def _sgd_step(self, seed= None, grad= None):
        
        self.candidate_seeds = self.get_candidate_seeds()

        if seed is None:
            seed = self.zo_random_seed
        if grad is None:
            grad = self.projected_grad

        for group in self.param_groups:
            lr = group['lr']
            zo_eps = group['zo_eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                
                torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
                torch.manual_seed(seed)
                scalar = lr * grad
                gen = torch.Generator(device= p.device)
                z = torch.empty(p.shape).to(p.device)
                z.normal_(mean=0, std=1, generator=gen).to(p.dtype)
                
                if weight_decay != 0:
                    p.data.add_(weight_decay, p)

                p.data.add_(z, alpha=-scalar)

        del z


    def _perturb_parameters(self, scaling_factor):
        for group in self.param_groups:
            zo_eps = group['zo_eps']
            for p in group['params']:
                torch.manual_seed(self.zo_random_seed)
                gen = torch.Generator(device= p.device)
                z = torch.empty(p.shape).to(p.device)
                z = z.normal_(mean=0, std=1, generator=gen).to(p.dtype)
                p.add_(scaling_factor * zo_eps * z)
        del z
