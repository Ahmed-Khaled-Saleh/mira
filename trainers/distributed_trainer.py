import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, ReduceOp
import os
from optimizers.mezo_torch import MeZOOptimizer




class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: MeZOOptimizer,
    ) -> None:
        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)
        self.model = model.to(self.local_rank)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        def closure():
            return self.model(**batch)

        loss = self.optimizer.step(closure)
        print(loss)
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))['input_ids'])
        print(f"[GPU{self.local_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)
        total_loss = 0
        for batch in self.train_loader:
            
            batch = {
                    'input_ids': batch['input_ids'].to(self.local_rank),
                    'labels': batch['labels'].to(self.local_rank),
                    'attention_mask': batch['attention_mask'].to(self.local_rank) 
                }
            
            # batch = {k: v.to(self.local_rank) for k, v in batch.items()}
            loss = self._run_batch(batch)
            total_loss += loss
        return total_loss / len(self.train_loader)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def train(self, max_epochs: int):
        print(f'Training at GPU: {self.local_rank}')
        epoch_losses = []
        for epoch in range(max_epochs):
            epoch_loss = self._run_epoch(epoch)
            # Synchronize loss across processes
            all_reduce(epoch_loss, op=ReduceOp.SUM)
            epoch_loss /= self.world_size
            
            epoch_losses.append(epoch_loss.item())
            print(f"[GPU{self.local_rank}] Epoch {epoch} | Loss: {epoch_loss:.4f}")
        
        return epoch_losses
    
def prepare_dataloader(dataset: Dataset, batch_size: int, data_collator):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        collate_fn=data_collator        
    )

