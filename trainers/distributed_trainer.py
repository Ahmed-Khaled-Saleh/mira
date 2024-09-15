import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

class DDPTrainer:
    def __init__(self, client, world_size):
        self.client = client
        self.world_size = world_size

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def prepare_dataloader(self, dataset, batch_size, data_collator):
        sampler = DistributedSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,  # Sampler handles shuffling
            collate_fn=data_collator,
            sampler=sampler
        )

    def run(self, rank, world_size):
        self.setup(rank, world_size)
        
        # Move model to the correct device
        self.client.model = self.client.model.to(rank)
        
        # Wrap model in DDP
        self.client.model = DDP(self.client.model, device_ids=[rank])
        
        # Prepare data loaders
        self.client.train_loader = self.prepare_dataloader(
            self.client.train_ds, 
            self.client.args.batch_size, 
            self.client.data_collator
        )
        self.client.eval_loader = self.prepare_dataloader(
            self.client.eval_ds, 
            self.client.args.batch_size, 
            self.client.data_collator
        )

        # Training loop
        for epoch in range(self.client.args.epochs):
            self.train_epoch(rank)
            if rank == 0:  # Only evaluate on one GPU
                self.evaluate(rank)

        self.cleanup()

    def train_epoch(self, rank):
        self.client.model.train()
        for batch in self.client.train_loader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            
            self.client.optimizer.zero_grad()
            loss = self._run_batch(batch)
            loss.backward()
            self.client.optimizer.step()

    def evaluate(self, rank):
        self.client.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.client.eval_loader:
                batch = {k: v.to(rank) for k, v in batch.items()}
                loss = self._run_batch(batch)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.client.eval_loader)
        print(f"Evaluation Loss: {avg_loss}")

    def _run_batch(self, batch):
        outputs = self.client.model(**batch)
        return self.client.criterion(outputs)

    def train(self, fed=False, epochs=10, local_iters=1, memory_record_dic=None, callbacks=[]):
        mp.spawn(self.run, args=(self.world_size,), nprocs=self.world_size, join=True)