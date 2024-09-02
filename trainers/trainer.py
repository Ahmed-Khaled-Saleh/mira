import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
from utils.validation import rouge_score

class Trainer:
    def __init__(
        self,
        client
    ) -> None:
        
        '''
            The Trainer class implements the training loop with the train() function.
            It accepts a client at initialization, which contains all necessary infromation 
            to implement a training loop for a federated learning setup.
        '''
        
        self.client = client
        self.client.train_loader = self.prepare_dataloader(self.client.train_ds, self.client.args.batch_size, self.client.data_collator)
        self.client.eval_loader = self.prepare_dataloader(self.client.eval_ds, self.client.args.batch_size, self.client.data_collator)
        self.client.train_loader_genr = self.prepare_dataloader(self.client.train_ds_genr, self.client.args.batch_size, self.client.data_collator)
        self.client.eval_loader_genr = self.prepare_dataloader(self.client.eval_ds_genr, self.client.args.batch_size, self.client.data_collator)
        self.client.train_iterator = iter(self.client.train_loader)
        
    def _run_batch(self, batch):
        self.client.optimizer.zero_grad()
        def closure():
            out = self.client.model(**batch)
            return self.client.criterion(out)

        loss, zo_random_seed, projected_grad = self.client.optimizer.step(closure)
        self.client._add_seed_pole(zo_random_seed, projected_grad)

        if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
            return loss
        return 0
    
    def _run_epoch(self):
        total_loss = 0
        progress_bar = tqdm(range(len(self.client.train_loader)))

        with torch.inference_mode():
            for i, batch in enumerate(self.client.train_loader):

                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }

                loss = self._run_batch(batch)
                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    continue
                total_loss += loss
                
                if i % 1000 == 999:
                    last_loss = total_loss / 1000 
                    progress_bar.update(i)
                    progress_bar.set_description(f'client {self.client.idx} Fuly Local Training , loss: {last_loss}')
    
        return total_loss / len(self.client.train_loader)
    

    def _run_epoch_fed(self, local_iters):
        total_loss = 0
        progress_bar = tqdm(range(local_iters))

        with torch.inference_mode():
            for r in range(local_iters):
                print("local iteration: ", r)
                num_trained = 0
                try:
                    batch = next(self.client.train_iterator)
                except StopIteration:
                    self.client.train_iterator = iter(self.client.train_loader)
                    batch = next(self.client.train_iterator)
                
                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }
                
                loss = self._run_batch(batch)
                print(f'Batch loss is {loss}')
                progress_bar.update(1)
                progress_bar.set_description(f'client {self.client.idx} train at step {r}, loss: {total_loss / num_trained if num_trained != 0 else 0.0}')

                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    total_loss += loss
                    num_trained += len(batch['input_ids'])

            if num_trained == 0:
                num_trained = 1e-10

        avg_round_loss = total_loss / num_trained
                
        return avg_round_loss

    
    def train(self,
              fed= False,
              epochs= 10,
              local_iters= 1,
              memory_record_dic= None,
              callbacks= []):
        
        print('Inside the train () function of client ', self.client.idx)
        self.client.model.to(self.client.device)

        if callbacks:
            callbacks[0](memory_record_dic)
        
        self.client.model.eval()

        val_loss = self.eval()
        train_acc = self.train_generate()
        eval_acc = self.eval_generate()

        train_loss = []
        for _ in range(epochs):

            if fed:
                avg_train_loss = self._run_epoch_fed(local_iters)
            else:
                avg_train_loss = self._run_epoch()

            train_loss.append(avg_train_loss.item())

        self.client.model = None
        
        if callbacks:
            callbacks[1](memory_record_dic, self.client.device)

        return train_loss, val_loss, train_acc, eval_acc
    
    def eval(self):
        total_loss = 0

        def _run_batch(batch):
            out = self.client.model(**batch)
            loss = self.client.criterion(out)
            return loss
        
        with torch.no_grad():
            for i, batch in enumerate(self.client.eval_loader):
                
                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }
                
                loss = _run_batch(batch)

                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    continue
                total_loss += loss              

        return total_loss / len(self.client.eval_loader)
    
    def train_generate(self):
        self.client.model = self.client.model.to(self.client.device)
        self.client.model.eval()
        
        progress_bar_train = tqdm(range(len(self.client.train_loader_genr)))
        acc_total_train = 0.0
        num_train = 0
        
        for batch in self.client.train_loader_genr:
            input_ids = batch['input_ids'].to(self.client.device)
            label_ids = batch['labels'].to(self.client.device)
            output_ids = self.client.model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                num_beams=1,
            )
            acc_total_train += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.client.tokenizer)
            progress_bar_train.update(1)
            num_train += len(batch['input_ids'])
            if num_train == 0:
                num_train = 1e-10
        print()
        print()
        # self.client.model = self.client.model.cpu()
        return acc_total_train / num_train
    
    def eval_generate(self):
        self.client.model = self.client.model.to(self.client.device)
        self.client.model.eval()
        
        progress_bar_eval = tqdm(range(len(self.client.eval_loader_genr)))
        acc_total_eval = 0.0
        num_eval = 0
        
        with torch.no_grad():
            for batch in self.client.eval_loader_genr:
                input_ids = batch['input_ids'].to(self.client.device)
                label_ids = batch['labels'].to(self.client.device)
                output_ids = self.client.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_eval += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.client.tokenizer)  # noqa: F405
                progress_bar_eval.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
        print()
        print()
        return acc_total_eval / num_eval
    
    def prepare_dataloader(self, dataset, batch_size: int, data_collator):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            # pin_memory=True,
            shuffle=True,
            collate_fn=data_collator        
        )

