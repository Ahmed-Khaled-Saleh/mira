import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
from utils.validation import rouge_score
from optimizers.mezo_torch import MeZOOptimizer
from optimizers.mezo_optimizer import MeZOFramework
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
        self.client.model.generation_config.pad_token_id = self.client.tokenizer.pad_token_id
        
        
    def _run_batch(self, batch):
        if not isinstance(self.client.optimizer, MeZOFramework):
            self.client.optimizer.zero_grad()

        def closure():
            input_length = batch['input_ids'].shape[1]
            if input_length  + 128 >= 1024:
                print("Warning: input length is too long")
                return torch.tensor(float(0), device=self.client.device)
            try:
                out = self.client.model(**batch)
                loss = self.client.criterion(out)
            except:
                print("Error in loss calculation")
                return torch.tensor(float(0), device=self.client.device)
            if torch.isnan(loss):
                print("Warning: NaN loss detected in closure")
                return torch.tensor(float(0), device=loss.device)
            return loss
        
        if isinstance(self.client.optimizer, MeZOOptimizer):
            loss, zo_random_seed, projected_grad = self.client.optimizer.step(closure)
            if torch.isnan(loss):
                print("Warning: NaN loss returned from optimizer step")
                return torch.tensor(float(0), device=loss.device)
        
            if torch.isnan(projected_grad).any():
                print("Warning: NaN detected in projected gradient after optimizer step")
                projected_grad = torch.zeros_like(projected_grad)

            self.client._add_seed_pole(zo_random_seed, projected_grad)

        elif isinstance(self.client.optimizer, MeZOFramework):
            logits, loss = self.client.optimizer.zo_step(batch, self.client.local_seed_pool)
            if torch.isnan(loss):
                print("Warning: NaN loss returned from optimizer step")
                return torch.tensor(float(0), device=loss.device)


        else:
            loss = closure()
            if loss.item() == 0:
                return loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.client.model.parameters(), max_norm=5.0)
            self.client.optimizer.step()
        
        return loss
    
    def add_models(self):
        for k, v in self.client.model.state_dict().items():
            if v.requires_grad:
                self.client.embedding[k].data.add_(v) 

    def avg_embed(self, epochs):
        for k, v in self.client.model.state_dict().items():
            if v.requires_grad:
                self.client.embedding[k].data.div_(epochs)


    def _run_epoch(self):
        total_loss = 0
        num_trained = 0
        progress_bar = tqdm(range(len(self.client.train_loader)))

        for i, batch in enumerate(self.client.train_loader):
                
                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }
                
                loss = self._run_batch(batch)

                if num_trained == 0:
                    num_trained = 1e-10

                print(f'Batch loss is {loss}')
                progress_bar.update(1)
                progress_bar.set_description(f'client {self.client.idx} total_losstrain at step {i}, loss: {total_loss / num_trained if num_trained != 0 else 0.0}')

                
                if loss.item() != 0:
                    total_loss += loss.item()
                    num_trained += len(batch['input_ids'])
    
        return total_loss / num_trained
    

    def _run_epoch_fed(self, local_iters):
        
        def local_train():
            total_loss = 0
            progress_bar = tqdm(range(local_iters))
            num_trained = 0
            
            
            for r in range(local_iters):
                print("local iteration: ", r)
                
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

                if num_trained == 0:
                    num_trained = 1e-10

                print(f'Batch loss is {loss}')
                progress_bar.update(1)
                progress_bar.set_description(f'client {self.client.idx} total_losstrain at step {r}, loss: {total_loss / num_trained if num_trained != 0 else 0.0}')

                
                if loss.item() != 0:
                    total_loss += loss.item()
                    num_trained += len(batch['input_ids'])

            return total_loss / num_trained
        

        if isinstance(self.client.optimizer, MeZOOptimizer):
            self.client.model = self.client.model.to(self.client.device)
            self.client.model.eval()
            with torch.inference_mode():
                avg_round_loss = local_train()
        
        else:
            avg_round_loss = local_train()
                    
                
        return avg_round_loss

    
    def train(self,
              fed= False,
              epochs= 10,
              local_iters= 1,
              memory_record_dic= None,
              callbacks= []):
        
        print('Inside the train () function of client ', self.client.idx)
        
        if callbacks:
            callbacks[0](memory_record_dic)

        init_val_loss = self.eval()
        val_loss = [init_val_loss]
        
        train_loss = []
        for _ in range(epochs):
            if self.client.args.name in ['mira_plus']:
                self.add_models()

            self.client.model = self.client.model.to(self.client.device)
            if fed and self.client.args.name in ['Na']:
                avg_train_loss = self._run_epoch_fed(local_iters)
            else:
                avg_train_loss = self._run_epoch()
            
            avg_val_loss = self.eval()
            val_loss.append(avg_val_loss)

            train_loss.append(avg_train_loss)

        if self.client.args.name in ['fedk']:
            self.client.model = None
        
        if callbacks:
            callbacks[1](memory_record_dic, self.client.device)

        if self.client.args.name in ['mira_plus']:
            self.avg_embed(epochs)

        return train_loss, val_loss
    
    def eval(self):
        total_loss = 0
        print("****************************************")
        print('Inside the eval () function of client ', self.client.idx)

        
        self.client.model = self.client.model.to(self.client.device)
        self.client.model.eval()
        num_eval = 0
        def _run_batch(batch):
            try:
                out = self.client.model(**batch)
                loss = self.client.criterion(out)
            except:
                print("Error in loss calculation in Eval() for client ", self.client.idx)

                return torch.tensor(float(0), device=self.client.device)
            if torch.isnan(loss):
                return torch.tensor(float(0), device=loss.device)
            return loss
        
        with torch.no_grad():
            for i, batch in enumerate(self.client.eval_loader):
                
                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }
                input_length = batch['input_ids'].shape[1]
                if input_length  + 128 >= 1024:
                    continue
                

                if num_eval == 0:
                    num_eval = 1e-10

                loss = _run_batch(batch)                 

                print(f"Client {self.client.idx}'s Batch loss inside eval() : {loss}")

                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    total_loss += loss.item()  
                    num_eval += len(batch['input_ids'])            
                
            print(f'Client {self.client.idx} Eval loss is : {total_loss / len(self.client.eval_loader)}')
            print("****************************************")

        if num_eval == 0:
            num_eval = 1e-10   
        return total_loss / num_eval
    
    def train_generate(self):
        print("****************************************")
        print('Inside the train_generate () function of client ', self.client.idx)

        self.client.model = self.client.model.to(self.client.device)
        self.client.model.eval()
        
        progress_bar_train = tqdm(range(len(self.client.train_loader_genr)))
        acc_total_train = 0.0
        num_train = 0
        total_items = len(self.client.train_loader_genr)

        with torch.no_grad():
            for batch in self.client.train_loader_genr:
                
                input_ids = batch['input_ids'].to(self.client.device)
                label_ids = batch['labels'].to(self.client.device)
                attention_mask=batch['attention_mask'].to(self.client.device)
                
                input_length = input_ids.shape[1]
                if input_length  + 128 >= 1024:
                    total_items -= 1
                    continue

                output_ids = self.client.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    num_beams=1,
                )
                
                generated_ids = output_ids[:, len(input_ids[0]):] 
                r_score = rouge_score(generated_ids, label_ids, self.client.tokenizer)  # noqa: F405

                acc_total_train += r_score
                
                print(f"Client {self.client.idx}'s Batch Rouge is : {r_score}")
                progress_bar_train.update(1)
           
        print(f'Client {self.client.idx} Rouge is : {acc_total_train / total_items}')
        print("****************************************")
        if total_items == 0:
            total_items = 1e-10
        return acc_total_train / total_items
    
    def eval_generate(self):
        print("****************************************")
        print('Inside the eval_generate () function of client ', self.client.idx)

        self.client.model = self.client.model.to(self.client.device)
        self.client.model.eval()
        
        progress_bar_eval = tqdm(range(len(self.client.eval_loader_genr)))
        acc_total_eval = 0.0
        total_items = len(self.client.eval_loader_genr)

        with torch.no_grad():
            for batch in self.client.eval_loader_genr:

                input_ids = batch['input_ids'].to(self.client.device)
                label_ids = batch['labels'].to(self.client.device)
                attention_mask=batch['attention_mask'].to(self.client.device)

                input_length = input_ids.shape[1]
                if input_length  + 128 >= 1024:
                    total_items -= 1
                    continue


                output_ids = self.client.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    num_beams=1,
                )

                generated_ids = output_ids[:, len(input_ids[0]):] 
                r_score = rouge_score(generated_ids, label_ids, self.client.tokenizer)  # noqa: F405

                acc_total_eval += r_score

                print(f"Client {self.client.idx}'s Batch Rouge is : {r_score}")
                progress_bar_eval.update(1)

        print(f'Client {self.client.idx} Rouge is : {acc_total_eval / total_items}')
        print("****************************************")
        
        if total_items == 0:
            total_items = 1e-10
        return acc_total_eval / total_items
    
    def prepare_dataloader(self, dataset, batch_size: int, data_collator):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator        
        )

