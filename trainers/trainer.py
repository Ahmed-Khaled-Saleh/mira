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
            try:
                out = self.client.model(**batch)
                loss = self.client.criterion(out)
            except:
                print(batch)
                import ipdb; ipdb.set_trace()
            print(f"Closure: Loss calculated, shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
            return loss
        
        if self.client.args.name in ['fedk', 'mira']:

            loss, zo_random_seed, projected_grad = self.client.optimizer.step(closure)
            self.client._add_seed_pole(zo_random_seed, projected_grad)

            # try:
            #     loss, zo_random_seed, projected_grad = self.client.optimizer.step(closure)
            # except TypeError as e:
            #     print(f"Error in optimizer step: {e}")
            #     print(f"Closure: {closure}")
            #     print(f"Optimizer state: {self.client.optimizer.state_dict()}")
            # raise
        
        else:
            loss = closure()
            loss.backward()
            self.client.optimizer.step()
        
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
                    total_loss += loss
                
                if i % 1000 == 999:
                    last_loss = total_loss / 1000 
                    progress_bar.update(i)
                    progress_bar.set_description(f'client {self.client.idx} Fuly Local Training , loss: {last_loss}')
    
        return total_loss / len(self.client.train_loader)
    

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
                num_trained += len(batch['input_ids'])

                if num_trained == 0:
                    num_trained = 1e-10

                print(f'Batch loss is {loss}')
                progress_bar.update(1)
                progress_bar.set_description(f'client {self.client.idx} total_losstrain at step {r}, loss: {total_loss / num_trained if num_trained != 0 else 0.0}')

                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    total_loss += loss
                    num_trained += len(batch['input_ids'])

            return total_loss / num_trained
        

        if self.client.args.name in ['fedk', 'mira']:
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

        val_loss = self.eval()

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

        return train_loss, val_loss
    
    def eval(self):
        total_loss = 0
        print("****************************************")
        print('Inside the eval () function of client ', self.client.idx)

        if self.client.args.name in ['fedk', 'mira']:
            self.client.model = self.client.model.to(self.client.device)
        self.client.model.eval()
        
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
                print(f"Client {self.client.idx}'s Batch loss inside eval() : {loss}")

                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    total_loss += loss              
                
            print(f'Client {self.client.idx} Eval loss is : {total_loss / len(self.client.eval_loader)}')
            print("****************************************")
                
        return (total_loss / len(self.client.eval_loader)).item()
    
    def train_generate(self):
        print("****************************************")
        print('Inside the train_generate () function of client ', self.client.idx)

        if self.client.args.name in ['fedk', 'mira']:
            self.client.model = self.client.model.to(self.client.device)
        self.client.model.eval()
        
        progress_bar_train = tqdm(range(len(self.client.train_loader_genr)))
        acc_total_train = 0.0
        num_train = 0
        with torch.no_grad():
            for batch in self.client.train_loader_genr:
                input_ids = batch['input_ids'].to(self.client.device)
                label_ids = batch['labels'].to(self.client.device)
                output_ids = self.client.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_train += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.client.tokenizer)
                
                print(f"Client {self.client.idx}'s Batch accuracy is : {acc_total_train / len(batch['input_ids'])}")
                progress_bar_train.update(1)

                num_train += len(batch['input_ids'])
                if num_train == 0:
                    num_train = 1e-10
            
        print(f'Client {self.client.idx} accuracy is : {acc_total_train / num_train}')
        print("****************************************")
        return acc_total_train / num_train
    
    def eval_generate(self):
        print("****************************************")
        print('Inside the eval_generate () function of client ', self.client.idx)

        if self.client.args.name in ['fedk', 'mira']:
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
                print(f"Client {self.client.idx}'s Batch accuracy is : {acc_total_eval / len(batch['input_ids'])}")
                progress_bar_eval.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10

        print(f'Client {self.client.idx} accuracy is : {acc_total_eval / num_eval}')
        print("****************************************")
        return acc_total_eval / num_eval
    
    def prepare_dataloader(self, dataset, batch_size: int, data_collator):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            # pin_memory=True,
            shuffle=True,
            collate_fn=data_collator        
        )

