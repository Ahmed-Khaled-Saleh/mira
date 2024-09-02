


class BaseClient:

    def __init__(self, 
                train_ds,
                eval_ds,
                model,
                criterion,
                optimizer):

        '''
        A client is defined as an object that contains those information:

        1- **Essentials**:
            Dataseet, Model, Criterion(Loss Function), and an Optimizer.
        2- **Extra**:
            Task-dependent and algrithm-specifics. usually defined in the chiild client not in the base.
        '''

        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer


    def loss(self, criterion, out, label):
        return self.criterion(out, label)


    def train():
        '''
            
        '''
        pass


    def eval():
        pass

