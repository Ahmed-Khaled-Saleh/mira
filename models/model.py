import torch
from torch import nn

class GPT2(nn.Module):
    def __init__(self, model, args, tokenizer):
        super(GPT2, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.zo_eps = self.args.zo_eps

    def forward(self, batch):
        self.model.eval()
        outputs = self.model(**batch)
        return outputs.loss