from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Source(nn.Module):
    def __init__(self, args, model_old):
        super().__init__()
        self.model = model_old
        self.model.eval()


    def reset(self):
        pass


    def forward(self, x, target):
        with torch.no_grad():
            return self.model(x)
