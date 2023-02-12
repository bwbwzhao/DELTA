from copy import deepcopy
import torch
import torch.nn as nn


class Norm(nn.Module):
    def __init__(self, args, model_old, eps=1e-5, reset_stats=False, no_stats=False):
        super().__init__()
        self.eps = eps
        self.reset_stats=reset_stats
        self.no_stats=no_stats

        self.model_old = model_old
        self.model_old.eval()
        self.model_old.requires_grad_(False)
        
        self.reset()


    def forward(self, x, target):
        out = self.model(x)
        # print(out.sum().item())
        return out


    def reset(self):
        self.model = deepcopy(self.model_old)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # use batch-wise statistics in forward
                m.train()
                # configure epsilon for stability, and momentum for updates
                # m.eps = self.eps
                # m.momentum = self.momentum
                # print(m.momentum)
                if self.reset_stats:
                    # reset state to estimate test stats without train stats
                    m.reset_running_stats()
                if self.no_stats:
                    # disable state entirely and use only batch stats
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
