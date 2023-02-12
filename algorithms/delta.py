from codecs import BOM_BE
from copy import deepcopy
from curses import noecho
from pickle import NEWOBJ_EX
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# inspired by https://github.com/bethgelab/robustness/tree/aa0a6798fe3973bae5f47561721b59b39f126ab7
def find_bns(parent, prior):
    replace_mods = []
    if parent is None:
        return []
    for name, child in parent.named_children():
        if isinstance(child, nn.BatchNorm2d):
            module = TBR(child, prior).cuda()
            replace_mods.append((parent, name, module))
        else:
            replace_mods.extend(find_bns(child, prior))
    return replace_mods


class TBR(nn.Module):
    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.prior = prior
        self.rmax = 3.0
        self.dmax = 5.0
        self.tracked_num = 0
        # self.running_mean = deepcopy(layer.running_mean)
        # self.running_std = deepcopy(torch.sqrt(layer.running_var) + 1e-5)
        self.running_mean = None
        self.running_std = None

    def forward(self, input):
        batch_mean = input.mean([0, 2, 3])
        batch_std = torch.sqrt(input.var([0, 2, 3], unbiased=False) + self.layer.eps)

        if self.running_mean is None:
            self.running_mean = batch_mean.detach().clone()
            self.running_std = batch_std.detach().clone()

        r = (batch_std.detach() / self.running_std) #.clamp_(1./self.rmax, self.rmax)
        d = ((batch_mean.detach() - self.running_mean) / self.running_std) #.clamp_(-self.dmax, self.dmax)
        
        input = (input - batch_mean[None,:,None,None]) / batch_std[None,:,None,None] * r[None,:,None,None] + d[None,:,None,None]
        # input = (input - self.running_mean[None,:,None,None]) / self.running_std[None,:,None,None]

        # if len(input)>=128:
        self.running_mean = self.prior * self.running_mean + (1. - self.prior) * batch_mean.detach()
        self.running_std = self.prior * self.running_std + (1. - self.prior) * batch_std.detach()
        # else:
        #     print('too small batch size, using last step model directly...')

        self.tracked_num+=1

        return input * self.layer.weight[None,:,None,None] + self.layer.bias[None,:,None,None]


class DELTA(nn.Module):
    def __init__(self, args, model_old):
        super().__init__()
        self.args = args

        self.model_old = model_old
        self.model_old.eval()
        self.model_old.requires_grad_(False)
        self.reset()


    def reset(self):
        self.model = deepcopy(self.model_old)

        if self.args.norm_type=='rn':
            replace_mods = find_bns(self.model, self.args.old_prior)
            for (parent, name, child) in replace_mods:
                setattr(parent, name, child)
        elif self.args.norm_type=='bn_training':
            self.model.train()
            for nm, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        self.model.requires_grad_(False)
        params = []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        p.requires_grad_(True)
                        params.append(p)
        
        if self.args.optim_type=='adam':
            self.optimizer = torch.optim.Adam(params, lr=self.args.optim_lr, betas=(self.args.optim_momentum, 0.999), weight_decay=self.args.optim_wd)
        elif self.args.optim_type=='sgd':
            self.optimizer = torch.optim.SGD(params, lr=self.args.optim_lr, momentum=self.args.optim_momentum, weight_decay=self.args.optim_wd)

        self.qhat = torch.zeros(1, self.args.class_num).cuda() + (1. / self.args.class_num)


    def forward(self, x, target):
        with torch.enable_grad():
            outputs = self.model(x)

            p = F.softmax(outputs, dim=1)
            p_max, pls = p.max(dim=1)
            logp = F.log_softmax(outputs, dim=1)

            ent_weight = torch.ones_like(pls)
            if self.args.loss_type=='entropy':
                entropys = -(p * logp).sum(dim=1)
                if self.args.ent_w:
                    ent_weight = torch.exp(math.log(self.args.class_num) * 0.5 - entropys.clone().detach())
                    use_idx = (ent_weight>1.)
                else:
                    use_idx = (entropys==entropys)
            elif self.args.loss_type=='cross_entropy':
                entropys = F.cross_entropy(outputs, pls, reduction='none')
                use_idx = (p_max>0.4)
            
            if self.args.dot is not None:
                class_weight = 1. / self.qhat
                class_weight = class_weight / class_weight.sum()
                sample_weight = class_weight.gather(1, pls.view(1,-1)).squeeze()
                sample_weight = sample_weight / sample_weight.sum() * len(pls)
                ent_weight = ent_weight * sample_weight

            loss = (entropys * ent_weight)[use_idx].mean()

            if use_idx.sum()!=0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.args.dot is not None:
                with torch.no_grad():
                    self.qhat = self.args.dot * self.qhat + (1. - self.args.dot) * F.softmax(outputs, dim=1).mean(dim=0, keepdim=True)
                    
            return outputs
