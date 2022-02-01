# -*- coding: utf-8 -*-
"""
P-Norm Loss
"""

import torch
from easydict import EasyDict as edict

class PNormLoss(torch.nn.Module):
    def __init__(self, p=1.5):
        super(PNormLoss,self).__init__()

        self.p_value = p
        self._last = float('inf')
    
    def forward(self, y_hat, y):
        res = torch.dist(y_hat, y, self.p)
        self._last = res.detach()
        return res

    def item(self):
        return self._last
