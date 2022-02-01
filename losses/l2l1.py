# -*- coding: utf-8 -*-
"""
Weighted L2 and L1 loss
"""

import torch
from easydict import EasyDict as edict

class L2L1Loss(torch.nn.Module):
    def __init__(self, betas=dict(l2=0.9, l1=0.1)):
        super(L2L1Loss,self).__init__()

        self.l2_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.betas = edict(betas)
        self._last = edict(l2=1e9, l1=1e9)

    def forward(self, y_hat, y):
        l2_loss = self.l2_criterion(y_hat, y)
        l1_loss = self.l1_criterion(y_hat, y)
        self._last.l2 = l2_loss.item()
        self._last.l1 = l1_loss.item()
        return self.betas.l2 * l2_loss + self.betas.l1 * l1_loss

    def item(self):
        return self.betas.l2 * self._last.l2 + self.betas.l1 * self._last.l1
