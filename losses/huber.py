# -*- coding: utf-8 -*-
"""
Huber Loss, not implemented PyTorch < 1.9
"""

import torch

class HuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss,self).__init__()

        self.l2_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.delta = delta

    def forward(self, y_hat, y):
        l2_loss = self.l2_criterion(y_hat, y)
        l1_loss = self.l1_criterion(y_hat, y)
        return l1_loss if l1_loss.item() < self.delta else l2_loss