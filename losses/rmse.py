# -*- coding: utf-8 -*-
"""
Root Mean Square Loss
"""

import torch


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(RMSELoss,self).__init__()
        self.eps = eps
        self.criterion = torch.nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.criterion(y_hat, y) + self.eps)
