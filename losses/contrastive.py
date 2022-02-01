# -*- coding: utf-8 -*-
"""
This class was impelmented to compute the contrastive loss
over a batch of sequence predictions
"""

import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.eps = eps

    def diag_3d_tensor(self, input_tensor):
        """Gets diagonals from a batch of square matrices

        Args:
            input_tensor (torch.Tensor): tensor of size B x N x N
        Returns:
            float: contrastive loss value
        """
        batch_sz, seq_len, _ = input_tensor.shape
        mask = torch.eye(seq_len, dtype=torch.bool).unsqueeze(0).repeat(batch_sz, 1, 1)
        return torch.masked_select(input_tensor, mask).reshape(batch_sz, -1)

    def forward(self, y_hat, y):
        dist_tensor = torch.cdist(y_hat, y).exp()
        dist = self.diag_3d_tensor(dist_tensor)
        # TODO: verify that direction of sum is dim 2 and not 1
        return (dist / dist_tensor.sum(2)).mean() + self.eps
