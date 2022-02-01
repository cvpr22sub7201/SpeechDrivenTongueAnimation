# -*- coding: utf-8 -*-
"""
L2 distance + Velocity loss
"""

import torch
from easydict import EasyDict as edict


class L2Velocity(torch.nn.Module):
    def __init__(self, betas=dict(pose=0.9, velocity=0.1), sample_span=20):
        """[summary]

        Args:
            betas ([type], optional): [description]. Defaults to dict(l2=0.9, l1=0.1).
            sample_span (int, optional): [description]. Defaults to 20.
        """
        super(L2Velocity, self).__init__()

        self.mse = torch.nn.MSELoss()
        self.betas = edict(betas)
        self._last = edict(pose=1e9, velocity=1e9)
        self._sample_span = sample_span

    def forward(self, pose_hat, pose):
        vel_hat = pose_hat[:, 1:, :] - pose_hat[:, :-1, :] / self._sample_span
        vel = pose[:, 1:, :] - pose[:, :-1, :] / self._sample_span

        pose_loss = self.mse(pose_hat, pose)
        velocity_loss = self.mse(vel_hat, vel)
        self._last.pose = pose_loss.item()
        self._last.velocity = velocity_loss.item()

        return self.betas.pose * pose_loss + self.betas.velocity * velocity_loss

    def item(self):
        return self.betas.pose * self._last.pose + self.betas.velocity * self._last.velocity
