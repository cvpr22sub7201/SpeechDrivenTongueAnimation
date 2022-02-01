# -*- coding: utf-8 -*-
"""
ShrinkageLoss
"""
import torch
import torch.nn as nn

class ShrinkageLoss(nn.Module):
    """ ShrinkageLoss class.
        Modified version of shrinkage loss tailored to images:
        http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf
        It basically computes a point-wise shrinkage loss.
    """
    def __init__(self, speed=10.0, loc=0.2, verbose=False):
        """ Initialize ShrinkageLoss class with user-defined parameters.
        Arguments:
            shrink_speed (float): Shrinkage speed, i.e., weight assigned to hard samples.
            shrink_loc (float):   Shrinkage localization, i.e., threshold for hard mining.
            verbose (bool):       Whether  the log will be shown in the shell.
        """
        nn.Module.__init__(self)
        self.shrink_speed = speed
        self.shrink_loc = loc
        
    def forward(self, estimate, ground_truth):
        """ Calculate shrinkage loss between the estimate and grount truth, if any.
            Otherwise, the loss is computed using the estimate, which is already
            the difference to the ground truth or the parameters.
        Arguments:
            estimate (tensor):     Estimate or delta (MxC, where M, C are
                                   the number of points and channels, respectively).
            ground_truth (tensor): Ground truth (optional). MxC, where M, C are
                                   the number of points and channels, respectively
        Return:
            Mean per-point shrinkage loss (float)
        """
        # Compute point errors (l2 norm).
        l2_loss = torch.norm(estimate - ground_truth, p=2, dim=1)
        
        # Compute mean shrinkage loss.
        shrink_loss = torch.mul(l2_loss,l2_loss)/(
            1.0 + torch.exp(self.shrink_speed*(self.shrink_loc - l2_loss)))
        return torch.mean(shrink_loss)