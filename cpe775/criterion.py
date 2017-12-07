import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

import numpy as np

class NRMSELoss(Module):

    eye_dist = {
        'outer_eyes_distance': [[36], [45]],
        'pupil_distance': [range(36, 42), range(42, 48)]
    }

    def __init__(self, eps=1e-9, norm='pupil_distance', weight=1):
        super(NRMSELoss, self).__init__()
        self.eps = eps
        self.norm = norm
        self.weight = weight

        if self.norm not in self.eye_dist:
            raise ValueError('norm {} not allowed'.format(norm))

    def forward(self, input, target):
        dio = self._get_eye_dist(target)

        distances = (target - input).norm(p=2, dim=-1)

        return self.weight*torch.mean(distances.mean(dim=-1)/dio)

    def _get_eye_dist(self, target):
        if self.norm is None:
            return 1

        left_eye = target[..., self.eye_dist[self.norm][0], :].mean(dim=-2)
        right_eye = target[..., self.eye_dist[self.norm][1], :].mean(dim=-2)

        return (left_eye - right_eye).norm(p=2, dim=-1)

class SmoothL1Loss(Module):

    def __init__(self, weight=1):
        super(SmoothL1Loss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        input = input.view(input.size(0), -1)
        target = target.view()

        x = (input - target)*self.weight

        x[x.abs() < 1] = 1/2*x[x.abs() < 1]**2
        x[x.abs() >= 1] = x[x.abs() >= 1].abs() - 1/2

        return x.sum(dim=-1).mean()

class WingLoss(Module):

    def __init__(self, weight=1, w=10, eps=2):
        super(WingLoss, self).__init__()
        self.weight = weight
        self.w = w
        self.eps = eps

    def forward(self, input, target):
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        x = (input - target)*self.weight
        x = x.abs()

        C = 1 - np.log(1 + self.w/self.eps)

        x[x < self.w] = self.w * torch.log(1 + x[x < self.w]/self.eps)
        x[x >= self.w] = x[x >= self.w] - C

        return x.sum(dim=-1).mean()
