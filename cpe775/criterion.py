import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

import numpy as np
class RMSELoss(Module):
    left_eye_outer_corner = 36
    right_eye_outer_corner = 45

    def __init__(self, eps=1e-9):
        super(RMSELoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        distances = (target - input).norm(p=2, dim=-1)

        dio = self._inter_ocular_distance(target)

        return torch.mean(distances.mean(-1)/dio)

    def _inter_ocular_distance(self, target):
        left_eye_outer_corner = target[..., self.left_eye_outer_corner, :]
        right_eye_outer_corner = target[..., self.right_eye_outer_corner, :]
        return (left_eye_outer_corner - right_eye_outer_corner).norm(p=2, dim=-1)
