from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        """
        super().__init__()

        self.d = d

        self.weight = nn.Parameter(torch.ones(d))
        self.register_parameter("weight", self.weight)

    def forward(self, x):
        eps = torch.finfo(x.dtype).eps

        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + eps)

        return self.weight * x_normed
