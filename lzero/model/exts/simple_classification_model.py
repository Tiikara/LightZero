import torch
import torch.nn as nn
import torch.nn.functional as F

from lzero.model.common import SimNorm


class SimpleClassificationModel(nn.Module):
    def __init__(self, channels, activation, group_size):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            SimNorm(simnorm_dim=group_size)
        )

    def forward(self, x):
        return self.body(x)
