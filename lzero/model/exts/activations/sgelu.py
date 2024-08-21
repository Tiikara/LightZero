import torch
import torch.nn as nn
import math

class SGELU(nn.Module):
    """
    https://arxiv.org/pdf/2305.07537

    Saturated Non-Monotonic Activation Functions
    Junjia Chen, Zhibin Pan
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x * ((1. + torch.erf(x / math.sqrt(2.))) / 2.), x)
