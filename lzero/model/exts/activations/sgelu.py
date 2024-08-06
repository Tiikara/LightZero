import torch.nn as nn

class SGELU(nn.Module):
    """
    https://arxiv.org/pdf/2305.07537

    Saturated Non-Monotonic Activation Functions
    Junjia Chen, Zhibin Pan
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x * ((1. + torch.erf(x / torch.sqrt(2))) / 2.), x)
