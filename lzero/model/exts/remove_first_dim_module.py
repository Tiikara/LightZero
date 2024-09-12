import torch
import torch.nn as nn
import torch.nn.functional as F


class RemoveFirstDimModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[..., 1:]

class RemoveFirstDimsModule(nn.Module):
    def __init__(self, number_dims):
        super().__init__()

        self.number_dims = number_dims

    def forward(self, x):
        return x[..., self.number_dims:]
