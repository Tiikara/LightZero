import torch
import torch.nn as nn
import torch.nn.functional as F


class AddDimToStartModule(nn.Module):
    def __init__(self, value):
        super().__init__()

        self.value = value

    def forward(self, x):
        original_shape = x.shape

        new_dim = torch.full(original_shape[:-1] + (1,), self.value, dtype=x.dtype, device=x.device)

        result = torch.cat([new_dim, x], dim=-1)

        return result

class AddDimsToStartModule(nn.Module):
    def __init__(self, dims, value):
        super().__init__()

        self.dims = dims
        self.value = value

    def forward(self, x):
        original_shape = x.shape

        new_dim = torch.full(original_shape[:-1] + (self.dims,), self.value, dtype=x.dtype, device=x.device)

        result = torch.cat([new_dim, x], dim=-1)

        return result
