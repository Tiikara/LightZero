import torch
import torch.nn as nn
import torch.nn.functional as F


class SummaryModule(nn.Module):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer

    def forward(self, x):
        res_m = self.layer(x)
        return x + res_m
