from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAttention1D(nn.Module):
    def __init__(self, num_features: int, backbone_attention: nn.Module, paths: nn.ModuleList, affine: bool = True):
        super().__init__()

        self.backbone_attention = backbone_attention
        self.paths = paths
        self.num_features = num_features
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

            self.register_parameter("weight", self.weight)
            self.register_parameter("bias", self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, *data]
        shape = x.shape

        x = x.view(-1, self.num_features)

        att = self.backbone_attention(x)
        w = F.softmax(att, dim=-1)

        # norm
        x_normalized = torch.zeros_like(x)
        for k in range(len(self.paths)):
            x_normalized += w[:, k].unsqueeze(1) * self.paths[k](x)

        if self.affine:
            x_normalized = self.weight * x_normalized + self.bias

        return x_normalized.view(*shape)
