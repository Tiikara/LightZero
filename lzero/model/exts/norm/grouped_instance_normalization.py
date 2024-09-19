import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedInstanceNormalization(nn.Module):

    def __init__(self, num_features: int, group_size: int) -> None:
        super().__init__()

        assert num_features % group_size == 0

        self.num_features = num_features
        self.num_groups = num_features // group_size
        self.group_size = group_size
        self.in_norm = nn.InstanceNorm1d(self.num_groups, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.num_features

        shp = x.shape

        x = x.view(-1, self.num_groups, self.group_size)

        return self.in_norm(x).view(shp)

    def __repr__(self) -> str:
        return f"GroupedInstanceNormalization(num_features={self.num_features}, num_groups=${self.num_groups})"
