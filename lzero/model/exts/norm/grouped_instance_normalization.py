import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedInstanceNormalization(nn.Module):

    def __init__(self, num_features: int, num_groups: int) -> None:
        super().__init__()

        self.num_groups = num_groups
        self.num_features = num_features
        self.in_norm = nn.InstanceNorm1d(num_groups, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) % self.num_groups == 0

        shp = x.shape

        x = x.view(-1, self.num_groups, x.size(-1) // self.num_groups)

        return self.in_norm(x).view(shp)

    def __repr__(self) -> str:
        return f"GroupedInstanceNormalization(num_features={self.num_features}, num_groups=${self.num_groups})"
