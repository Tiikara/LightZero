import torch
import torch.nn as nn
import torch.nn.functional as F


class CapSEM(nn.Module):
    def __init__(self, num_capsules, capsule_dim, num_groups, eps=1e-6, temperature=1.0):
        super().__init__()

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_groups = num_groups
        self.temperature = temperature
        self.eps = eps

        assert num_capsules % num_groups == 0
        self.capsules_per_group = num_capsules // num_groups

    def forward(self, x):
        shp = x.shape

        x = x.view(-1, self.num_groups, self.capsules_per_group, self.capsule_dim)

        norms = torch.norm(x, dim=-1)

        softmax_norms = F.softmax(norms / self.temperature, dim=-1)

        normalized_capsules = F.normalize(x, dim=-1, eps=self.eps)

        output = normalized_capsules * softmax_norms.unsqueeze(-1)

        return output.view(*shp)
