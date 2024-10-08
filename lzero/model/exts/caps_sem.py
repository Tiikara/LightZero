import torch
import torch.nn as nn
import torch.nn.functional as F


class CapSEM(nn.Module):
    """
    CapsNet Regularization based on SEM (https://arxiv.org/abs/2204.00616)
    """
    def __init__(self, num_capsules, capsule_dim, group_size, eps=1e-6, temperature=1.0):
        super().__init__()

        assert num_capsules % group_size == 0

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_groups = num_capsules // group_size
        self.temperature = temperature
        self.eps = eps
        self.group_size = group_size

    def forward(self, x):
        shp = x.shape

        x = x.view(-1, self.num_groups, self.group_size, self.capsule_dim)

        norms = torch.norm(x, dim=-1)

        softmax_norms = F.softmax(norms / self.temperature, dim=-1)

        normalized_capsules = F.normalize(x, dim=-1, eps=self.eps)

        output = normalized_capsules * softmax_norms.unsqueeze(-1)

        return output.view(*shp)
