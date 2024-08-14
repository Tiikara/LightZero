import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSimNorm(nn.Module):

    def __init__(self, simnorm_dim: int, temperature: float = 1.0, hard: bool = False) -> None:
        """
        Overview:
            Simplicial normalization using Gumbel-Softmax. Adapted from https://arxiv.org/abs/2204.00616.
        Arguments:
            - simnorm_dim (:obj:`int`): The dimension for simplicial normalization.
            - temperature (:obj:`float`): Temperature parameter for Gumbel-Softmax.
            - hard (:obj:`bool`): If True, use straight-through estimator for discrete output.
        """
        super().__init__()
        self.dim = simnorm_dim
        self.temperature = temperature
        self.hard = hard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the GumbelSimNorm layer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor to normalize.
        Returns:
            - x (:obj:`torch.Tensor`): The normalized tensor.
        """
        shp = x.shape
        # Ensure that there is at least one simplex to normalize across.
        if shp[1] != 0:
            x = x.view(*shp[:-1], -1, self.dim)
            x = F.gumbel_softmax(x, tau=self.temperature, hard=self.hard, dim=-1)
            return x.view(*shp)
        else:
            return x

    def __repr__(self) -> str:
        return f"GumbelSimNorm(dim={self.dim})"
