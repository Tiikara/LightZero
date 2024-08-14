import torch
from torch import nn


class CatLayersModule(nn.Module):
    def __init__(
            self,
            layers,
            dim
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []

        for layer in self.layers:
            res.append(layer(x))

        return torch.cat(res, dim=self.dim)

    def __repr__(self) -> str:
        return f"CatLayersModule(dim={self.dim})"
