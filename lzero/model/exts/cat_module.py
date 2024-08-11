import torch
from torch import nn

class CatModule(nn.Module):
    def __init__(
            self,
            layer,
            dim
    ) -> None:
        super().__init__()
        self.layer = layer,
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.layer(x)], dim=1)
