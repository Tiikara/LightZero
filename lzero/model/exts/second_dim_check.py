import torch
from torch import nn


class SecondDimCheck(nn.Module):
    def __init__(
            self,
            layer
    ) -> None:
        super().__init__()

        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 0:
            return self.layer(x)
        else:
            return x
