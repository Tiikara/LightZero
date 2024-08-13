import torch
from torch import nn


class ReturnShapeModule(nn.Module):

    def __init__(
            self,
            inner: nn.Module
    ) -> None:
        super().__init__()

        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape

        x = self.inner(x)

        return x.reshape(*shp)
