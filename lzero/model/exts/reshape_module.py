import torch
from torch import nn


class ReshapeModule(nn.Module):

    def __init__(
            self,
            out_shape: tuple
    ) -> None:
        super().__init__()

        self.out_shape = out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*self.out_shape)

    def __repr__(self) -> str:
        return f"ReshapeModule(shape={self.out_shape})"
