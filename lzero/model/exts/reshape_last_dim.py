import torch
from torch import nn


class ReshapeLastDim(nn.Module):

    def __init__(
            self,
            out_shape: tuple
    ) -> None:
        super().__init__()

        self.out_shape = out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *self.out_shape)

    def __repr__(self) -> str:
        return f"ReshapeLastDim(shape={self.out_shape})"
