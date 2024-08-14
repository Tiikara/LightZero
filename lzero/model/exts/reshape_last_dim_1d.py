import torch
from torch import nn


class ReshapeLastDim1D(nn.Module):

    def __init__(
            self,
            out_features: int
    ) -> None:
        super().__init__()

        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.out_features)

    def __repr__(self) -> str:
        return f"ReshapeLastDim1D(out_features={self.out_features})"
