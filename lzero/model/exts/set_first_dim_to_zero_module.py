import torch
from torch import nn


class SetFirstDimToZeroModule(nn.Module):
    def __init__(
            self,
            emb
    ) -> None:
        super().__init__()
        self.emb = emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.view(-1, self.emb)[:, 0] = 0.

        return x
