import torch
from torch import nn

from lzero.model.exts.summary_module import SummaryModule

class ResFeedForwardBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            bias=True,
            dropout: float = None
    ):
        super().__init__()

        self.body = nn.Sequential(
            SummaryModule(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_channels, bias=bias),
                    activation,
                    nn.Linear(hidden_channels, in_channels, bias=bias)
                )
            ),
            nn.LayerNorm(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
