from typing import Optional

import torch
from ding.torch_utils import ResBlock
from torch import nn


class ResDownSampleLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN',
                 ) -> None:

        super().__init__()

        self.blocks = nn.Sequential(
            *[
                ResBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(num_blocks)
            ]
        )

        self.downsample = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            norm_type=norm_type,
            res_type='downsample',
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x
