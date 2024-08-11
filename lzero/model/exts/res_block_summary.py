import torch
from ding.torch_utils import ResBlock
from torch import nn

from .torch_encodings import PositionalEncodingPermute2D, Summer


class ResBlockSummaryPositional(nn.Module):
    def __init__(self,
                 in_channels,
                 activation,
                 norm_type,
                 res_type,
                 bias,
                 out_channels=None,
                 ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.res_block = ResBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            activation = activation,
            norm_type = norm_type,
            res_type = res_type,
            bias=bias
        )
        self.positional = Summer(PositionalEncodingPermute2D(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional(x)

        return self.res_block(x)
