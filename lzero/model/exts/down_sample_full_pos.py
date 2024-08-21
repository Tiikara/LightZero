from typing import Optional

import torch
from ding.utils import SequenceType
from torch import nn

from .res_block_summary_positional import ResBlockSummaryPositional
from .torch_encodings import PositionalEncodingPermute2D, Summer

class DownSampleFullPos(nn.Module):

    def __init__(self, observation_shape: SequenceType, out_channels: int = 64,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN',
                 ) -> None:
        """
        Overview:
            Define downSample convolution network. Encode the observation into hidden state.
            This network is often used in video games like Atari. In board games like go and chess,
            we don't need this module.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[12, 96, 96]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - out_channels (:obj:`int`): The output channels of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`Optional[str]`): The normalization type used in network, defaults to 'BN'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.observation_shape = observation_shape

        self.conv1 = nn.Conv2d(
            observation_shape[0],
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,  # disable bias for better convergence
        )
        if norm_type == 'BN':
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm_type == 'LN':
            self.norm1 = nn.LayerNorm([out_channels // 2, observation_shape[-2] // 2, observation_shape[-1] // 2],
                                      eps=1e-5)

        self.resblocks1 = nn.ModuleList(
            [
                ResBlockSummaryPositional(
                    in_channels=out_channels // 2,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(1)
            ]
        )
        self.downsample_block = nn.Sequential(
            ResBlockSummaryPositional(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                activation=activation,
                norm_type=norm_type,
                res_type='downsample',
                bias=False
            )
        )

        self.resblocks2 = nn.ModuleList(
            [
                ResBlockSummaryPositional(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResBlockSummaryPositional(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = activation

        self.out_features = out_channels
        self.out_size = observation_shape[1] // 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)

        if self.observation_shape[1] == 64:
            output = x
        elif self.observation_shape[1] == 96:
            x = self.pooling2(x)
            output = x
        else:
            raise " Not supported"

        return output
