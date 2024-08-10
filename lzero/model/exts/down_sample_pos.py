import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ding.torch_utils import MLP, ResBlock
from ding.torch_utils.nn_module import conv2d_block
from ding.utils import SequenceType
from ditk import logging
from lzero.model.common import SimNorm
from lzero.model.common import NormByType
from functools import partial
from .efficientnet_v2 import MBConvConfig, MBConv, ConvBNAct
from .capsnet_layers import PrimaryCaps, RoutingCaps, Squash
import math
from functools import partial
from collections import OrderedDict

from lzero.model.common import DownSample
from .capsnet_ext_modules import CapsInitialModule
from .torch_encodings import PositionalEncodingPermute2D, Summer
from .coordconv import AddCoords

import torch
from torch import nn
import timm
from .caps_sem import CapSEM
from typing import Union

class ResBlockChannelled(nn.Module):
    def __init__(
            self,
            in_channels: int,
            activation: nn.Module = nn.ReLU(),
            norm_type: str = 'BN',
            bias: bool = True,
            out_channels: Union[int, None] = None,
    ) -> None:
        """
        Overview:
            Init the 2D convolution residual block.
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor.
            - activation (:obj:`nn.Module`): The optional activation function.
            - norm_type (:obj:`str`): Type of the normalization, default set to 'BN'(Batch Normalization), \
                supports ['BN', 'LN', 'IN', 'GN', 'SyncBN', None].
            - res_type (:obj:`str`): Type of residual block, supports ['basic', 'bottleneck', 'downsample']
            - bias (:obj:`bool`): Whether to add a learnable bias to the conv2d_block. default set to True.
            - out_channels (:obj:`int`): Number of channels in the output tensor, default set to None, \
                which means out_channels = in_channels.
        """
        super(ResBlock, self).__init__()
        self.act = activation
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = conv2d_block(
            in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
        )
        self.conv2 = conv2d_block(
            out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
        )
        self.conv3 = conv2d_block(in_channels, out_channels, 3, 1, 1, activation=None, norm_type=None, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        identity = self.conv3(identity)
        x = self.act(x + identity)
        return x

class ResCoordBlock(nn.Module):
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

        if res_type == 'downsample':
            self.res_block = ResBlock(
                in_channels = in_channels + 3,
                out_channels = out_channels,
                activation = activation,
                norm_type = norm_type,
                res_type = res_type,
                bias=bias
            )
        else:
            self.res_block = nn.Sequential(
                ResBlockChannelled(
                    in_channels = in_channels + 3,
                    out_channels = in_channels,
                    activation = activation,
                    norm_type = norm_type,
                    bias=bias
                ),
                ResBlock(
                    in_channels = in_channels + 3,
                    out_channels = out_channels,
                    activation = activation,
                    norm_type = norm_type,
                    res_type = res_type,
                    bias=bias
                )
            )

        self.add_coords = AddCoords(
            rank = 2,
            with_r=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_coords(x)

        return self.res_block(x)

class DownSamplePos(nn.Module):

    def __init__(self, observation_shape: SequenceType, out_channels: int,
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

        self.add_coords = AddCoords(
            rank=2,
            with_r=True
        )

        self.conv1 = nn.Conv2d(
            observation_shape[0] + 3,
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
                ResCoordBlock(
                    in_channels=out_channels // 2,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(1)
            ]
        )
        self.downsample_block = nn.Sequential(
            ResCoordBlock(
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
                ResCoordBlock(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResCoordBlock(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.add_coords(x)

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
            x = self.pooling2(x)
            output = x

        return output
