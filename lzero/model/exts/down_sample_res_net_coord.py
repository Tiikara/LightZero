import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ding.torch_utils import MLP, ResBlock
from ding.torch_utils.network.nn_module import conv2d_block
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
from .res_down_sample_block import ResDownSampleBlock

import torch
from torch import nn
import timm
from .caps_sem import CapSEM
from typing import Union


class DownSampleResNetCoord(nn.Module):

    def __init__(self, observation_shape: SequenceType,
                 out_channels: int,
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
        assert observation_shape[1] == observation_shape[2]

        self.observation_shape = observation_shape

        assert observation_shape[1] % 2 == 0

        current_channels = 32
        current_size = observation_shape[1] // 2

        downsamples = [
            AddCoords(
                rank=2
            ),
            ResDownSampleBlock(
                in_channels=observation_shape[0] + 2,
                out_channels=current_channels,
                activation=activation,
                norm_type=norm_type
            )
        ]

        while current_size > 4:
            new_channels = current_channels * 2
            current_size = current_size // 2

            downsamples.append(
                ResDownSampleBlock(
                    in_channels=current_channels,
                    out_channels=new_channels,
                    activation=activation,
                    norm_type=norm_type
                )
            )

            current_channels = new_channels

        self.body = nn.Sequential(*downsamples)

        self.out_features = current_channels
        self.out_size = current_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        return self.body(x)
