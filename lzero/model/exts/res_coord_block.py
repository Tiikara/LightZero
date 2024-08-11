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

import torch
from torch import nn
import timm
from .caps_sem import CapSEM
from typing import Union
from .res_block_channelled import ResBlockChannelled

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
                in_channels = in_channels + 2,
                out_channels = out_channels,
                activation = activation,
                norm_type = norm_type,
                res_type = res_type,
                bias=bias
            )
        elif res_type == 'basic':
            self.res_block = nn.Sequential(
                ResBlockChannelled(
                    in_channels = in_channels + 2,
                    out_channels = out_channels,
                    activation = activation,
                    norm_type = norm_type,
                    bias=bias
                )
            )
        else:
            raise res_type + ' not supported'

        self.add_coords = AddCoords(
            rank = 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_coords(x)

        return self.res_block(x)
