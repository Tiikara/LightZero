import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ding.torch_utils import MLP, ResBlock
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
from .second_dim_check import SecondDimCheck
from .down_sample_res_net_coord import DownSampleResNetCoord

import torch
from torch import nn
import timm
from .caps_sem import CapSEM


class RepresentationNetworkUniZeroCapsnetResDownsampleCoord(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            num_res_blocks: int = 1,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            downsample: bool = True,
            num_channels: int = 64,
            num_capsules: int = 32
    ) -> None:
        """
        Overview:
            Representation network used in UniZero. Encode the 2D image obs into latent state.
            Currently, the network only supports obs images with both a width and height of 64.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - embedding_dim (:obj:`int`): The dimension of the latent state.
            - group_size (:obj:`int`): The dimension for simplicial normalization.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"
        logging.info(f"Using norm type: {norm_type}")
        logging.info(f"Using activation type: {activation}")

        self.observation_shape = observation_shape
        self.downsample = downsample

        self.downsample_net = DownSampleResNetCoord(
            observation_shape,
            num_channels,
            activation=activation,
            norm_type=norm_type,
        )

        self.activation = activation
        self.embedding_dim = embedding_dim

        assert self.embedding_dim % num_capsules == 0

        out_capsules_dim = self.embedding_dim // num_capsules

        self.out_capsules = (num_capsules, out_capsules_dim) # num_capsules x (embedding_dim / num_capsules) = embedding_dim
        self.caps = nn.Sequential(
            CapsInitialModule(
                in_channels=self.downsample_net.out_features,
                in_size=self.downsample_net.out_size,
                activation = activation,
                initial_capsule_size=self.out_capsules,
                out_capsules_size=self.out_capsules,
                bias=False
            ),
            CapSEM(
                num_capsules=self.out_capsules[0],
                capsule_dim=self.out_capsules[1],
                group_size=group_size
            )
        )

        self.out_create_layers = [
            lambda: SecondDimCheck(
                CapSEM(
                    num_capsules=self.out_capsules[0],
                    capsule_dim=self.out_capsules[1],
                    group_size=group_size
                )
            )
        ]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.downsample_net(x)

        x = self.caps(x)

        return x.reshape(-1, self.out_capsules[0] * self.out_capsules[1])
