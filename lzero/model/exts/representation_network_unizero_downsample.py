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
from .capsnet_ext_modules import CapsInitialModule, CapsInitialModuleForward1D, PrimaryCapsForward1D
from .second_dim_check import SecondDimCheck
from .down_sample_res_net import DownSampleResNet
from .reshape_last_dim_1d import ReshapeLastDim1D
from .return_shape_module import ReturnShapeModule
from .reshape_last_dim import ReshapeLastDim
from .base_down_sample import BaseDownSample
from .gumbel_simnorm import GumbelSimNorm
from .cat_layers_module import CatLayersModule

import torch
from torch import nn
import timm
from .caps_sem import CapSEM
from .torch_encodings import Summer, PositionalEncodingPermute2D


class RepresentationNetworkUniZeroDownsample(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            downsample: bool = True,
            num_capsules: int = 32,
            use_linear_input_for_caps: bool = False,
            double_linear_input_for_caps: bool = False,
            use_routing: bool = True,
            use_squash_in_transformer: bool = False,
            downsample_network_config=None,
            head_type: str = None,
            head_config=None
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

        if downsample_network_config.type == 'base':
            self.downsample_net = BaseDownSample(
                observation_shape=observation_shape,
                activation=activation,
                norm_type=norm_type
            )
        elif downsample_network_config.type == 'res_net':
            res_net_config = downsample_network_config.res_net

            self.downsample_net = DownSampleResNet(
                observation_shape,
                start_channels=res_net_config.start_channels,
                activation=activation,
                norm_type=norm_type,
                use_coords=res_net_config.use_coords,
                channels_scale=res_net_config.channels_scale,
                num_blocks=res_net_config.num_blocks
            )
        else:
            raise "Not supported " + downsample_network_config.type

        self.activation = activation
        self.embedding_dim = embedding_dim

        if head_type == 'caps':
            assert self.embedding_dim % num_capsules == 0

            out_capsules_dim = self.embedding_dim // num_capsules

            # num_capsules x (embedding_dim / num_capsules) = embedding_dim
            self.out_capsules = (num_capsules, out_capsules_dim)

            if use_linear_input_for_caps:
                caps = [
                    ReshapeLastDim1D(
                        out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                    )
                ]

                if double_linear_input_for_caps:
                    caps += [
                        nn.Linear(
                            self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                            self.out_capsules[0] * self.out_capsules[1],
                            bias=False
                        ),
                        activation,
                        nn.Linear(
                            self.out_capsules[0] * self.out_capsules[1],
                            self.out_capsules[0] * self.out_capsules[1],
                            bias=False
                        )
                    ]
                else:
                    caps += [
                        nn.Linear(
                            self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                            self.out_capsules[0] * self.out_capsules[1],
                            bias=False
                        )
                    ]

                caps += [
                    PrimaryCapsForward1D(
                        capsule_size=self.out_capsules
                    )
                ]

                if use_routing:
                    caps += [
                        RoutingCaps(
                            in_capsules=self.out_capsules,
                            out_capsules=self.out_capsules,
                            bias=False
                        )
                    ]

                caps += [
                    CapSEM(
                        num_capsules=self.out_capsules[0],
                        capsule_dim=self.out_capsules[1],
                        group_size=group_size
                    ),
                    ReshapeLastDim1D(
                        out_features=self.out_capsules[0] * self.out_capsules[1]
                    ),
                ]

                self.head = nn.Sequential(*caps)
            else:
                assert use_routing is True

                self.head = nn.Sequential(
                    CapsInitialModule(
                        in_channels=self.downsample_net.out_features,
                        in_size=self.downsample_net.out_size,
                        activation=activation,
                        initial_capsule_size=self.out_capsules,
                        out_capsules_size=self.out_capsules,
                        bias=False
                    ),
                    CapSEM(
                        num_capsules=self.out_capsules[0],
                        capsule_dim=self.out_capsules[1],
                        group_size=group_size
                    ),
                    ReshapeLastDim1D(
                        out_features=self.out_capsules[0] * self.out_capsules[1]
                    ),
                )

            if use_squash_in_transformer:
                self.out_create_layers = [
                    lambda: SecondDimCheck(
                        ReturnShapeModule(
                            nn.Sequential(
                                ReshapeLastDim(
                                    out_shape=self.out_capsules
                                ),
                                Squash(),
                                CapSEM(
                                    num_capsules=self.out_capsules[0],
                                    capsule_dim=self.out_capsules[1],
                                    group_size=group_size
                                )
                            )
                        )
                    )
                ]
            else:
                self.out_create_layers = [
                    lambda: SecondDimCheck(
                        CapSEM(
                            num_capsules=self.out_capsules[0],
                            capsule_dim=self.out_capsules[1],
                            group_size=group_size
                        )
                    )
                ]
        elif head_type == 'gumbel_simnorm':
            self.head = nn.Sequential(
                    ReshapeLastDim1D(
                        out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                    ),
                    nn.Linear(self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size, self.embedding_dim, bias=False),
                    GumbelSimNorm(simnorm_dim=group_size)
                )

            self.out_create_layers = [
                lambda: GumbelSimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'simnorm':
            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size, self.embedding_dim, bias=False),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'simnorm_positional':
            simnorm_positional_config = head_config.simnorm_positional

            if simnorm_positional_config.pool_type == 'max':
                pool = nn.AdaptiveMaxPool2d(1)
            elif simnorm_positional_config.pool_type == 'avg':
                pool = nn.AdaptiveAvgPool2d(1)
            elif simnorm_positional_config.pool_type == 'adaptive':
                pool = CatLayersModule(
                    layers=[
                        nn.AdaptiveMaxPool2d(1),
                        nn.AdaptiveMaxPool2d(1)
                    ],
                    dim=1
                )
            else:
                raise 'Not supported ' + simnorm_positional_config.pool_type

            self.head = nn.Sequential(
                Summer(PositionalEncodingPermute2D(self.downsample_net.out_features)),
                pool,
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features
                ),
                nn.Linear(self.downsample_net.out_features, self.embedding_dim, bias=False),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        else:
            raise 'Not Supported ' + head_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.downsample_net(x)

        return self.head(x)
