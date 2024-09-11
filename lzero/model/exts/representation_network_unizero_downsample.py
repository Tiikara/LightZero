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

from .down_sample_full_pos import DownSampleFullPos
from .efficientnet_v2 import MBConvConfig, MBConv, ConvBNAct
from .capsnet_layers import PrimaryCaps, RoutingCaps, Squash
import math
from functools import partial
from collections import OrderedDict

from lzero.model.common import DownSample
from .capsnet_ext_modules import CapsInitialModule, CapsInitialModuleForward1D, PrimaryCapsForward1D
from .multiply_module import MultiplyModule
from .norms.rms_norm import RMSNorm
from .remove_first_dim_module import RemoveFirstDimModule
from .res_fc_block import ResFCBlock
from .res_feed_forward_block import ResFeedForwardBlock
from .second_dim_check import SecondDimCheck
from .down_sample_res_net import DownSampleResNet
from .reshape_last_dim_1d import ReshapeLastDim1D
from .return_shape_module import ReturnShapeModule
from .reshape_last_dim import ReshapeLastDim
from .base_down_sample import BaseDownSample
from .gumbel_simnorm import GumbelSimNorm
from .cat_layers_module import CatLayersModule
from .adaptive_object_aware_pooling import AdaptiveObjectAwarePooling
from .coordconv import AddCoords
from .coord_max_pool_2d import CoordMaxPool2d, CoordMaxPool2dPerChannel
from .channel_wise_max_pool_2d_with_crossinfo import ChannelWiseMaxPoolWithCrossInfo
from .down_sample_flat import DownSampleFlat

import torch
from torch import nn
import timm
from .caps_sem import CapSEM
from .simple_classification_model import SimpleClassificationModel
from .spatial_softmax import SpatialSoftmax
from .spatial_softmax_positional import SpatialSoftmaxPositional
from .torch_encodings import Summer, PositionalEncodingPermute2D
from .vae_net import VAENet
from .add_dim_to_start_module import AddDimToStartModule

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
            downsample_network_config=None,
            head_type: str = None,
            head_config=None,
            projection_config=None
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
        elif downsample_network_config.type == 'pos':
            self.downsample_net = DownSampleFullPos(
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
        elif downsample_network_config.type == 'flat':
            flat_config = downsample_network_config.flat

            self.downsample_net = DownSampleFlat(
                observation_shape,
                start_channels=flat_config.start_channels,
                activation=activation,
                norm_type=norm_type,
                channels_scale=flat_config.channels_scale,
                num_blocks=flat_config.num_blocks
            )
        else:
            raise "Not supported " + downsample_network_config.type

        self.activation = activation
        self.embedding_dim = embedding_dim

        if head_type == 'caps':
            caps_config = head_config.caps

            assert self.embedding_dim % num_capsules == 0

            out_capsules_dim = self.embedding_dim // num_capsules

            # num_capsules x (embedding_dim / num_capsules) = embedding_dim
            self.out_capsules = (num_capsules, out_capsules_dim)

            if caps_config.use_linear_input_for_caps:
                caps = [
                    ReshapeLastDim1D(
                        out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                    )
                ]

                if caps_config.double_linear_input_for_caps:
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

                if caps_config.use_routing:
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
                assert caps_config.use_routing is True

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

            if caps_config.use_squash_in_transformer:
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
        elif head_type == 'caps_max_coords':
            assert self.embedding_dim % num_capsules == 0

            out_capsules_dim = self.embedding_dim // num_capsules

            # num_capsules x (embedding_dim / num_capsules) = embedding_dim
            self.out_capsules = (num_capsules, out_capsules_dim)

            self.head = nn.Sequential(
                CoordMaxPool2dPerChannel(kernel_size=self.downsample_net.out_size, with_r=True),
                PrimaryCapsForward1D(
                    capsule_size=(self.downsample_net.out_features, 4)
                ),
                RoutingCaps(
                    in_capsules=(self.downsample_net.out_features, 4),
                    out_capsules=self.out_capsules,
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
        elif head_type == 'caps_max_positional':
            assert self.embedding_dim % num_capsules == 0

            out_capsules_dim = self.embedding_dim // num_capsules

            # num_capsules x (embedding_dim / num_capsules) = embedding_dim
            self.out_capsules = (num_capsules, out_capsules_dim)

            self.head = nn.Sequential(
                Summer(PositionalEncodingPermute2D(self.downsample_net.out_features)),
                nn.AdaptiveMaxPool2d(1),
                PrimaryCapsForward1D(
                    capsule_size=(self.downsample_net.out_features, 1)
                ),
                RoutingCaps(
                    in_capsules=(self.downsample_net.out_features, 1),
                    out_capsules=self.out_capsules,
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
        elif head_type == 'cap_sem_max_positional':
            assert self.embedding_dim % num_capsules == 0

            out_capsules_dim = self.embedding_dim // num_capsules

            # num_capsules x (embedding_dim / num_capsules) = embedding_dim
            self.out_capsules = (num_capsules, out_capsules_dim)

            self.head = nn.Sequential(
                Summer(PositionalEncodingPermute2D(self.downsample_net.out_features)),
                nn.AdaptiveMaxPool2d(1),
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features
                ),
                nn.Linear(self.downsample_net.out_features, self.embedding_dim, bias=False),
                ReshapeLastDim(
                    out_shape=self.out_capsules
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

            self.out_create_layers = [
                lambda: SecondDimCheck(
                    ReturnShapeModule(
                        nn.Sequential(
                            ReshapeLastDim(
                                out_shape=self.out_capsules
                            ),
                            CapSEM(
                                num_capsules=self.out_capsules[0],
                                capsule_dim=self.out_capsules[1],
                                group_size=group_size
                            )
                        )
                    )
                )
            ]
        elif head_type == 'gumbel_simnorm':
            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim, bias=False),
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
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim, bias=False),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'simnorm_positional_multiply':
            self.head = nn.Sequential(
                MultiplyModule(
                    PositionalEncodingPermute2D(self.downsample_net.out_features)
                ),
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim,
                    bias=False
                ),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'simnorm_positional':
            simnorm_positional_config = head_config.simnorm_positional

            if simnorm_positional_config.pool_type == 'max':
                pool = nn.AdaptiveMaxPool2d(1)
                pool_features = self.downsample_net.out_features
            elif simnorm_positional_config.pool_type == 'avg':
                pool = nn.AdaptiveAvgPool2d(1)
                pool_features = self.downsample_net.out_features
            elif simnorm_positional_config.pool_type == 'mixed_max_avg':
                pool = CatLayersModule(
                    layers=[
                        nn.AdaptiveMaxPool2d(1),
                        nn.AdaptiveAvgPool2d(1)
                    ],
                    dim=1
                )
                pool_features = self.downsample_net.out_features * 2
            else:
                raise 'Not supported ' + simnorm_positional_config.pool_type

            self.head = nn.Sequential(
                Summer(PositionalEncodingPermute2D(self.downsample_net.out_features)),
                pool,
                ReshapeLastDim1D(
                    out_features=pool_features
                ),
                nn.Linear(pool_features, self.embedding_dim, bias=False),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'simnorm_coord_max_pool':
            self.head = nn.Sequential(
                CoordMaxPool2d(kernel_size=self.downsample_net.out_size, with_r=True),
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * 4
                ),
                nn.Linear(self.downsample_net.out_features * 4, self.embedding_dim, bias=False),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'simnorm_positional_object_aware':
            self.head = nn.Sequential(
                nn.Conv2d(self.downsample_net.out_features, self.embedding_dim, kernel_size=1),
                nn.BatchNorm2d(self.embedding_dim),
                activation,
                AdaptiveObjectAwarePooling(
                    in_channels=self.embedding_dim,
                    pre_layer_features=nn.Sequential(
                        Summer(PositionalEncodingPermute2D(self.embedding_dim))
                    )
                ),
                ReshapeLastDim1D(
                    out_features=self.embedding_dim
                ),
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'simnorm_coords_object_aware':
            self.head = nn.Sequential(
                nn.Conv2d(self.downsample_net.out_features, self.embedding_dim, kernel_size=1),
                nn.BatchNorm2d(self.embedding_dim),
                activation,
                AdaptiveObjectAwarePooling(
                    in_channels=self.embedding_dim,
                    attention_channels=self.embedding_dim + 2,
                    pre_layer_features=AddCoords(rank=2)
                ),
                ReshapeLastDim1D(
                    out_features=self.embedding_dim + 2
                ),
                nn.Linear(self.embedding_dim + 2, self.embedding_dim, bias=False),
                SimNorm(simnorm_dim=group_size)
            )

            self.out_create_layers = [
                lambda: SimNorm(simnorm_dim=group_size)
            ]
        elif head_type == 'linear_classification_2fc':
            self.projection_model = nn.Sequential(
                ResFCBlock(
                    in_channels = self.embedding_dim,
                    activation = activation,
                    norm_type = norm_type,
                    bias = False
                )
            )

            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim,
                    bias=False
                ),
            )

            self.out_create_layers = []
        elif head_type == 'linear':
            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim,
                    bias=False
                ),
            )

            self.out_create_layers = []
        elif head_type == 'linear_norm':
            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim,
                    bias=False
                ),
                nn.LayerNorm(self.embedding_dim)
            )

            self.out_create_layers = [
                lambda: nn.LayerNorm(self.embedding_dim)
            ]
        elif head_type == 'linear_norm_except_one':
            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim - 1,
                    bias=False
                ),
                nn.LayerNorm(self.embedding_dim - 1),
                AddDimToStartModule(0)
            )

            self.out_create_layers = [
                lambda: SecondDimCheck(
                    nn.Sequential(
                        RemoveFirstDimModule(),
                        nn.LayerNorm(self.embedding_dim - 1),
                        AddDimToStartModule(0)
                    )
                )
            ]
        elif head_type == 'linear_norm_gelu_except_one':
            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim - 1,
                    bias=False
                ),
                nn.LayerNorm(self.embedding_dim - 1),
                nn.GELU(approximate='tanh'),
                AddDimToStartModule(0)
            )

            self.out_create_layers = [
                lambda: SecondDimCheck(
                    nn.Sequential(
                        RemoveFirstDimModule(),
                        nn.LayerNorm(self.embedding_dim - 1),
                        nn.GELU(approximate='tanh'),
                        AddDimToStartModule(0)
                    )
                )
            ]
        elif head_type == 'linear_rms_norm_except_one':
            self.head = nn.Sequential(
                ReshapeLastDim1D(
                    out_features=self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size
                ),
                nn.Linear(
                    self.downsample_net.out_features * self.downsample_net.out_size * self.downsample_net.out_size,
                    self.embedding_dim - 1,
                    bias=False
                ),
                RMSNorm(self.embedding_dim - 1),
                AddDimToStartModule(0)
            )

            self.out_create_layers = [
                lambda: SecondDimCheck(
                    nn.Sequential(
                        RemoveFirstDimModule(),
                        RMSNorm(self.embedding_dim - 1),
                        AddDimToStartModule(0)
                    )
                )
            ]
        else:
            raise 'Not Supported ' + head_type

        if projection_config.type is not None:
            projection_model_layers = []

            if projection_config.type == '2fc':
                for _ in range(projection_config.num_layers):
                    projection_model_layers.append(
                        ResFCBlock(
                            in_channels = self.embedding_dim,
                            activation = activation,
                            norm_type = norm_type,
                            bias = False
                        )
                    )
            elif projection_config.type == 'res_feed_forward':
                for _ in range(projection_config.num_layers):
                    projection_model_layers.append(
                        ResFeedForwardBlock(
                            in_channels = self.embedding_dim,
                            hidden_channels = self.embedding_dim,
                            activation = activation,
                            bias = False
                        )
                    )

                projection_model_layers.append(
                    nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                )
            elif projection_config.type == 'linear':
                projection_model_layers.append(
                    nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                )

            if projection_config.last_norm == 'BN':
                projection_model_layers.append(
                    nn.BatchNorm1d(self.embedding_dim)
                )

            self.projection_model = nn.Sequential(
                *projection_model_layers
            )

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
