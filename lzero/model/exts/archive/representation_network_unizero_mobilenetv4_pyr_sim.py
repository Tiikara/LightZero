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

import math
from functools import partial
from collections import OrderedDict

from .torch_encodings import PositionalEncodingPermute2D

import torch
from torch import nn
import timm

class RepresentationNetworkUniZeroMobilenetV4PirSym(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            num_res_blocks: int = 1,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
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
        self.activation = activation
        self.embedding_dim = embedding_dim

        ##
        # mobilenetv4_conv_small
        ##
        # in 224x224
        # torch.Size([1, 32, 112, 112])
        # torch.Size([1, 32, 56, 56])
        # torch.Size([1, 64, 28, 28])
        # torch.Size([1, 96, 14, 14])
        # torch.Size([1, 960, 7, 7])
        self.downsample_net = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=True,
            features_only=True,
            out_indices=[0, 1, 2, 3], # 3 - [1, 96, width // 16, height // 16]
            drop_rate=0.1,
            drop_path_rate=0.1,
            act_layer=lambda **kwargs: activation
        )

        layers_to_bottleneck = [
            (32, 1), # 32 * 32 * 1 = 1024
            (32, 2), # 16 * 16 * 2 = 512
            (64, 16), # 8 * 8 * 16 = 1024
            (96, 96) # 4 * 4 * 96 = 1536
        ]


        self.bottlenecks = nn.ModuleList([])
        self.bottlenecks_sizes = []

        current_size = observation_shape[1]

        self.full_features = 0

        for layer_to_bottleneck in layers_to_bottleneck:
            current_size = current_size // 2

            self.bottlenecks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=layer_to_bottleneck[0],
                        out_channels=layer_to_bottleneck[1],
                        kernel_size=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(layer_to_bottleneck[1]),
                    activation
                )
            )

            self.bottlenecks_sizes.append(
                (current_size, layer_to_bottleneck[1])
            )

            self.full_features += layer_to_bottleneck[1] * current_size * current_size

        self.head = nn.Sequential(
            nn.Linear(
                self.full_features,
                self.embedding_dim,
                bias=False
            ),
            SimNorm(simnorm_dim=group_size)
        )

        self.out_create_layers = [
            lambda: SimNorm(simnorm_dim=group_size)
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

        features = []

        for i, feature in enumerate(x):
            size = self.bottlenecks_sizes[i]

            features.append(
                self.bottlenecks[i](feature).reshape(-1, size[0] * size[0] * size[1]) # WxHxC
            )

        x = torch.cat(features, dim=1)

        x = self.head(x)

        return x
