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

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class PANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.activation(x + residual)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, activation, norm_type):
        super().__init__()
        self.conv = ResBlock(
            in_channels=in_channels, activation=activation, norm_type=norm_type, res_type='bottleneck', bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class RepresentationNetworkUniZeroMobilenetV4PAN(nn.Module):
    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            num_res_blocks: int = 1,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.activation = activation
        self.embedding_dim = embedding_dim

        self.downsample_net = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=False,
            features_only=True,
            out_indices=[0, 1, 2, 3],
            drop_rate=0.1,
            drop_path_rate=0.1,
            act_layer=lambda **kwargs: activation
        )

        mobilenet_channels = [32, 32, 64, 96]
        pan_channels = 64

        self.lateral_convs = nn.ModuleList([
           nn.Sequential(
               nn.Conv2d(c, pan_channels, kernel_size=1),
               nn.BatchNorm2d(pan_channels)
           ) for c in mobilenet_channels
        ])

        self.upsample_blocks = nn.ModuleList([
            PixelShuffleUpsample(pan_channels, pan_channels, 2) for _ in range(3)
        ])

        self.pan_blocks = nn.ModuleList([
            PANBlock(
                in_channels=pan_channels, out_channels=pan_channels, activation=activation
            ) for _ in mobilenet_channels
        ])

        self.spatial_attention = SpatialAttention(pan_channels)

        self.head = nn.Sequential(
            nn.Linear(
                pan_channels * 4,
                self.embedding_dim,
                bias=False
            ),
            SimNorm(simnorm_dim=group_size)
        )

        self.out_create_layers = [
            lambda: SimNorm(simnorm_dim=group_size)
        ]

    def forward(self, x):
        features = self.downsample_net(x)

        pan_features = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, features)]

        for i in range(len(pan_features) - 1, 0, -1):
            pan_features[i - 1] += self.upsample_blocks[i - 1](pan_features[i])
            pan_features[i - 1] = self.pan_blocks[i - 1](pan_features[i - 1])

        x = self.pan_blocks[-1](pan_features[0])

        x = self.spatial_attention(x)

        x = self.head(x)

        return x
