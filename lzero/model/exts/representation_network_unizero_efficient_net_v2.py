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

from dynamic_efficientnet_v2_downsample import DynamicEfficientNetV2DownSample

import torch
from torch import nn

def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

class RepresentationNetworkUniZeroEfficientNetV2(nn.Module):

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
        assert observation_shape[1] == observation_shape[2], "width and height must be equals"
        logging.info(f"Using norm type: {norm_type}")
        logging.info(f"Using activation type: {activation}")

        self.observation_shape = observation_shape
        self.activation = activation
        self.embedding_dim = embedding_dim

        self.downsample_net = DynamicEfficientNetV2DownSample(
            power_width=0.5,
            power_deep=0.1,
            input_size=observation_shape[1],
            out_features=embedding_dim,
            act_layer=lambda: activation
        )

        efficientnet_v2_init(self.downsample_net)

        self.last_linear = nn.Linear(
            self.downsample_net.final_stage_channels * self.downsample_net.final_stage_size * self.downsample_net.final_stage_size,
            self.embedding_dim,
            bias=False
        )

        self.sim_norm = SimNorm(simnorm_dim=group_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.downsample_net(x)

        x = self.last_linear(x.reshape(-1, self.downsample_net.final_stage_channels * self.downsample_net.final_stage_size * self.downsample_net.final_stage_size))

        # NOTE: very important for training stability.
        x = self.sim_norm(x)

        return x
