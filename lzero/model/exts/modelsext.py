#################
# https://github.com/hankyul2/EfficientNetV2-pytorch/tree/main
#################

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

import torch
from torch import nn

class DynamicEfficientNetV2DownSample(nn.Module):
    def __init__(self, power_deep=1.0, power_width=1.0, input_size=224, in_channels=3, out_features=1000, dropout=0., stochastic_depth=0.,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.power_deep = power_deep
        self.power_width = power_width
        self.input_size = input_size
        self.out_features = out_features
        self.dropout = dropout

        # Base config EfficientNetV2-S
        self.base_config = [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 1, 64, 128, 6, True, False), # (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            # (6, 3, 2, 160, 256, 15, True, False),
        ]

        self.adapted_config = self.adapt_config()

        self.layer_infos = [MBConvConfig(*layer_config, act=act_layer) for layer_config in self.adapted_config]
        self.norm_layer = norm_layer
        self.act = act_layer

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in self.layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(in_channels, self.layer_infos[0].in_ch, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(*self.make_stages(self.layer_infos, block))

    def adapt_config(self):
        adapted_config = []
        current_size = self.input_size // 2

        for i, (e, k, s, in_ch, out_ch, layers, se, fused) in enumerate(self.base_config):
            scaled_layers = max(1, int(layers * self.power_deep))

            scaled_in_ch = max(16, int(in_ch * self.power_width))
            scaled_out_ch = max(16, int(out_ch * self.power_width))

            # Adapt stride
            if current_size > 7 and s == 2:
                current_size //= 2
            else:
                s = 1

            adapted_config.append([e, k, s, scaled_in_ch, scaled_out_ch, scaled_layers, se, fused])

        while current_size > 8:
            last_config = adapted_config[-1]
            new_out_ch = int(last_config[4] * 1.2)
            adapted_config.append((last_config[0], last_config[1], 2, last_config[4], new_out_ch,
                                   max(1, int(last_config[5] * 0.8)), last_config[6], last_config[7]))
            current_size //= 2

        self.final_stage_channels = adapted_config[-1][4]
        self.final_stage_size = current_size

        return adapted_config

    def make_stages(self, layer_infos, block):
        layers = []
        for layer_info in layer_infos:
            for i in range(layer_info.num_layers):
                layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
                layer_info.in_ch = layer_info.out_ch
                layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x):
        return self.blocks(self.stem(x))

def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

class RepresentationNetworkUniZeroOptimized(nn.Module):

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
