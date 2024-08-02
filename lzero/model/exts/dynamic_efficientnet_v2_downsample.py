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
    #################
    # https://github.com/hankyul2/EfficientNetV2-pytorch/tree/main
    #################

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
