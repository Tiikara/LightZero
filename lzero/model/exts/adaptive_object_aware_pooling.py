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

import torch
from torch import nn
import timm
from .caps_sem import CapSEM


class AdaptiveObjectAwarePooling(nn.Module):
    def __init__(self, in_channels, pre_layer_features, epsilon=1e-6):
        super().__init__()

        self.pre_layer_features = pre_layer_features

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.eps = epsilon

    def forward(self, x):
        features = self.pre_layer_features(x)
        attention = self.attention(x)

        weighted_features = features * attention

        return (weighted_features.sum(dim=[2, 3]) / (attention.sum(dim=[2, 3]) + self.eps)).unsqueeze(-1).unsqueeze(-1)

