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
from .common import SimNorm
from .common import NormByType

class DownSampleOptimized(nn.Module):

    def __init__(self, observation_shape: SequenceType,
                 start_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN'
                 ) -> None:
        """
        Overview:
            Define downSample convolution network. Encode the observation into hidden state.
            This network is often used in video games like Atari. In board games like go and chess,
            we don't need this module.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[12, 96, 96]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - out_channels (:obj:`int`): The output channels of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`Optional[str]`): The normalization type used in network, defaults to 'BN'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"
        assert observation_shape[-2] == observation_shape[-1], "width and height must be equal"

        self.observation_shape = observation_shape
        self.activation = activation

        current_channels = start_channels
        current_size = observation_shape[-1]

        self.conv_initial = nn.Conv2d(
            observation_shape[0],
            current_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,  # disable bias for better convergence
        )

        current_size = current_size // 2

        self.norm_initial = NormByType(
            norm_type=norm_type,
            channels=current_channels,
            size0=current_size,
            size1=current_size
        )

        self.blocks = nn.ModuleList([])

        while current_size > 8:
            self.blocks.append(
                ResBlock(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='basic',
                    bias=False
                )
            )

            out_channels = current_channels * 2

            self.blocks.append(
                ResBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='downsample',
                    bias=False
                )
            )

            current_size = current_size // 2
            current_channels = out_channels

        self.out_channels = current_channels * current_size
        self.out_size = current_size

        self.out_conv = nn.Conv2d(
            current_channels,
            self.out_channels,
            kernel_size=current_size,
            stride=current_size - 1,
            padding=0,
            bias=False
        )

        self.out_norm = NormByType(
            norm_type=norm_type,
            channels=self.out_channels,
            size0=1,
            size1=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """

        x = self.conv_initial(x)
        x = self.norm_initial(x)
        x = self.activation(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)
        x = self.out_norm(x)
        x = self.activation(x)

        return x

class RepresentationNetworkUniZeroOptimized(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            num_res_blocks: int = 1,
            start_channels: int = 16,
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

        self.downsample_net = DownSampleOptimized(
            observation_shape,
            activation=activation,
            norm_type=norm_type,
            start_channels=start_channels
        )

        self.resblocks = nn.ModuleList(
            [
                nn.Conv2d(
                    self.downsample_net.out_channels,
                    self.downsample_net.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                NormByType(
                    norm_type=norm_type,
                    channels=self.downsample_net.out_channels,
                    size0=1,
                    size1=1
                ),
                activation
            ]
        )

        self.last_linear = nn.Linear(
            self.downsample_net.out_channels,
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

        for block in self.resblocks:
            x = block(x)

        # NOTE: very important.
        # flatten_size = out_channels -> 768
        x = self.last_linear(x.reshape(-1, self.downsample_net.out_channels * self.downsample_net.out_size * self.downsample_net.out_size))
        x = x.view(-1, self.embedding_dim)

        # NOTE: very important for training stability.
        x = self.sim_norm(x)

        return x
