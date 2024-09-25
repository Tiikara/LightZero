import math

import torch
from ding.torch_utils import ResBlock
from ding.utils import SequenceType
from ditk import logging
from torch import nn

from lzero.model.common import DownSample
from .convnext_block import ConvNeXtLayerNorm, ConvNeXtBlock


class BaseConvNeXtDownSampleBody(nn.Module):

    def __init__(self, observation_shape: SequenceType, out_channels: int,
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

        self.observation_shape = observation_shape

        self.stem = nn.Sequential(
            nn.Conv2d(observation_shape[0], out_channels // 2, kernel_size=3, stride=2, padding=1),
            ConvNeXtLayerNorm(out_channels // 2, eps=1e-6, data_format="channels_first")
        )

        self.resblocks1 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    in_channels=out_channels // 2
                ) for _ in range(1)
            ]
        )

        self.downsample_block = nn.Sequential(
            ConvNeXtLayerNorm(out_channels // 2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
        )

        self.resblocks2 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    in_channels=out_channels
                ) for _ in range(1)
            ]
        )

        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ConvNeXtLayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
        )

        self.resblocks3 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    in_channels=out_channels
                ) for _ in range(1)
            ]
        )

        self.last_norm = ConvNeXtLayerNorm(out_channels, eps=1e-6, data_format="channels_first")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.stem(x)

        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)

        return x

class BaseConvNeXtDownSample(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            num_res_blocks: int = 1,
            num_channels: int = 64,
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

        assert observation_shape[1] == observation_shape[2]

        self.observation_shape = observation_shape

        self.downsample_net = BaseConvNeXtDownSampleBody(
            observation_shape,
            num_channels,
        )

        self.resblocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    in_channels=num_channels
                ) for _ in range(num_res_blocks)
            ]
        )

        self.last_norm = ConvNeXtLayerNorm(num_channels, eps=1e-6, data_format="channels_first")

        self.out_features = num_channels
        self.out_size = observation_shape[1] // 8

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

        x = self.last_norm(x)

        return x
