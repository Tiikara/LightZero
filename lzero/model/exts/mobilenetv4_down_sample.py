import math

import torch
from ding.torch_utils import ResBlock
from ding.utils import SequenceType
from ditk import logging
from torch import nn
import timm

from lzero.model.common import DownSample
from .convnext_block import ConvNeXtLayerNorm, ConvNeXtBlock


class MobileNetV4DownSample(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            activation = nn.GELU()
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

        self.mobilenet = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=True,
            num_classes=0,
        )

        self.dwconv = nn.Sequential(
            nn.Conv2d(960, 960, kernel_size=2, groups=960, bias=False),
            nn.BatchNorm2d(960),
            activation
        )

        self.out_features = 960
        self.out_size = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """

        x = self.mobilenet.forward_features(x)

        x = self.dwconv(x)

        return x
