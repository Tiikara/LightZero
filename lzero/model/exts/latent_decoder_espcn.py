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


class LatentDecoderESPCN(nn.Module):

    def __init__(self, embedding_dim: int, output_shape: SequenceType, num_channels: int = 64, activation: nn.Module = nn.GELU(approximate='tanh')):
        """
        Overview:
            https://arxiv.org/pdf/1609.05158

            Decoder network based on Sub-Pixe Network used in UniZero. Decode the latent state into 2D image obs.
        Arguments:
            - embedding_dim (:obj:`int`): The dimension of the latent state.
            - output_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.GELU(approximate='tanh').
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_shape = output_shape  # (C, H, W)
        self.num_channels = num_channels
        self.activation = activation

        # Assuming that the output shape is (C, H, W) = (12, 96, 96) and embedding_dim is 256
        # We will reverse the process of the representation network
        self.initial_size = (
            num_channels, output_shape[1] // 8, output_shape[2] // 8)  # This should match the last layer of the encoder
        initial_channels = np.prod(self.initial_size)
        self.fc = nn.Linear(self.embedding_dim, initial_channels)

        self.hidden_channels = 64

        # Upsampling blocks
        self.conv_blocks = nn.ModuleList([
            # 8x8
            MBConv(MBConvConfig(6, 3, 1, num_channels, self.hidden_channels * 4, 1, True, False, act=lambda: activation)),
            nn.PixelShuffle(2), # -> 16x16
            MBConv(MBConvConfig(4, 3, 1, self.hidden_channels, self.hidden_channels * 4, 1, True, False, act=lambda: activation)),
            nn.PixelShuffle(2), # -> 32x32
            MBConv(MBConvConfig(4, 3, 1, self.hidden_channels, self.hidden_channels * 4, 1, False, True, act=lambda: activation)),
            nn.PixelShuffle(2), # -> 64x64
            MBConv(MBConvConfig(1, 3, 1, self.hidden_channels, output_shape[0], 1, False, True, act=lambda: activation)),
        ])
        # TODO: last layer use sigmoid?

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Map embeddings back to the image space
        x = self.fc(embeddings)  # (B, embedding_dim) -> (B, C*H/8*W/8)
        x = x.view(-1, *self.initial_size)  # (B, C*H/8*W/8) -> (B, C, H/8, W/8)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)  # Upsample progressively

        # The output x should have the shape of (B, output_shape[0], output_shape[1], output_shape[2])
        return x
