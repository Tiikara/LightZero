from typing import Union

import torch
from ding.torch_utils.network.nn_module import conv2d_block
from torch import nn


class ResBlockChannelled(nn.Module):
    def __init__(
            self,
            in_channels: int,
            activation: nn.Module = nn.ReLU(),
            norm_type: str = 'BN',
            bias: bool = True,
            out_channels: Union[int, None] = None,
    ) -> None:
        """
        Overview:
            Init the 2D convolution residual block.
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor.
            - activation (:obj:`nn.Module`): The optional activation function.
            - norm_type (:obj:`str`): Type of the normalization, default set to 'BN'(Batch Normalization), \
                supports ['BN', 'LN', 'IN', 'GN', 'SyncBN', None].
            - res_type (:obj:`str`): Type of residual block, supports ['basic', 'bottleneck', 'downsample']
            - bias (:obj:`bool`): Whether to add a learnable bias to the conv2d_block. default set to True.
            - out_channels (:obj:`int`): Number of channels in the output tensor, default set to None, \
                which means out_channels = in_channels.
        """
        super().__init__()

        self.act = activation
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = conv2d_block(
            in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
        )
        self.conv2 = conv2d_block(
            out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
        )
        self.conv3 = conv2d_block(in_channels, out_channels, 3, 1, 1, activation=None, norm_type=None, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        identity = self.conv3(identity)
        x = self.act(x + identity)
        return x
