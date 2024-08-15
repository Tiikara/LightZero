from typing import Optional

import torch
from ding.torch_utils import ResBlock
from ding.utils import SequenceType
from torch import nn


class DownSampleFlat(nn.Module):

    def __init__(self, observation_shape: SequenceType,
                 start_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN',
                 channels_scale: float = 2.,
                 num_blocks: int = 1
                 ) -> None:
        """
        Overview:
            Define downSample convolution network. Encode the observation into hidden state.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"
        assert observation_shape[1] == observation_shape[2]
        assert channels_scale == 1.

        self.observation_shape = observation_shape

        current_channels = start_channels

        in_channels = observation_shape[0]

        downsamples = [
            nn.Conv2d(in_channels, current_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(current_channels),
            activation
        ]

        for _ in range(1, num_blocks):
            new_channels = int(current_channels * channels_scale)

            downsamples.append(
                ResBlock(
                    in_channels=current_channels,
                    out_channels=new_channels,
                    activation=activation,
                    norm_type=norm_type
                )
            )

            current_channels = new_channels

        self.body = nn.Sequential(*downsamples)

        self.out_features = current_channels
        self.out_size = observation_shape[2] // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        return self.body(x)
