import torch
from torch import nn

def fc_block(in_channels, out_channels, activation=None, norm=None, bias=True):
    """
    https://github.com/opendilab/DI-engine/blob/ae3ddc6867875cb51ba008c79c00badae4f91dc2/ding/torch_utils/network/nn_module.py#L242
    """

    block = [nn.Linear(in_channels, out_channels, bias = bias)]
    if norm is not None:
        block.append(norm)
    if activation is not None:
        block.append(activation)

    return nn.Sequential(*block)

class ResFCBlock(nn.Module):
    """
    https://github.com/opendilab/DI-engine/blob/ae3ddc6867875cb51ba008c79c00badae4f91dc2/ding/torch_utils/network/res_block.py#L9

    Overview:
        Residual Block with 2 fully connected layers.
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self, in_channels: int,
            activation: nn.Module = nn.ReLU(),
            norm_type: str = 'BN',
            bias=True,
            dropout: float = None
    ):
        """
        Overview:
            Init the fully connected layer residual block.
        Arguments:
            - in_channels (:obj:`int`): The number of channels in the input tensor.
            - activation (:obj:`nn.Module`): The optional activation function.
            - norm_type (:obj:`str`): The type of the normalization, default set to 'BN'.
            - dropout (:obj:`float`): The dropout rate, default set to None.
        """
        super().__init__()

        assert norm_type == 'BN'

        self.act = activation
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc1 = fc_block(in_channels, in_channels, activation=self.act, norm=nn.BatchNorm1d(in_channels), bias=bias)
        self.fc2 = fc_block(in_channels, in_channels, activation=None, norm=nn.BatchNorm1d(in_channels), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the output of the redisual block.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x + identity)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
