import torch
from torch import nn

from .capsnet_layers import PrimaryCaps, RoutingCaps
from .cat_module import CatModule
from .capsnet_layers import Squash

class PrimaryCapsForward(nn.Module):
    def __init__(
            self,
            capsule_size
    ):
        super().__init__()

        self.num_capsules, self.dim_capsules = capsule_size
        self.squash = Squash()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # (batchsize, ch, x, y) -> (batchsize, x, y, ch)
        x = x.view(-1, self.num_capsules, self.dim_capsules)  # reshape
        return self.squash(x)

class CapsInitialModuleForward(nn.Module):

    def __init__(
            self,
            in_channels,
            in_size,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            out_capsules_size=(32, 16),
            bias=True
    ) -> None:
        super().__init__()

        self.activation = activation

        initial_capsule_size = (in_size * in_size, in_channels)
        self.out_capsules_size = out_capsules_size

        self.caps = nn.Sequential(
            nn.Sequential(
                PrimaryCapsForward(
                    capsule_size=initial_capsule_size,
                ),
                RoutingCaps(
                    in_capsules=initial_capsule_size,
                    out_capsules=self.out_capsules_size,
                    bias=bias
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.caps(x)


class CapsInitialModule(nn.Module):

    def __init__(
            self,
            in_channels,
            in_size,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            initial_capsule_size=(32, 8),
            out_capsules_size=(32, 16),
            bias=True
    ) -> None:
        super().__init__()

        self.activation = activation

        caps_channels = initial_capsule_size[0] * initial_capsule_size[1]
        self.out_capsules_size = out_capsules_size

        modules = []

        if in_channels < caps_channels:
            expand_channels = caps_channels-in_channels

            modules.append(
                CatModule(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=expand_channels,
                            kernel_size=1,
                            bias=bias
                        ),
                        nn.BatchNorm2d(expand_channels),
                        self.activation
                    ),
                    dim=1
                )
            )
        elif in_channels > caps_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=caps_channels,
                        kernel_size=1,
                        bias=bias
                    ),
                    nn.BatchNorm2d(caps_channels),
                    self.activation
                )
            )

        modules.append(
            nn.Sequential(
                PrimaryCaps(
                    in_channels=caps_channels,
                    kernel_size=in_size,
                    capsule_size=initial_capsule_size,
                    bias=bias
                ),
                RoutingCaps(
                    in_capsules=initial_capsule_size,
                    out_capsules=self.out_capsules_size,
                    bias=bias
                )
            )
        )

        self.caps = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.caps(x)
