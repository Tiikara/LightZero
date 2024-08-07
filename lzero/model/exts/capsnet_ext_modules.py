import torch
from torch import nn

from .capsnet_layers import PrimaryCaps, RoutingCaps

class CatModule(nn.Module):
    def __init__(
            self,
            process_module
    ) -> None:
        super().__init__()
        self.process_module = process_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.process_module(x)], dim=1)

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
                    )
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
                    bias=False
                ),
                RoutingCaps(
                    in_capsules=initial_capsule_size,
                    out_capsules=self.out_capsules_size,
                    bias=False
                )
            )
        )

        self.caps = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.caps(x)
