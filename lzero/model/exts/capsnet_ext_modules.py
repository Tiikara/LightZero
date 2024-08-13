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


class PrimaryCapsForward1D(nn.Module):
    def __init__(
            self,
            capsule_size
    ):
        super().__init__()

        self.num_capsules, self.dim_capsules = capsule_size
        self.squash = Squash()

    def forward(self, x):
        x = x.view(-1, self.num_capsules, self.dim_capsules)  # reshape
        return self.squash(x)


class PrimaryCapsWithoutSquash(nn.Module):
    def __init__(
            self,
            in_channels,
            kernel_size,
            capsule_size,
            stride=1,
            bias=True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_capsules, self.dim_capsules = capsule_size
        self.stride = stride

        self.dw_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_capsules * self.dim_capsules,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(self.num_capsules * self.dim_capsules)

    def forward(self, x):
        x = self.dw_conv2d(x)
        x = self.norm(x)
        return x.view(-1, self.num_capsules, self.dim_capsules)

class RoutingCapsWithoutSquash(nn.Module):
    def __init__(self, in_capsules, out_capsules, bias = True):
        super().__init__()

        self.N0, self.D0 = in_capsules
        self.N1, self.D1 = out_capsules
        self.squash = Squash()

        # initialize routing parameters
        self.W = nn.Parameter(torch.Tensor(self.N1, self.N0, self.D0, self.D1))
        nn.init.kaiming_normal_(self.W)

        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.zeros(self.N1, self.N0, 1))

    def forward(self, x):
        ## prediction vectors
        # ji,kjiz->kjz = k and z broadcast, then ji,ji->j = sum(a*b,axis=1)
        u = torch.einsum("...ji,kjiz->...kjz", x, self.W)  # (batch_size/B, N1, N0, D1)

        ## coupling coefficients
        # ij,kj->i = ij,kj->k = sum(matmul(a,a.T),axis=0) != ij,ij->i
        c = torch.einsum("...ij,...kj->...i", u, u)  # (B, N1, N0)
        c = c[..., None]  # (B, N1, N0, 1) for bias broadcasting
        c = c / torch.sqrt(torch.tensor(self.D1).float())  # stabilize
        c = torch.softmax(c, axis=1)

        if self.bias:
            c += self.b

        ## new capsules
        return torch.sum(u * c, dim=-2)  # (B, N1, D1)

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

class CapsInitialModuleWithoutSquash(nn.Module):

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
                PrimaryCapsWithoutSquash(
                    in_channels=caps_channels,
                    kernel_size=in_size,
                    capsule_size=initial_capsule_size,
                    bias=bias
                ),
                RoutingCapsWithoutSquash(
                    in_capsules=initial_capsule_size,
                    out_capsules=self.out_capsules_size,
                    bias=bias
                )
            )
        )

        self.caps = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.caps(x)


class CapsInitialModuleForward1D(nn.Module):

    def __init__(
            self,
            initial_capsule_size=(32, 8),
            out_capsules_size=(32, 16),
            bias=True
    ) -> None:
        super().__init__()

        self.caps = nn.Sequential(
            PrimaryCapsForward1D(
                capsule_size=initial_capsule_size
            ),
            RoutingCaps(
                in_capsules=initial_capsule_size,
                out_capsules=out_capsules_size,
                bias=bias
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
