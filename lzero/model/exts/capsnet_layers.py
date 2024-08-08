import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##
# https://github.com/akhdanfadh/efficient-capsnet-pytorch
#
# https://arxiv.org/pdf/2101.12491
#
# EFFICIENT-CAPSNET: CAPSULE NETWORK WITH SELF-ATTENTION ROUTING
# Vittorio Mazzia, Francesco Salvetti
#
# Modifications:
# (1) Added Bias property
##

class Squash(nn.Module):
    def __init__(self, eps=1e-8):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        coef = 1 - 1 / (torch.exp(norm) + self.eps)
        unit = x / (norm + self.eps)
        return coef * unit

class PrimaryCaps(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        capsule_size,
        stride=1,
        bias=True
    ):
        super(PrimaryCaps, self).__init__()
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
        self.squash = Squash()

    def forward(self, x):
        x = self.dw_conv2d(x)
        x = x.view(-1, self.num_capsules, self.dim_capsules)  # reshape
        return self.squash(x)


class RoutingCaps(nn.Module):
    def __init__(self, in_capsules, out_capsules, bias = True):
        super(RoutingCaps, self).__init__()
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
        s = torch.sum(u * c, dim=-2)  # (B, N1, D1)
        return self.squash(s)


class CapsLen(nn.Module):
    def __init__(self, eps=1e-7):
        super(CapsLen, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.sqrt(
            torch.sum(x**2, dim=-1) + self.eps
        )  # (batch_size, num_capsules)


class CapsMask(nn.Module):
    def __init__(self):
        super(CapsMask, self).__init__()

    def forward(self, x, y_true=None):
        if y_true is not None:  # training mode
            mask = y_true
        else:  # testing mode
            # convert list of maximum value's indices to one-hot tensor
            temp = torch.sqrt(torch.sum(x**2, dim=-1))
            mask = F.one_hot(torch.argmax(temp, dim=1), num_classes=temp.shape[1])

        masked = x * mask.unsqueeze(-1)
        return masked.view(x.shape[0], -1)  # reshape


def caps_loss(predicted_capsules, true_capsules, alpha=0.5, eps=1e-6):
    pred_norm = F.normalize(predicted_capsules, p=2, dim=-1)
    true_norm = F.normalize(true_capsules, p=2, dim=-1)

    pred_lengths = torch.norm(predicted_capsules, dim=-1)
    true_lengths = torch.norm(true_capsules, dim=-1)
    length_error = F.mse_loss(pred_lengths, true_lengths, reduction='none')

    cos_sim = torch.sum(pred_norm * true_norm, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)
    direction_error = torch.acos(cos_sim) / math.pi

    return alpha * length_error + (1 - alpha) * direction_error
