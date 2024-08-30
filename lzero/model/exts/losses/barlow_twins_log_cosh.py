import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from lzero.model.exts.losses import log_cosh_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLogCosh(nn.Module):
    def __init__(self, lambda_coeff=0.01):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def barlow_twins_loss(self, p1, p2, use_bn=True):
        if use_bn:
            p1 = F.batch_norm(p1, running_mean=None, running_var=None, training=True)
            p2 = F.batch_norm(p2, running_mean=None, running_var=None, training=True)

        c = torch.mm(p1.T, p2) / p1.shape[0]

        on_diag = log_cosh_loss(torch.diagonal(c), 1).mean()
        off_diag = log_cosh_loss(off_diagonal(c), 0).mean()

        loss = on_diag + self.lambda_coeff * off_diag

        return loss
