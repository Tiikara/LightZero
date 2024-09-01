import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """
    https://arxiv.org/pdf/2105.04906

    inv_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,

    inv_coeff: float = 1.0,
    var_coeff: float = 1.0,
    cov_coeff: float = 0.04,
    """

    def __init__(self, inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
        super().__init__()

        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        # invariance loss
        sim_loss = F.mse_loss(x, y, reduction='none').mean()

        # variance loss
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        # covariance loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum() / x.shape[1] + off_diagonal(cov_y).pow_(2).sum() / y.shape[1]

        loss = self.inv_coeff * sim_loss + self.var_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

class VICRegSingleLoss(nn.Module):
    """
    https://arxiv.org/pdf/2105.04906

    inv_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,

    inv_coeff: float = 1.0,
    var_coeff: float = 1.0,
    cov_coeff: float = 0.04,
    """

    def __init__(self, var_coeff=25.0, cov_coeff=1.0):
        super().__init__()

        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x):
        # variance loss
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x))

        # covariance loss
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum() / x.shape[1]

        loss = self.var_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
