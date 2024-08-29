import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, encoder, lambda_coeff=0.01):
        super().__init__()
        self.encoder = encoder
        self.lambda_coeff = lambda_coeff

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3)
        ])

    def forward(self, x, obs):
        obs_z = self.transform(obs)
        x_z = self.encoder(obs_z)

        assert len(x_z.shape) == 2

        return self.barlow_twins_loss(x, x_z)

    def barlow_twins_loss(self, p1, p2):
        p1 = (p1 - p1.mean(0)) / (p1.std(0) + 1e-6)
        p2 = (p2 - p2.mean(0)) / (p2.std(0) + 1e-6)

        c = torch.mm(p1.T, p2) / p1.shape[0]

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).mean()

        loss = on_diag + self.lambda_coeff * off_diag

        return loss
