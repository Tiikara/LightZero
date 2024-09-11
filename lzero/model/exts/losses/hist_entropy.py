import torch
import matplotlib.pyplot as plt
from torch import nn

def hist_entropy_soft(x, num_bins=50, sigma=0.1):
    x = (x - x.min() + 1e-5) / (x.max() - x.min() + 1e-5)

    x = x.unsqueeze(-1)
    bins = torch.linspace(0, 1, num_bins, device=x.device).unsqueeze(0).unsqueeze(0)

    soft_hist = torch.exp(-((x - bins) ** 2) / (2 * sigma ** 2))
    soft_hist = soft_hist.mean(dim=1)

    soft_hist = soft_hist / soft_hist.sum(dim=-1, keepdim=True)

    entropy = -torch.sum(soft_hist * torch.log(soft_hist + 1e-5), dim=-1)

    return entropy.mean()

