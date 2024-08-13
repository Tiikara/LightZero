import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()

        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)  # BN after 1x1 conv

        self.conv = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(1)  # BN after final conv

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1x1 convolution with BN
        conv1x1_out = self.bn1(self.conv1x1(x))

        # Channel-wise mean and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate all features
        x = torch.cat([conv1x1_out, avg_out, max_out], dim=1)

        # Apply convolution, BN, and sigmoid
        x = self.bn2(self.conv(x))
        return self.sigmoid(x)

