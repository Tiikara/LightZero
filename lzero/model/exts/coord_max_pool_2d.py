import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, with_r=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.with_r = with_r

    def forward(self, x):
        batch_size, channels, orig_height, orig_width = x.shape

        pooled, indices = F.max_pool2d(x, self.kernel_size, self.stride, return_indices=True)

        y_coords = indices // orig_width
        x_coords = indices % orig_width

        # Normalize coordinates to range [-1, 1]
        y_coords = y_coords.float() / (orig_height - 1) * 2 - 1
        x_coords = x_coords.float() / (orig_width - 1) * 2 - 1

        result = torch.cat([pooled, y_coords, x_coords], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(y_coords, 2) + torch.pow(x_coords, 2))
            result = torch.cat([result, rr], dim=1)

        return result


class CoordMaxPool2dPerChannel(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, with_r=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.with_r = with_r

    def forward(self, x):
        batch_size, channels, orig_height, orig_width = x.shape

        pooled, indices = F.max_pool2d(x, self.kernel_size, self.stride, return_indices=True)

        _, _, pooled_height, pooled_width = pooled.shape

        y_coords = indices // orig_width
        x_coords = indices % orig_width

        # Normalize coordinates to range [-1, 1]
        y_coords = y_coords.float() / (orig_height - 1) * 2 - 1
        x_coords = x_coords.float() / (orig_width - 1) * 2 - 1

        output_channels = 4 if self.with_r else 3
        result = torch.zeros(batch_size, channels, output_channels, pooled_height, pooled_width, device=x.device)

        result[:, :, 0, :, :] = pooled
        result[:, :, 1, :, :] = y_coords
        result[:, :, 2, :, :] = x_coords

        if self.with_r:
            rr = torch.sqrt(torch.pow(y_coords, 2) + torch.pow(x_coords, 2))
            result[:, :, 3, :, :] = rr  # Radial distances

        # Reshape to (batch_size, channels, output_channels, height * width)
        result = result.view(batch_size, channels, output_channels, -1)

        return result
