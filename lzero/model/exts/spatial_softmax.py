import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSoftmax(nn.Module):
    def __init__(self, w, h, temperature=1.0, learnable_temperature=False):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=learnable_temperature)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1., 1., w),
            torch.linspace(-1., 1., h)
        )

        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y',  pos_y)

    def forward(self, feature):
        b, c, h, w = feature.shape

        feature = feature.view(b, c, -1)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)

        expected_x = torch.sum(self.pos_x * softmax_attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=-1, keepdim=True)

        expected_xy = torch.cat([expected_x, expected_y], -1)
        feature_keypoints = expected_xy.view(b, c * 2)

        return feature_keypoints
