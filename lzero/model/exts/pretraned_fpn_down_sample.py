import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, activation):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(ConvBnReLU(in_channels, out_channels, activation=activation, kernel_size=1))
            self.fpn_convs.append(ConvBnReLU(out_channels, out_channels, activation=activation, kernel_size=3, padding=1))

    def forward(self, x):
        laterals = [conv(f) for f, conv in zip(x, self.lateral_convs)]

        for i in range(len(x) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')

        outputs = [conv(lat) for lat, conv in zip(laterals, self.fpn_convs)]
        return outputs

class PretrainedFPNDownSample(nn.Module):
    def __init__(self, latent_dim, activation):
        super().__init__()

        self.model = timm.create_model(
            'convnextv2_atto.fcmae',
            pretrained=True,
            features_only=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False

        assert latent_dim % 4 == 0

        # FPN
        in_channels_list = [40, 80, 160, 320]
        self.fpn = FeaturePyramidNetwork(in_channels_list, activation=activation, out_channels=latent_dim // 4)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.out_features = latent_dim
        self.out_size = 1

    def forward(self, x):
        features = self.model(x)

        # FPN
        fpn_features = self.fpn([features[0], features[1], features[2], features[3]])

        pooled = [self.global_pool(f) for f in fpn_features]
        concatenated = torch.cat(pooled, dim=1)

        return concatenated
