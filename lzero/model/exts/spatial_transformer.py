import torch.nn as nn

from .torch_encodings import PositionalEncodingPermute2D


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.conv = nn.Conv2d(in_channels, d_model // 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(d_model // 2)
        self.pos_encoder = PositionalEncodingPermute2D(d_model // 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, activation="gelu", dropout=0.1, bias=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        b, c, h, w = x.size()

        x = self.conv(x)

        x = self.bn(x)

        x = self.pos_encoder(x)

        x = x.view(b, self.d_model, -1).permute(2, 0, 1)

        memory = self.transformer_encoder(x)

        return memory.permute(1, 2, 0).view(b, self.d_model, h, w)
