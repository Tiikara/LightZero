import torch
import torch.nn as nn
import torch.nn.functional as F

from lzero.model.exts.torch_encodings import PositionalEncodingPermute2D


class SpatialSoftmaxPositional(nn.Module):
    def __init__(self, pos_dim, temperature=1.0, learnable_temperature=False):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=learnable_temperature)

        self.positional = PositionalEncodingPermute2D(pos_dim)
        self.pos_dim = pos_dim

    def forward(self, x):
        b, c, h, w = x.shape

        feature = x.view(b, c, -1)  # [b, c, h*w]
        weight = F.softmax(feature / self.temperature, dim=-1)  # [b, c, h*w]

        # Compute raw activation for object presence
        presence_activation = torch.max(feature, dim=-1)[0]  # [b, c]

        # Reshape positional encoding
        pos_emb = self.positional(x)  # [b, pos_dim, h, w]
        pos_emb = pos_emb.view(b, self.pos_dim, -1)  # [b, pos_dim, h*w]

        # Apply positional encoding to weights
        weighted_pos = weight.unsqueeze(2) * pos_emb.unsqueeze(1)  # [b, c, pos_dim, h*w]

        # Sum over spatial dimensions
        feature_keypoints = weighted_pos.sum(dim=-1)  # [b, c, pos_dim]

        output = torch.cat([feature_keypoints, presence_activation.unsqueeze(-1)], dim=-1)  # [b, c, pos_dim + 1]

        return output.view(b, c * (self.pos_dim + 1))


if __name__ == "__main__":
    model = SpatialSoftmaxPositional(pos_dim=32, temperature=1.0, learnable_temperature=True)
    input_tensor = torch.randn(2, 64, 20, 20)  # Example input: batch_size=2, channels=64, height=20, width=20
    res = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {res.shape}")
    print(f"Current temperature: {model.temperature.item()}")
