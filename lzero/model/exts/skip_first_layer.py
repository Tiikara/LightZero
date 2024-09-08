import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipFirstLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer

    def forward(self, x):
        first = x[..., 0].unsqueeze(-1)
        sec = x[..., 1:]

        result = torch.cat([first, self.layer(sec)], dim=-1)

        return result


if __name__ == '__main__':
    f = torch.rand((3, 2, 3))
    print(f)
    print(SkipFirstLayer(nn.Identity())(f))
