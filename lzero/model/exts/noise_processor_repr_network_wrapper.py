import torch
from torch import nn


def apply_gaussian_noise(tensor, std_devs, min_val=0., max_val=1.):
    """
    :param tensor: [batch_size, C, H, W]
    :param std_devs: [batch_size]
    :return: noised tensor
    """
    assert tensor.size(0) == std_devs.size(0)

    noise = torch.randn_like(tensor, device=tensor.device)

    std_devs = std_devs.view(-1, 1, 1, 1)

    noisy_tensor = tensor + noise * std_devs

    clipped_tensor = torch.clamp(noisy_tensor, min_val, max_val)

    return clipped_tensor

class NoiseProcessorReprNetworkWrapper(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            max_noise: float = 0.25
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.out_create_layers = encoder.out_create_layers
        self.max_noise = max_noise

    def forward_noised(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: [batch_size, C, H, W]
        """
        assert len(x.shape) == 4

        std_devs = torch.rand(x.size(0), device=x.device) * self.max_noise

        x_noised = apply_gaussian_noise(x, std_devs)

        x_encoded = self.encoder(x_noised)

        x_encoded[:, 0] = std_devs

        return x_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encoder(x)

        x_encoded[:, 0] = 0.

        return x_encoded
