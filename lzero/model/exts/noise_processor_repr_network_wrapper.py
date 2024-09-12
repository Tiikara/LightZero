import torch
from torch import nn


def apply_gaussian_noise(tensor, std_devs):
    """
    :param tensor: [batch_size, C, H, W]
    :param std_devs: [batch_size]
    :return: noised tensor
    """
    assert tensor.size(0) == std_devs.size(0)

    noise = torch.randn_like(tensor, device=tensor.device)

    std_devs = std_devs.view(-1, 1, 1, 1)

    return tensor + noise * std_devs


def apply_gaussian_noise_with_norm(tensor, std_devs):
    """
    :param tensor: [batch_size, C, H, W]
    :param std_devs: [batch_size]
    :return: noised tensor
    """
    mean = tensor.mean(dim=[1, 2, 3], keepdim=True)
    std = tensor.std(dim=[1, 2, 3], keepdim=True)

    x_noised = apply_gaussian_noise(tensor, std_devs)

    mean_noise = x_noised.mean(dim=[1, 2, 3], keepdim=True)
    std_noise = x_noised.std(dim=[1, 2, 3], keepdim=True)
    x_normalized = (x_noised - mean_noise) / (std_noise + 0.001)

    return x_normalized * std + mean


class NoiseProcessorReprNetworkWrapper(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            max_noise: float = 0.25,
            noise_proba: float = 0.5
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.out_create_layers = encoder.out_create_layers
        self.projection_model = encoder.projection_model
        self.max_noise = max_noise
        self.noise_proba = noise_proba

    def forward_noised(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, C, H, W]
        """
        assert len(x.shape) == 4

        use_noise_mask = (torch.rand(x.size(0), device=x.device) < self.noise_proba).float()

        noise_strength = use_noise_mask * torch.rand(x.size(0), device=x.device)

        std_devs = noise_strength * self.max_noise

        x_noised = apply_gaussian_noise(x, std_devs)

        x_noised = torch.clamp(x_noised, 0., 1.)

        x_encoded = self.encoder(x_noised)

        x_encoded[:, 0] = noise_strength

        return x_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encoder(x)

        x_encoded[:, 0] = 0.

        return x_encoded
