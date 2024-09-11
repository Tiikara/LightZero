import torch
from torch import nn


class VariationalLatentNoiseNetworkWrapper(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            max_noise: float = 0.25,
            noise_proba: float = 0.5
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.out_create_layers = encoder.out_create_layers
        self.max_noise = max_noise
        self.noise_proba = noise_proba

    def reparameterization(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(mean, device=mean.device)
        z = mean + var * epsilon
        return z

    def forward_noised(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, C, H, W]
        """
        assert len(x.shape) == 4

        use_noise_mask = (torch.rand(x.size(0), device=x.device) < self.noise_proba).float()

        noise_strength = use_noise_mask * torch.rand(x.size(0), device=x.device)

        std_devs = noise_strength * self.max_noise

        x_encoded = self.encoder(x)

        var = std_devs.view(-1, 1)

        x_encoded = self.reparameterization(x_encoded, var)

        x_encoded[:, 0] = noise_strength

        return x_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encoder(x)

        x_encoded[:, 0] = 0.

        return x_encoded
