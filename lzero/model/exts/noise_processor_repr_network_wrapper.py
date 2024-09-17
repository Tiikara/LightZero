from dataclasses import dataclass

import torch
from torch import nn

from lzero.model.exts.noise_scheduler import NoiseScheduler


@dataclass
class NoiseRandomDistributionPowerConfig:
    power: float

@dataclass
class NoiseRandomDistributionConfig:
    type: str
    power: NoiseRandomDistributionPowerConfig

@dataclass
class NoiseStrengthRandomConfig:
    noise_proba: float
    random_distribution_config: NoiseRandomDistributionConfig

@dataclass
class NoiseStrengthSampleConfig:
    noise_samples_perc: float
    random_distribution_config: NoiseRandomDistributionConfig

@dataclass
class NoiseStrengthConfig:
    type: str
    random: NoiseStrengthRandomConfig
    sample: NoiseStrengthSampleConfig

@dataclass
class NoiseSchedulerConfig:
    initial_noise: float
    final_noise: float
    schedule_length: int
    decay_type: str

@dataclass
class NoiseConfig:
    use_norm: bool
    noise_strength_config: NoiseStrengthConfig
    noise_scheduler: NoiseSchedulerConfig

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
            config: NoiseConfig,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.out_create_layers = encoder.out_create_layers
        self.projection_model = encoder.projection_model
        self.config = config

        noise_scheduler_config = config.noise_scheduler
        self.noise_scheduler = NoiseScheduler(
            initial_noise = noise_scheduler_config.initial_noise,
            final_noise = noise_scheduler_config.final_noise,
            schedule_length = noise_scheduler_config.schedule_length,
            decay_type = noise_scheduler_config.decay_type
        )

    def get_random_distribution(self, x, config: NoiseRandomDistributionConfig) -> torch.Tensor:
        if config.type == 'linear':
            return torch.rand(x.size(0), device=x.device)
        if config.type == 'max':
            return torch.ones(x.size(0), dtype=torch.float, device=x.device)
        elif config.type == 'power':
            config = config.power

            return 1. - (torch.rand(x.size(0), device=x.device) ** config.power)
        else:
            raise Exception('Not supported ' + config.type)

    def get_noise_strength(self, x, config: NoiseStrengthConfig) -> torch.Tensor:
        if config.type == 'random':
            config = config.random

            use_noise_mask = (torch.rand(x.size(0), device=x.device) < config.noise_proba).float()
            noise_strength = use_noise_mask * self.get_random_distribution(x, config.random_distribution_config)
        elif config.type == 'max':
            noise_strength = torch.ones(x.size(0), dtype=torch.float, device=x.device)
        elif config.type == 'sample':
            config = config.sample

            batch_size = x.size(0)
            num_noised = int(batch_size * config.noise_samples_perc)

            # Create a mask with a fixed number of True values
            noise_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            noise_mask[:num_noised] = True

            # Shuffle the mask to randomize which images are noised
            noise_mask = noise_mask[torch.randperm(batch_size)]
            noise_strength = noise_mask * self.get_random_distribution(x, config.random_distribution_config)
        else:
            raise Exception('Not supported ' + config.type)

        return noise_strength

    def forward_noised(self, x: torch.Tensor, noise_level=1.) -> torch.Tensor:
        """
        :param noise_level: Level of noise
        :param x: [batch_size, C, H, W]
        """
        assert len(x.shape) == 4

        noise_strength = self.get_noise_strength(
            x,
            config=self.config.noise_strength_config
        )

        std_devs = noise_strength * noise_level * self.noise_scheduler.step()

        if self.config.use_norm:
            x_noised = apply_gaussian_noise_with_norm(x, std_devs)
        else:
            x_noised = apply_gaussian_noise(x, std_devs)

        x_noised = torch.clamp(x_noised, 0., 1.)

        x_encoded = self.encoder(x_noised)

        x_encoded[:, 0] = std_devs / (torch.max(self.noise_scheduler.initial_noise, self.noise_scheduler.final_noise))

        return x_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encoder(x)

        x_encoded[:, 0] = 0.

        return x_encoded

if __name__ == "__main__":
    from easydict import EasyDict

    noise = NoiseProcessorReprNetworkWrapper(
        encoder=nn.Identity(),
        config=EasyDict(
            dict(
                noise_strength_config=dict(
                    type='sample',
                    random=dict(
                        noise_proba=0.95,
                        random_distribution_config=dict(
                            type='max',
                            power=dict(
                                power=2.
                            )
                        )
                    ),
                    sample=dict(
                        noise_samples_perc=0.95,
                        random_distribution_config=dict(
                            type='max',
                            power=dict(
                                power=2.
                            )
                        )
                    )
                ),
                noise_scheduler=dict(
                    initial_noise = 0.25,
                    final_noise = 0.01,
                    schedule_length = 500,
                    decay_type = 'cos_cycle'
                )
            )
        )
    )

    for step in range(300):
        noise.forward_noised(torch.rand(650, 4, 4, 4))

