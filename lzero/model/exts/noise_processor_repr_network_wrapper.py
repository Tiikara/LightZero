from dataclasses import dataclass
from typing import List

import random
import torch
from torch import nn

from lzero.model.exts.noise_scheduler import NoiseScheduler


@dataclass
class NoiseRandomDistributionPowerConfig:
    power: float

@dataclass
class NoiseRandomDistributionSelectRandConfig:
    noise_proba: float

@dataclass
class NoiseRandomDistributionSampleConfig:
    noise_samples_perc: float

@dataclass
class NoiseRandomDistributionSampleSeqConfig:
    seq_length: int
    noise_samples_perc: float

@dataclass
class NoiseRandomDistributionInSeqSampleRangeSeqConfig:
    seq_length: int
    noise_samples_perc_from: float
    noise_samples_perc_to: float

@dataclass
class NoiseRandomDistributionConfig:
    type: str
    sample_seq: NoiseRandomDistributionSampleSeqConfig
    sample: NoiseRandomDistributionSampleConfig
    select_rand: NoiseRandomDistributionSelectRandConfig
    rand_power: NoiseRandomDistributionPowerConfig
    in_seq_sample_range: NoiseRandomDistributionInSeqSampleRangeSeqConfig

@dataclass
class NoiseStrengthConfig:
    mult_random_distributions: List[NoiseRandomDistributionConfig]

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

        if hasattr(encoder, 'out_create_layers'):
            self.out_create_layers = encoder.out_create_layers

        if hasattr(encoder, 'projection_model'):
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
        if config.type == 'rand_linear':
            return torch.rand(x.size(0), device=x.device)
        elif config.type == 'select_rand':
            config = config.select_rand

            return (torch.rand(x.size(0), device=x.device) < config.noise_proba).float()
        if config.type == 'max':
            return torch.ones(x.size(0), dtype=torch.float, device=x.device)
        elif config.type == 'rand_power':
            config = config.rand_power

            return 1. - (torch.rand(x.size(0), device=x.device) ** config.power)
        elif config.type == 'sample':
            config = config.sample

            batch_size = x.size(0)
            num_noised = int(batch_size * config.noise_samples_perc)

            # Create a mask with a fixed number of True values
            noise_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            noise_mask[:num_noised] = True

            # Shuffle the mask to randomize which images are noised
            return noise_mask[torch.randperm(batch_size)].float()
        elif config.type == 'sample_seq':
            config = config.sample_seq

            batch_size = x.size(0)

            # Create a mask with a fixed number of True values
            noise_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            noise_mask = noise_mask.view(-1, config.seq_length)

            num_noised = int(noise_mask.size(0) * config.noise_samples_perc)
            noise_mask[:num_noised, :] = True

            # Shuffle the mask to randomize which batch are noised
            return noise_mask[torch.randperm(noise_mask.size(0)), :].view(batch_size).float()
        elif config.type == 'in_seq_sample_range':
            config = config.in_seq_sample_range

            batch_size = x.size(0)

            # Create a mask with a fixed number of True values
            noise_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            noise_mask = noise_mask.view(-1, config.seq_length)

            num_noised = int(noise_mask.size(1) * random.uniform(config.noise_samples_perc_from, config.noise_samples_perc_to))
            if num_noised == 0:
                return torch.zeros(x.size(0), dtype=torch.float, device=x.device)
            else:
                noise_mask[:, :num_noised] = True

                shuffled_indices = torch.argsort(torch.rand_like(noise_mask.float(), device=noise_mask.device), dim=1)

                shuffled_mask = torch.gather(noise_mask, 1, shuffled_indices)

                print(shuffled_mask)

                return shuffled_mask.view(batch_size).float()
        else:
            raise Exception('Not supported ' + config.type)

    def get_noise_strength(self, x, config: NoiseStrengthConfig) -> torch.Tensor:
        noise_strength = None

        for random_distribution_config in config.mult_random_distributions:
            distribution = self.get_random_distribution(x, random_distribution_config)

            if noise_strength is None:
                noise_strength = distribution
            else:
                noise_strength *= distribution

        return noise_strength

    def forward_noised(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, C, H, W]
        """
        assert len(x.shape) == 4

        noise_strength = self.get_noise_strength(
            x,
            config=self.config.noise_strength_config
        )

        std_devs = noise_strength * self.noise_scheduler.step()

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
                use_norm=False,
                noise_strength_config=dict(
                    mult_random_distributions=[
                        dict(
                            type='sample_seq',
                            sample_seq=dict(
                                noise_samples_perc=0.75,
                                seq_length=10
                            )
                        ),
                        dict(
                            type='in_seq_sample_range',
                            in_seq_sample_range=dict(
                                noise_samples_perc_from=0.7,
                                noise_samples_perc_to=1.,
                                seq_length=10
                            )
                        ),
                        dict(
                            type='rand_linear'
                        )
                    ]
                ),
                noise_scheduler=dict(
                    initial_noise = 0.25,
                    final_noise = 0.,
                    schedule_length = 1000,
                    decay_type = 'constant'
                )
            )
        )
    )

    for step in range(300):
        noise.forward_noised(torch.rand(600, 4, 4, 4))

