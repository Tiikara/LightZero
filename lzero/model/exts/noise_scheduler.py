import torch
import math

class NoiseScheduler:
    def __init__(
            self,
            initial_noise: float = 0.25,
            final_noise: float = 0.01,
            schedule_length: int = 1000000,
            decay_type: str = 'linear',
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.initial_noise = torch.tensor(initial_noise, device=device)
        self.final_noise = torch.tensor(final_noise, device=device)
        self.schedule_length = schedule_length
        self.decay_type = decay_type
        self.device = device
        self.current_step = torch.tensor(0, device=device)
        self.current_noise = self.initial_noise.clone()

    def step(self):
        self.current_step += 1
        t = min(self.current_step.float() / self.schedule_length, 1.0)

        # Calculate base noise level based on decay type
        if self.decay_type == 'linear':
            noise = self.initial_noise + t * (self.final_noise - self.initial_noise)
        elif self.decay_type == 'constant':
            noise = self.initial_noise
        elif self.decay_type == 'exponential':
            noise = self.final_noise + (self.initial_noise - self.final_noise) * torch.exp(-5 * t)
        elif self.decay_type == 'cosine':
            noise = self.final_noise + 0.5 * (self.initial_noise - self.final_noise) * (1 + math.cos(math.pi * t))
        elif self.decay_type == 'step':
            noise = torch.where(t < 0.5, self.initial_noise, self.final_noise)
        elif self.decay_type == 'cos_cycle':
            cycle_step = self.current_step % self.schedule_length
            cycle_t = cycle_step.float() / self.schedule_length
            noise = self.final_noise + 0.5 * (self.initial_noise - self.final_noise) * (1 + torch.cos(2 * math.pi * cycle_t))
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        self.current_noise = noise
        return noise

    def get_noise(self):
        return self.current_noise


if __name__ == "__main__":
    NoiseScheduler()

    device = torch.device("cpu")
    scheduler = NoiseScheduler(
        initial_noise=0.25,
        final_noise=0.,
        schedule_length=100,
        decay_type='cos_cycle',
        device=device
    )

    for step in range(300):
        noise = scheduler.step()
        print(f"Step {step}, Noise: {noise.item():.4f}")
