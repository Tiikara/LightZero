import torch

def generate_noise_levels(batch_size, num_levels=10, device="cpu"):
    raw_base_levels = torch.linspace(0, 1, num_levels, device=device)

    base_levels = raw_base_levels.unsqueeze(0).repeat(batch_size, 1)

    random_offsets = torch.rand(batch_size, num_levels - 2, device=device) * 0.09
    random_signs = torch.randint(0, 2, (batch_size, num_levels - 2), device=device) * 2 - 1

    noise_levels = base_levels.clone()
    noise_levels[:, 1:-1] += random_offsets * random_signs

    noise_levels = torch.clamp(noise_levels, 0, 1)

    return raw_base_levels, noise_levels

if __name__ == "__main__":
    batch_size = 8
    noise_levels = generate_noise_levels(batch_size)
    print(noise_levels.shape)  # torch.Size([32, 10])
    print(noise_levels)
    print(noise_levels[0])
    print(noise_levels[1])
