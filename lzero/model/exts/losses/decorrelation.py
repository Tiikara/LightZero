import torch
import torch.nn as nn


def decorrelation_reg(latent_representations: torch.Tensor):
    batch_size, latent_dim = latent_representations.shape

    centered_representations = latent_representations - latent_representations.mean(dim=0, keepdim=True)

    cov_matrix = (centered_representations.T @ centered_representations) / (batch_size - 1)

    identity = torch.eye(latent_dim, device=latent_representations.device)
    independence_loss = ((cov_matrix - identity) ** 2).mean()

    return independence_loss

if __name__ == "__main__":
    latent_dim = 768
    batch_size = 64

    input_data = torch.randn(batch_size, latent_dim)

    ind_loss = decorrelation_reg(input_data)

    print(f"Independence regularization loss: {ind_loss.item()}")

    # total_loss = main_loss + lambda * ind_loss
