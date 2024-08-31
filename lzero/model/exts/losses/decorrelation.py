import torch
import torch.nn as nn

from lzero.model.exts.losses import log_cosh_loss


def decorrelation_reg(latent_representations: torch.Tensor):
    cov_matrix = torch.cov(latent_representations.T)

    identity = torch.eye(latent_dim, device=latent_representations.device)
    independence_loss = ((cov_matrix - identity) ** 2).mean()

    return independence_loss

if __name__ == "__main__":
    latent_dim = 4
    batch_size = 8

    input_data = torch.zeros(batch_size, latent_dim)

    ind_loss = decorrelation_reg(input_data)

    print(f"Independence regularization loss: {ind_loss.item()}")

    # total_loss = main_loss + lambda * ind_loss
