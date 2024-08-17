import torch
import torch.nn.functional as F


def entropy_regularization(logits):
    prob_latent = F.softmax(logits, dim=-1)
    log_prob_latent = F.log_softmax(logits, dim=-1)

    return -(prob_latent * log_prob_latent).sum(dim=-1)
