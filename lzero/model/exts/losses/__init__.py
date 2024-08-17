import torch
import torch.nn.functional as F


def entropy_cont(logits):
    prob_latent = F.softmax(logits, dim=-1)
    log_prob_latent = F.log_softmax(logits, dim=-1)

    return -(prob_latent * log_prob_latent).sum(dim=-1)


def target_value_loss_relu(value, target_value, value_range):
    torch.relu(torch.abs(value - target_value) - value_range)

def quadratic_dead_zone_regularization(x, T1, T2, alpha=1.0):
    error = torch.maximum(T1 - x, torch.zeros_like(x)) + torch.maximum(x - T2, torch.zeros_like(x))
    return alpha * error**2

def dead_zone_regularization(x, T1, T2, alpha=1.0):
    return alpha * torch.relu(torch.maximum(T1 - x, x - T2))


def target_value_loss_quadratic(value, target_value):
    return (value - target_value)**2


def target_value_loss_log(value, target_value, value_range):
    def log_barrier(x, low, high):
        return -torch.log(high - x) - torch.log(x - low)

    low = target_value - value_range
    high = target_value + value_range
    return log_barrier(value, low, high)
