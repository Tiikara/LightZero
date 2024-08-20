import torch
import torch.nn.functional as F
import math

def norm_l1(x, eps=1e-6):
    return x / (x.sum(dim=-1).unsqueeze(-1) + eps)

def sign_preserving_normalization(x, epsilon=1e-10):
    # Разделяем положительные и отрицательные значения
    pos = torch.max(x, torch.zeros_like(x))
    neg = torch.abs(torch.min(x, torch.zeros_like(x)))

    total = torch.sum(pos, dim=-1) + torch.sum(neg, dim=-1) + epsilon
    pos_norm = pos / total
    neg_norm = neg / total

    return pos_norm - neg_norm

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # log(cosh(x)) = log((exp(x) + exp(-x)) / 2) = log(exp(x) * (1 + exp(-2x)) / 2 = x + log(1 + exp(-2x)) - log(2)
    x = y_pred - y_true
    return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0) # stable: torch.log(torch.cosh(x))

def entropy_softmax(logits):
    prob_latent = F.softmax(logits, dim=-1)
    log_prob_latent = F.log_softmax(logits, dim=-1)

    return -(prob_latent * log_prob_latent).sum(dim=-1)

def entropy_linear(logits, eps=1e-6):
    return entropy(norm_l1(logits))

def entropy(logits, eps=1e-6):
    return -(logits * torch.log(logits + eps)).sum(dim=-1)


def target_value_loss_relu(value, target_value, value_range):
    torch.relu(torch.abs(value - target_value) - value_range)

def quadratic_dead_zone_regularization(x, T1, T2, alpha=1.0):
    error = torch.maximum(T1 - x, torch.zeros_like(x)) + torch.maximum(x - T2, torch.zeros_like(x))
    return alpha * error**2

def smooth_quadratic_dead_zone_regularization(x, T1, T2, alpha=1.0, epsilon=1e-6):
    smooth_relu = lambda x_r: torch.sqrt(x_r**2 + epsilon**2) + x_r

    error_low = 0.5 * (smooth_relu(T1 - x) - (T1 - x))
    error_high = 0.5 * (smooth_relu(x - T2) - (x - T2))

    error = error_low + error_high
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
