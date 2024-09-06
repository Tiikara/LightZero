import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lzero.model.exts.losses import entropy_softmax, entropy, entropy_linear, norm_l1, log_cosh_loss
from lzero.model.exts.noise_processor_repr_network_wrapper import NoiseProcessorReprNetworkWrapper

logits_observations = torch.tensor([
    [0., 0.5, 0.11],
    [0., 0.5, 0.11],
    [1., 10., -1.0],
    [1., 10., 1.],
    [14., 16., 12.],
    [12., 12., 12.],
    [12., 13., 12.],
    [1., 1., 1.],
    [0.33, 0.33, 0.33],
    [0.33, 0.5, 0.33],
])

eps = 1e-6

print("=== Softmax")
print(F.softmax(logits_observations, dim=-1))
print("\n=== Normalize to 1 summ")
print(norm_l1(logits_observations))

print("\n=== Entropy softmax")
print(entropy_softmax(logits_observations))
print("\n=== Entropy Linear")
print(entropy_linear(logits_observations))

print(np.log(3))

print(torch.cosh(torch.tensor([-3566]) + eps))
print(log_cosh_loss(torch.tensor([-3566]), torch.tensor([0])))

class TestEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_create_layers = []

    def forward(self, x):
        return x.view(x.size(0), -1)

t = NoiseProcessorReprNetworkWrapper(TestEncoder(), max_noise=0.1).forward_noised(torch.rand((12, 3, 12, 12)))

print(torch.max(t), torch.min(t))
