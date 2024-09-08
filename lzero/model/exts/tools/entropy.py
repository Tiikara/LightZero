import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lzero.model.exts.add_dim_to_start_module import AddDimToStartModule
from lzero.model.exts.losses import entropy_softmax, entropy, entropy_linear, norm_l1, log_cosh_loss
from lzero.model.exts.noise_processor_repr_network_wrapper import NoiseProcessorReprNetworkWrapper
from lzero.model.exts.remove_first_dim_module import RemoveFirstDimModule

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

t = torch.rand(2, 2)
t = AddDimToStartModule(0)(t)
print(t.shape)
print(t)
print("slice")
print(t[:, 1:])
t = RemoveFirstDimModule()(t)
print(t.shape)
print(t)
