import torch
import torch.nn.functional as F
import numpy as np

from lzero.model.exts.losses import entropy_softmax, entropy, entropy_linear

logits_observations = torch.tensor([
    [0., 0., 0.],
    [0., 0., 0.11],
    [1., 10., 1.],
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
print(logits_observations / (logits_observations.sum(dim=-1).unsqueeze(-1) + eps))

print("\n=== Entropy softmax")
print(entropy_softmax(logits_observations))
print("\n=== Entropy Linear")
print(entropy_linear(logits_observations))

print(np.log(3))
