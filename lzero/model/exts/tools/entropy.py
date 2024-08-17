import torch
import torch.nn.functional as F

from lzero.model.exts.losses import entropy_cont

logits_observations = torch.tensor([
    [0.0,0.0,1.0],
    [14., 16., 12.],
    [12., 12., 12.],
    [12., 13., 12.],
    [1., 1., 1.],
    [0.33, 0.33, 0.33],
    [0.33, 0.5, 0.33],
])

epsilon = 1e-6

print(F.softmax(logits_observations, dim=-1))
print(entropy_cont(logits_observations))

logits_observations = torch.zeros([1, 768])

print(F.softmax(logits_observations, dim=-1))
print(entropy_cont(logits_observations))
