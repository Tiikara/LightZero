import torch
import torch.nn.functional as F



logits_observations = torch.tensor([[0.0,0.0,1.0], [14., 16., 12.], [12., 12., 12.], [1., 1., 1.], [0.33, 0.33, 0.33]])

epsilon = 1e-6

prob_latent = F.softmax(logits_observations, dim=-1)

log_prob_latent = F.log_softmax(logits_observations, dim=-1)

reg_loss_entropy = (prob_latent * log_prob_latent).sum(dim=-1)


print(reg_loss_entropy)
