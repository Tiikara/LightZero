import torch
import torch.nn.functional as F


def log_gumbel_softmax(logits, tau=1., dim=-1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

    return F.log_softmax(gumbels, dim=dim)
