import torch
import torch.nn as nn
import torch.nn.functional as F

class UAN(nn.Module):
    """
    https://arxiv.org/abs/2409.04757
    """
    def __init__(self, num_features: int, num_clusters:int=3, eps:float=1e-5, affine:bool=True):
        super().__init__()

        self.num_features = num_features
        self.num_clusters = num_clusters
        self.eps = eps
        self.affine = affine

        # Инициализация параметров кластеров
        self.lambda_k = nn.Parameter(torch.ones(num_clusters) / num_clusters)
        self.mu_k = nn.Parameter(torch.randn(num_clusters, num_features))
        self.sigma_k = nn.Parameter(torch.rand(num_clusters, num_features) * 0.01 + 0.001)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Инициализация параметров, как описано в статье
        self.lambda_k.data.fill_(1 / self.num_clusters)
        nn.init.uniform_(self.mu_k, -1.0, 1.0)
        nn.init.uniform_(self.sigma_k, 0.001, 0.01)
        self.sigma_k.data.clamp_(min=self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, latent_dim]
        assert x.dim() == 2 and x.size(1) == self.num_features

        # posterior probabilities
        diff = x.unsqueeze(1) - self.mu_k.unsqueeze(0)  # [batch_size, num_clusters, latent_dim]
        squared_diff = (diff ** 2) / (2 * self.sigma_k.unsqueeze(0) ** 2 + self.eps)
        log_probs = -squared_diff.sum(dim=-1) - torch.log(self.sigma_k.prod(dim=-1) + self.eps).unsqueeze(0)
        log_probs += torch.log(self.lambda_k + self.eps).unsqueeze(0)
        posterior = F.softmax(log_probs, dim=1)  # [batch_size, num_clusters]

        # norm
        x_normalized = torch.zeros_like(x)
        for k in range(self.num_clusters):
            x_k = (x - self.mu_k[k]) / (self.sigma_k[k] + self.eps)
            x_normalized += posterior[:, k].unsqueeze(1) * torch.sqrt(self.lambda_k[k]) * x_k

        if self.affine:
            x_normalized = self.weight * x_normalized + self.bias

        return x_normalized
