import torch
import torch.nn as nn
import torch.nn.functional as F


class CapSEM(nn.Module):
    def _init_(self, num_capsules, capsule_dim, temperature=1.0):
        super(CapSEM, self)._init_()

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.temperature = temperature

    def forward(self, x):
        # x имеет форму [batch_size, num_capsules, capsule_dim]
        batch_size = x.size(0)

        # Вычисляем нормы капсул
        norms = torch.norm(x, dim=-1)

        # Применяем softmax к нормам с учетом температуры
        softmax_norms = F.softmax(norms / self.temperature, dim=-1)

        # Нормализуем векторы капсул
        normalized_capsules = F.normalize(x, dim=-1)

        # Применяем softmax norms к нормализованным капсулам
        output = normalized_capsules * softmax_norms.unsqueeze(-1)

        return output
