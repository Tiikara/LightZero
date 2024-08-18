import torch
import numpy as np
import matplotlib.pyplot as plt

from lzero.model.exts.losses import dead_zone_regularization, quadratic_dead_zone_regularization

T1, T2 = 0.25, 0.75
alpha = 1.0

x = torch.linspace(-0.5, 1.5, 1000)

y = quadratic_dead_zone_regularization(x, T1, T2, alpha)

x_np = x.numpy()
y_np = y.numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_np, y_np)
plt.title('Dead Zone Regularization')
plt.xlabel('x')
plt.ylabel('Regularization Value')

plt.axvline(x=T1, color='r', linestyle='--', label='T1')
plt.axvline(x=T2, color='g', linestyle='--', label='T2')

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.legend()
plt.grid(True)

plt.ylim(-0.1, max(y_np)*1.1)

plt.show()
