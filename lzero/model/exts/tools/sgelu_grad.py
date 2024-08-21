import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

from lzero.model.exts.activations.sgelu import SGELU

x = torch.linspace(-5, 5, 1000, requires_grad=True)

sgelu = SGELU()
y = sgelu(x)
y.backward(torch.ones_like(x))
analytical_gradient = nn.GELU(approximate='tanh')(x)

plt.figure(figsize=(12, 8))
plt.plot(x.detach().numpy(), y.detach().numpy(), label='SGELU')
plt.plot(x.detach().numpy(), x.grad.numpy(), label='Numeric grad')
plt.plot(x.detach().numpy(), analytical_gradient.detach().numpy(), label='Analytic grad', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SGELU grad')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 8))
mask = (x > -0.5) & (x < 0.5)
plt.plot(x[mask].detach().numpy(), y[mask].detach().numpy(), label='SGELU')
plt.plot(x[mask].detach().numpy(), x.grad[mask].numpy(), label='Numeric grad')
plt.plot(x[mask].detach().numpy(), analytical_gradient[mask].detach().numpy(), label='Analytic grad', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SGELU grad (around zero)')
plt.legend()
plt.grid(True)

plt.show()
