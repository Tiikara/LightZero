import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

from lzero.model.exts.activations.sgelu import SGELU


x = torch.linspace(-5, 5, 1000)

gelu_output = nn.GELU(approximate='tanh')(x)
sgelu_output = SGELU()(x)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), gelu_output.numpy(), label='GELU (tanh approx)')
plt.plot(x.numpy(), sgelu_output.numpy(), label='SGELU')
plt.xlabel('x')
plt.ylabel('y')
plt.title('GELU (tanh approx) / SGELU')
plt.legend()
plt.grid(True)
plt.show()
