import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import delta


# Define the gmmc_loss function
def gmmc_loss(e, alpha, beta):
    e = torch.abs(e)
    return beta ** alpha * (1 - torch.exp(-(e / beta) ** alpha))

# Define the derivative of gmmc_loss with respect to e
def gmmc_loss_derivative(e, alpha, beta):
    e = torch.abs(e)
    return alpha * beta ** (alpha - 1) * torch.exp(-(e / beta) ** alpha) * (e / beta) ** (alpha - 1)

# Parameters for plotting
e_values = torch.linspace(-3, 3, 2001)  # Range of e values
alpha_values = [1.1, 1.5, 2.0]      # Different alpha values within the range [1, 2]
beta_values = [1.0,10.0]                             # Fixed beta value for comparison


save_dir = "../runs_compare/gloss"
os.makedirs(save_dir, exist_ok=True)
# Plot gmmc_loss for different alpha values
plt.figure(figsize=(6, 4))
for alpha in alpha_values:
    for beta in beta_values:
        loss_values = gmmc_loss(e_values, alpha, beta)
        plt.plot(e_values.numpy(), loss_values.numpy(), label=f'α={alpha},β={beta}')

# Customize plot for gmmc_loss
plt.title("Generalized Multi-kernel Maximum Correntropy Loss")
plt.xlabel("e")
# plt.ylabel(r"J_{CL}(e, α, β)")
plt.ylabel(r"$J_{CL}(e, \alpha, \beta)$")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(save_dir, "gloss.png"))
plt.show()

# Plot the derivative of gmmc_loss for different alpha values
plt.figure(figsize=(6, 4))
for alpha in alpha_values:
    for beta in beta_values:
        derivative_values = gmmc_loss_derivative(e_values, alpha, beta)
        plt.plot(e_values.numpy(), derivative_values.numpy(), label=f'α={alpha},β={beta}')

# Customize plot for derivative of gmmc_loss
plt.title("Derivative of Generalized Multi-kernel Maximum Correntropy Loss")
plt.xlabel("e")
plt.ylabel(r"$\frac{\partial J_{CL}(e, \alpha, \beta)}{\partial e}$")
plt.legend(title="Alpha (α) values")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(save_dir, "gloss_derivative.png"))
plt.show()
