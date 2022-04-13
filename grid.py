# %%
import torch
from torch.distributions import Normal
from torch.nn.functional import softplus
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from tqdm import tqdm

# %%

num_samples = 1000
num_samples_entropy = 10000


def entropy(μ, σ):
    pi_distribution = Normal(μ, σ)
    pi_action = pi_distribution.sample((num_samples_entropy,))

    logp_pi = pi_distribution.log_prob(pi_action)
    logp_pi -= 2 * (np.log(2) - pi_action - softplus(-2 * pi_action))

    return -logp_pi.mean().item()


μs = np.linspace(-2, 2, num_samples)
σs = np.logspace(0.1, 10, num_samples)
σs = np.linspace(np.exp(-20), 1, num_samples)
σs = np.linspace(1e-3, 3, num_samples)


X, Y = np.meshgrid(μs, σs)


# %%

data = np.zeros((num_samples, num_samples))

for idx, idy in tqdm(product(range(num_samples), range(num_samples)), ncols=100):
    μ = X[idx, idy]
    σ = Y[idx, idy]
    result = entropy(μ, σ)
    # print(f"{μ:.4f}, {σ:.4f}, {result:.4f}")
    data[idx, idy] = result

Z = data

# %%
fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp)  # Add a colorbar to a plot
ax.set_title("Entropy")
ax.set_xlabel("μ")
ax.set_ylabel("σ")
plt.show()


# plt.contour(μs, σs, data)
# plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, Z)
ax.set_xlabel("μ")
ax.set_ylabel("σ")
ax.set_zlabel("Entropy")
plt.show()

# %%
idx, idy = np.unravel_index(np.argmax(data), np.array(data).shape)
μ = X[idx, idy]
σ = Y[idx, idy]
print(f"μ: {μ}, σ: {σ}")

# %%

data = np.load("data.npy")

# %%
