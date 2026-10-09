"""The per-slot network: one MLP shared by every action slot, so it runs on any number of slots."""

import torch
from torch import nn


# flow = expm1(z) * SCALE * u0: z is the log of the flow in units of 1/1000 of the edge's nominal capacity, so a
# wafer route shipping 0.1 % of u0 weighs in the loss as much as a fuel pipe shipping half of it
SCALE = 1e-3

# Small and Full differ in size and difficulty: each board gets its own width and depth (and its own weights)
SIZES = {"small": (64, 2), "full": (128, 3)}


class SlotNet(nn.Module):
    def __init__(self, n_features, hidden, layers):
        super().__init__()
        dims = [n_features] + [hidden] * layers
        body = []
        for a, b in zip(dims, dims[1:]):
            body += [nn.Linear(a, b), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.out = nn.Linear(dims[-1], 1)
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, x):
        """x: (..., n_features) -> (...) z = log1p(flow / (SCALE * u0)), >= 0."""
        h = self.body((x - self.mean) / self.std)
        return nn.functional.softplus(self.out(h)).squeeze(-1)
