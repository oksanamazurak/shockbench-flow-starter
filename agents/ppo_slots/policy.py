"""PPO's networks for ppo_slots: a per-slot policy (shared by every action slot) and a week value."""

import torch
from torch import nn


# Small and Full differ in size and difficulty: each board gets its own width and depth (and its own weights)
SIZES = {"small": (64, 2), "full": (128, 3)}
A_MIN, A_MAX = -4.0, 2.0  # the log multiplier of naive's flow: from 2 % of it to 7 times it


def _mlp(n_in, hidden, layers, n_out):
    dims = [n_in] + [hidden] * layers
    body = []
    for a, b in zip(dims, dims[1:]):
        body += [nn.Linear(a, b), nn.Tanh()]
    return nn.Sequential(*body, nn.Linear(dims[-1], n_out))


class SlotPolicy(nn.Module):
    """x (..., slots, features) -> mean log multiplier per slot; one log-std shared by every slot."""

    def __init__(self, n_features, hidden, layers):
        super().__init__()
        self.net = _mlp(n_features, hidden, layers, 1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # starts as naive: multiplier exp(0) = 1 on every slot
        self.log_std = nn.Parameter(torch.tensor(-0.5))
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


class WeekValue(nn.Module):
    """The value of a week: an MLP over the masked mean and max of the slot features."""

    def __init__(self, n_features, hidden):
        super().__init__()
        self.net = _mlp(2 * n_features, hidden, 2, 1)
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def forward(self, x, m):
        z = (x - self.mean) / self.std
        w = m.unsqueeze(-1)
        avg = (z * w).sum(-2) / w.sum(-2).clamp_min(1.0)
        top = torch.where(w > 0, z, torch.full_like(z, -10.0)).max(-2).values
        return self.net(torch.cat([avg, top], -1)).squeeze(-1)
