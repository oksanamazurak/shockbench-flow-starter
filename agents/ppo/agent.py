"""PPO submission (agents/mine_heuristic/train_ppo.py trains it, writes policy.pt here): the server has torch, not SB3.

``ppo_features.summarize_warnings`` is also used by ``agents/mine_heuristic/train_ppo.py``'s training-time observation
wrapper, so the feature the policy was trained on and the feature computed here at inference are the same function.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from ppo_features import summarize_warnings


# loaded at import, which the CPU budget does not meter
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # torch 2.14 marks TorchScript deprecated; it still loads
    POLICY = torch.jit.load(str(Path(__file__).resolve().parent / "policy.pt"), map_location="cpu")
POLICY.eval()
KEYS = list(POLICY.obs_keys)  # the observation fields of the flat vector, in order


class Agent:
    def __init__(self, config):
        u0 = config["static"]["edges"]["u0"]  # each edge's nominal capacity per week
        self.action_edges = np.array(config["static"]["action_slots"]["edge"])
        self.capacity = np.array([u0[e] for e in self.action_edges], dtype=float)
        self.horizon = int(config["static"]["T"])

    def act(self, observation):
        obs = dict(observation)
        obs["warning_summary"] = summarize_warnings(
            obs, self.action_edges, self.capacity, float(np.asarray(obs["week"]).reshape(())), self.horizon
        )
        x = np.concatenate([np.asarray(obs[k], dtype=np.float64).ravel() for k in KEYS])
        with torch.inference_mode():
            fraction = POLICY(torch.from_numpy(x)).numpy()  # of capacity, in [0, 1]
        return {"flows": fraction * self.capacity * observation["action_mask"]}
