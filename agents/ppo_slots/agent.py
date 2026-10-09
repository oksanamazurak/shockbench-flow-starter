"""ppo_slots: a per-slot policy trained by PPO (scripts/ppo_slots.py) that scales the naive rule's flows.

Each week the naive rule (mpc2's vendored fallback, no LP) gives a base plan; ``features.Features`` describes every
action slot; the policy returns a log multiplier per slot and the agent ships naive * exp(a). Small and Full have their
own weights (policy_small.pt, policy_full.pt, chosen by T); without them the agent plays naive.

Training (``TRAIN`` set by the trainer's copy of this folder): actions are sampled with the policy's std and every
week's features, mask, actions and log-probabilities are written to an .npz for the PPO update.
"""

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from features import Features  # noqa: E402
from mpc2_base import Agent as Mpc2  # noqa: E402
from policy import A_MAX, A_MIN, SlotPolicy  # noqa: E402


torch.set_num_threads(1)
TRAIN = None  # the trainer writes {"out": folder} here in its copy


def _load(task):
    path = HERE / f"policy_{task}.pt"
    if not path.is_file():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = SlotPolicy(ck["n_features"], ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state"])
    return net.eval()


NETS = {task: _load(task) for task in ("small", "full")}  # loaded at import: outside the first week's CPU time


class Agent(Mpc2):
    def __init__(self, config):
        super().__init__(config)
        self.feat = Features(config)
        self.net = NETS["full" if int(config["T"]) > 60 else "small"]
        self.rng = np.random.default_rng(int(config["policy_seed"]) % 2**32)
        self.rows = {"X": [], "mask": [], "a": [], "mu": []}

    def act(self, observation):
        obs = self.decoder.decode(observation)
        if not self.started:
            self.policy.reset(self.static, obs, self.seed)
            self.started = True
        mask = np.asarray(observation["action_mask"], dtype=float)
        seen = observation.get("action_mask.observed")
        if seen is not None and np.asarray(seen).reshape(-1)[0] == 0:
            mask = np.ones_like(mask)
        action = self._flat(self.policy._fallback.act(obs))
        naive = action["flows"] * mask
        if self.net is None:
            action["flows"] = naive
            return action
        X = self.feat.week(observation, naive)
        with torch.no_grad():
            mu = self.net(torch.from_numpy(X)).numpy().astype(np.float64)
        a = mu
        if TRAIN is not None:
            a = mu + float(torch.exp(self.net.log_std.detach())) * self.rng.standard_normal(len(mu))
            live = (mask > 0) & (naive > 0)
            self.rows["X"].append(X)
            self.rows["mask"].append(live.astype(np.float32))
            self.rows["a"].append(a.astype(np.float32))
            self.rows["mu"].append(mu.astype(np.float32))
            if int(np.asarray(observation["week"]).reshape(-1)[0]) >= self.feat.T:
                name = Path(TRAIN["out"]) / f"{self.seed}.npz"
                np.savez_compressed(name, **{k: np.stack(v) for k, v in self.rows.items()})
        action["flows"] = naive * np.exp(np.clip(a, A_MIN, A_MAX))
        return action
