"""nn_mpc: a per-slot network trained to imitate agents/mpc2, on top of the naive rule.

Each week the naive rule (mpc2's vendored fallback, no LP) gives a base plan; ``features.Features`` describes every
action slot; the network (``model.SlotNet``, one MLP shared by every slot) returns the flow on a log scale of the
edge's nominal capacity. Small and Full have their own weights (weights_small.pt, weights_full.pt, chosen by T);
without them the agent plays naive. Overrides and release modes are naive's.
"""

import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from features import Features  # noqa: E402
from model import SCALE, SlotNet  # noqa: E402
from mpc2_base import Agent as Mpc2  # noqa: E402


torch.set_num_threads(1)


def _load(task):
    path = HERE / f"weights_{task}.pt"
    if not path.is_file():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = SlotNet(ck["n_features"], ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state"])
    return net.eval()


NETS = {task: _load(task) for task in ("small", "full")}  # loaded at import: outside the first week's CPU time


class Agent(Mpc2):
    def __init__(self, config):
        super().__init__(config)
        self.feat = Features(config)
        self.net = NETS["full" if int(config["T"]) > 60 else "small"]

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
        if self.net is not None:
            X = torch.from_numpy(self.feat.week(observation, naive))
            with torch.no_grad():
                z = self.net(X).numpy().astype(np.float64)
            action["flows"] = np.expm1(np.minimum(z, 20.0)) * SCALE * self.feat.u0 * mask
        else:
            action["flows"] = naive
        return action
