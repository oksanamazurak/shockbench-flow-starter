"""Diagnostic: mpc_det whose window sees the true marks (all, demand only, or all but demand). Training-time only.

Splits mpc_det's gap to the clairvoyant plan into the part a better forecast could close and the part the rolling
window itself leaves.
"""

import time

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow.dynamics.env import rollout
from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
from shockbench_flow.policies.registry import PolicyContext
from shockbench_flow_agent import EpisodeSet
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _policy_seed, _world


class PeekMpc(MpcDet):
    def __init__(self, marks, mode, H, rules=True):
        self.weeks = int(H) if str(H).isdigit() else None
        super().__init__(MpcDetParams(H="L" if self.weeks else H, planning_rules=rules), PolicyContext())
        self.marks, self.mode = marks, mode

    def _horizon(self, inst, plan):
        return self.weeks if self.weeks else super()._horizon(inst, plan)

    def _window_arrays(self, inst, obs, H_t):
        base = {k: np.array(v) for k, v in super()._window_arrays(inst, obs, H_t).items()}
        t = int(obs["week"])
        true = {k: np.array(getattr(self.marks, k)[t - 1 : t - 1 + H_t]) for k in base}
        if self.mode == "all":
            out = true
        elif self.mode == "demand":
            out = {**base, "demand": true["demand"]}
        elif self.mode == "graph":
            out = {**true, "demand": base["demand"]}
        else:
            out = base
        return L.read_only(L.with_now(out) if self.mode != "all" else out)


def _one(mode, H, rules, task, entropy, n, spec3, spec4):
    inst, omega, marks, fallback = _world(task, entropy, n, spec3, spec4)
    policy = PeekMpc(marks, mode, H, rules)
    traj = rollout(inst, policy, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks,
                   fallback=fallback)
    comp = {}
    for r in traj.records:
        for k, v in r.costs.as_dict().items():
            comp[k] = comp.get(k, 0.0) + v
    comp["salvage"] = -traj.salvage
    return traj.J_cents, comp


def main(modes="none,demand,graph,all", H="L", rules=True, task="small", episodes="dev", entropy=0, n_jobs=6):
    es = EpisodeSet.build(task, episodes, entropy=entropy)
    spec = es._spec
    modes = modes.split(",") if isinstance(modes, str) else list(modes)
    for mode in modes:
        start = time.perf_counter()
        out = Parallel(n_jobs=n_jobs)(
            delayed(_one)(mode, H, rules, task, entropy, n, spec[3], spec[4]) for n in es.episodes
        )
        table = es.rss([c for c, _ in out])
        comp = {k: np.mean([c[k] for _, c in out]) / 1e9 for k in out[0][1]}
        print(f"{mode} H={H} rules={rules}: rss {table['rss']:.4f} by level {table['rss_by_stratum']} "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
        print("   $bn/episode: " + ", ".join(f"{k} {v:.1f}" for k, v in comp.items()), flush=True)


if __name__ == "__main__":
    fire.Fire(main)
