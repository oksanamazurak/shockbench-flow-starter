"""Diagnostic: the clairvoyant LP's cost components against mpc_det's realised ones, per episode. Training-time only."""

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow.dynamics.env import rollout
from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, lp_costs, solve_oracle
from shockbench_flow.policies.registry import PolicyContext, make_policy
from shockbench_flow_agent import EpisodeSet
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _policy_seed, _world

KEYS = ("freight", "tariff", "holding", "shortage", "disposal", "shed")


def _one(task, entropy, n, spec3, spec4):
    inst, omega, marks, fallback = _world(task, entropy, n, spec3, spec4)
    model = build_lp(inst, marks)
    res = solve_oracle(model, method=ORACLE_METHOD)
    weekly, salv = lp_costs(model, res.x)
    oracle = np.array([[getattr(w, k) for k in KEYS] for w in weekly])
    traj = rollout(inst, make_policy("mpc_det", PolicyContext()), omega, "standard",
                   _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    mpc = np.array([[getattr(r.costs, k) for k in KEYS] for r in traj.records])
    shed_o = np.array([sum(res.x[model.index[("ysh", t, g)]] for g in range(len(inst.grids))) for t in range(1, inst.T + 1)])
    shed_m = np.array([float(np.sum(r.shed)) for r in traj.records])
    return n, oracle, salv, mpc, traj.salvage, shed_o, shed_m


def main(task="small", episodes="dev", entropy=0, n_jobs=6):
    es = EpisodeSet.build(task, episodes, entropy=entropy)
    spec = es._spec
    out = Parallel(n_jobs=n_jobs)(delayed(_one)(task, entropy, n, spec[3], spec[4]) for n in es.episodes)
    print("episode: component oracle/mpc ($bn)")
    tot_o = np.zeros(len(KEYS))
    tot_m = np.zeros(len(KEYS))
    for n, o, so, m, sm, shed_o, shed_m in out:
        tot_o += o.sum(0)
        tot_m += m.sum(0)
        parts = ", ".join(f"{k} {a / 1e9:.0f}/{b / 1e9:.0f}" for k, a, b in zip(KEYS, o.sum(0), m.sum(0)))
        print(f"{n}: {parts}, salvage {so / 1e9:.1f}/{sm / 1e9:.1f}")
        if n == out[0][0]:
            print("   shed by week oracle:", np.round(shed_o, 0).tolist())
            print("   shed by week mpc   :", np.round(shed_m, 0).tolist())
            print("   shortage by week oracle:", np.round(o[:, 3] / 1e9, 1).tolist())
            print("   shortage by week mpc   :", np.round(m[:, 3] / 1e9, 1).tolist())
    print("mean:", ", ".join(f"{k} {a / 1e9 / len(out):.1f}/{b / 1e9 / len(out):.1f}" for k, a, b in zip(KEYS, tot_o, tot_m)))


if __name__ == "__main__":
    fire.Fire(main)
