"""Score the package's built-in baselines (mpc_det, mpc_scen, ...) on cached episodes, each against the first (paired).

    uv run python scripts/baseline_scores.py --names='mpc_det;mpc_det|H=L+8;mpc_scen|S=4&H=L+4'
"""

import time

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow.dynamics.env import rollout
from shockbench_flow.policies.registry import PolicyContext, make_policy, params_class
from shockbench_flow_agent import EpisodeSet
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _policy_seed, _world


def _policy(spec):
    name, _, rest = spec.partition("|")
    if not rest:
        return make_policy(name, PolicyContext())
    kw = {}
    for item in rest.split("&"):
        k, v = item.split("=", 1)
        kw[k] = int(v) if v.isdigit() else {"True": True, "False": False}.get(v, v)
    return make_policy(name, PolicyContext(), params_class(name)(**kw))


def _one(spec, task, entropy, n, spec3, spec4):
    inst, omega, marks, fallback = _world(task, entropy, n, spec3, spec4)
    policy = _policy(spec)
    start = time.process_time()
    traj = rollout(inst, policy, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks,
                   fallback=fallback)
    return traj.J_cents, (time.process_time() - start) / inst.T


def main(names="mpc_det", task="small", episodes="dev", entropy=0, n_jobs=6):
    es = EpisodeSet.build(task, episodes, entropy=entropy, n_jobs=n_jobs)
    spec = es._spec
    naive = np.array([r["J_naive_cents"] for r in es.references], dtype=float)
    oracle = np.array([r["J_oracle_cents"] or 0 for r in es.references], dtype=float)
    names = names.split(";") if isinstance(names, str) else list(names)
    base = None
    for name in names:
        start = time.perf_counter()
        out = Parallel(n_jobs=n_jobs)(delayed(_one)(name, task, entropy, n, spec[3], spec[4]) for n in es.episodes)
        J = np.array([c for c, _ in out], dtype=float)
        table = es.rss(J)
        line = f"{name}: rss {table['rss']:.4f}"
        if base is None:
            base = J
        else:
            gain = (base - J) / np.maximum(naive - oracle, 1.0)
            se = gain.std(ddof=1) / np.sqrt(len(gain))
            line += f"  vs first {gain.mean():+.4f} +- {se:.4f} (wins {int((J < base).sum())}/{len(J)})"
        cpu = np.mean([w for _, w in out])
        print(f"{line}  cpu/week {cpu:.3f}s ({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
