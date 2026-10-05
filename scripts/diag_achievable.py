"""Diagnostic: the clairvoyant LP with the simulator's base-first rule in every week, scored as an RSS. Training only.

The plain clairvoyant LP may shed a grid's base load to power its fabs, which a base-first grid of the simulator never
does; pricing shed base load above any fab's use of the energy in every week gives a cost closer to what any policy can
reach. Its RSS is a rough ceiling for an agent on the same episodes.
"""

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, solve_oracle
from shockbench_flow_agent import EpisodeSet
from shockbench_flow_agent.scoring import _world


def _one(task, entropy, n, spec3, spec4):
    inst, omega, marks, fallback = _world(task, entropy, n, spec3, spec4)
    model = build_lp(inst, marks, planning_rules=True)
    prio = np.array(model.meta["priority"], dtype=float)
    first = prio.copy()
    for go in range(len(inst.grids)):
        j1 = model.index.get(("ysh", 1, go))
        if j1 is None or prio[j1] == 0:
            continue
        for t in range(1, inst.T + 1):
            prio[model.index[("ysh", t, go)]] = prio[j1]
    out = []
    for p in (np.zeros_like(prio), first, prio):
        model.meta["priority"] = p
        res = solve_oracle(model, method=ORACLE_METHOD)
        out.append(res.J_cents if res.J_cents is not None else np.nan)
    return out


def main(task="small", episodes="dev", entropy=0, n_jobs=6):
    es = EpisodeSet.build(task, episodes, entropy=entropy)
    spec = es._spec
    out = Parallel(n_jobs=n_jobs)(delayed(_one)(task, entropy, n, spec[3], spec[4]) for n in es.episodes)
    for i, label in enumerate(("rules, no prices", "rules, base-first week 1", "rules, base-first every week")):
        table = es.rss([o[i] for o in out])
        print(f"clairvoyant LP with {label}: rss {table['rss']:.4f} by level {table['rss_by_stratum']}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
