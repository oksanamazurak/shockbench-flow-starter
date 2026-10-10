"""Mean value of a unit of stock per (week, stock slot), from the clairvoyant LP's duals, for agents/mpc2.

    uv run python scripts/terminal_values.py --task=small --root=9301 --episodes=60
    uv run python scripts/terminal_values.py --task=full --root=9401 --episodes=16

For every episode the oracle LP (``shockbench_flow.oracle.lp.build_lp``) is solved by HiGHS and the marginal of each
stock balance row ("balance", t, slot) is read: minus it is what one more unit of stock arriving in week t would save.
Its mean over the episodes (roots kept apart from the bench suites, dev and tuning) is written to
agents/mpc2/terminal_<task>.npz as ``value`` (T + 1, slots): row t is the value of a unit held at the end of week t
(row T: the salvage the LP credits). mpc2 prices the stock left at the end of its window with it.
"""

import time

import fire
import numpy as np
from joblib import Parallel, delayed

from sbf_starter import ROOT


def _one(task, root, n, grid_hold=0.0):
    from scipy.optimize import linprog
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import split_label
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.oracle.lp import build_lp

    from sbf_starter.bench import _generator

    inst, params = _generator(task, None)
    marks = compute_marks(inst, sample_omega(inst, params, root, n, split_label(root)))
    m = build_lp(inst, marks)
    kw = {}
    if m.A_ub.shape[0]:
        kw.update(A_ub=m.A_ub, b_ub=m.b_ub)
    c = np.array(m.objective(), dtype=float)
    if grid_hold:  # as agents/mpc2's grid_hold: LNG left at a grid at a week's end costs grid_hold
        for s_, st in enumerate(inst.stock_slots):
            nd = inst.nodes[st.node]
            if nd.type == "grid" and float(nd.grid.days_cover.get(st.k, 0.0)) < 90:
                for t_ in range(1, inst.T + 1):
                    j = m.index.get(("I", t_, s_))
                    if j is not None:
                        c[j] += grid_hold
    res = linprog(c, A_eq=m.A_eq, b_eq=m.b_eq, bounds=np.column_stack([m.lb, m.ub]), method="highs", **kw)
    T, S = inst.T, len(inst.stock_slots)
    value = np.zeros((T + 1, S))
    if res.status != 0:
        return None
    lam = np.asarray(res.eqlin.marginals)
    for r, name in enumerate(m.eq_names):
        if name[0] == "balance":
            t, slot = int(name[1]), int(name[2])
            if 2 <= t <= T:
                value[t - 1, slot] = -lam[r]  # held at the end of week t - 1, it enters week t's balance
    for slot, st in enumerate(inst.stock_slots):
        value[T, slot] = float(st.salvage)
    return value


def main(
    task: str = "small",
    root: int = 9301,
    episodes: int = 60,
    n_jobs: int = 8,
    out: str | None = None,
    grid_hold: float = 0.0,
) -> None:
    start = time.perf_counter()
    from sbf_starter.bench import _generator

    _generator(task, None, n_jobs=n_jobs)
    jobs = (delayed(_one)(task, root, n, grid_hold) for n in range(episodes))
    vals = [v for v in Parallel(n_jobs=n_jobs)(jobs) if v is not None]
    value = np.mean(vals, axis=0)
    out = ROOT / "agents" / "mpc2" / f"terminal_{task}.npz" if out is None else ROOT / out
    np.savez_compressed(out, value=value, episodes=len(vals), root=root)
    print(f"{len(vals)} episodes in {time.perf_counter() - start:.0f} s -> {out}")
    positive = float(np.median(value[value > 0]))
    print("value range", float(value.min()), float(value.max()), "median of positive", positive)


if __name__ == "__main__":
    fire.Fire(main)
