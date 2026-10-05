"""Score parameter sets of agents/mpc on episodes of a training root, each against the first set (paired).

    uv run python scripts/tune_mpc.py --task=small --episodes=24 --entropy=1 \
        --grid='[{}, {"closure_end": true}, {"H_extra": 4}]'
"""

import functools
import json
import os
import sys
import time
from pathlib import Path

import fire
import numpy as np
from shockbench_flow_agent import EpisodeSet

AGENT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--agent=")), "mpc")
AGENT_DIR = Path(__file__).resolve().parents[1] / "agents" / AGENT
sys.path.insert(0, str(AGENT_DIR))
os.environ["PYTHONPATH"] = os.pathsep.join([str(AGENT_DIR), os.environ.get("PYTHONPATH", "")])

import agent as mpc_agent  # noqa: E402


def main(grid="[{}]", task="small", episodes=24, entropy=1, n_jobs=6, dev=False, agent="mpc"):
    grid = json.loads(grid) if isinstance(grid, str) else grid
    es = EpisodeSet.build(task, "dev" if dev else episodes, entropy=0 if dev else entropy, n_jobs=n_jobs)
    naive = np.array([r["J_naive_cents"] for r in es.references], dtype=float)
    oracle = np.array([r["J_oracle_cents"] or 0 for r in es.references], dtype=float)
    base = None
    for params in grid:
        start = time.perf_counter()
        factory = functools.partial(mpc_agent.Agent, params=params)
        rows = es.play(factory, n_jobs=n_jobs)
        J = np.array([r["J_policy_cents"] for r in rows], dtype=float)
        rss = es.rss(J)
        line = f"rss {rss['rss']:.4f}"
        if base is None:
            base = J
        else:
            gain = (base - J) / np.maximum(naive - oracle, 1.0)
            se = gain.std(ddof=1) / np.sqrt(len(gain)) if len(gain) > 1 else 0.0
            line += f"  vs first: {gain.mean():+.4f} +- {se:.4f} (wins {int((J < base).sum())}/{len(J)})"
        fallback = sum(r["fallback_weeks"] for r in rows)
        weeks = [w for r in rows for w in (r.get("cpu_weeks") or [])]
        cpu = f"cpu/week mean {np.mean(weeks):.2f}s max {np.max(weeks):.2f}s" if weeks else ""
        print(f"{line}  fb {fallback}  {cpu}  {time.perf_counter() - start:.0f}s  {json.dumps(params)}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
