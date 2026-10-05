"""An evolutionary search over agents/mpc's forecast and LP parameters (same method as 06_policy_search.py).

    uv run python examples/07_mpc_policy_search.py
    uv run python examples/07_mpc_policy_search.py --task=small --generations=10 --population=12

A candidate is the mpc agent's agent.py plus a params.json overriding warn_a, warn_b, msg_weight (one number
per message kind 0-4), msg_bump, msg_bump_weeks, holding_scale, closure_power, H, tariff_bump and
tariff_weight (one per tariff channel). Fitness is its score on your
own root, under the CPU budget. The best is kept only if it beats the start (this session's default PARAMS) on
the held-out dev episodes.
"""

import json
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import fire
import numpy as np

from sbf_starter import scoring
from sbf_starter.agents import resolve


MPC = resolve("mpc") / "agent.py"
# order: warn_a, warn_b, msg_weight[0..4], msg_bump, msg_bump_weeks, holding_scale, closure_power, H,
#        tariff_bump, tariff_weight[0..2]
LOWER = np.array([0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
UPPER = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 6.0, 5.0, 3.0, 10.0, 0.5, 1.0, 1.0, 1.0])


def _load_defaults() -> dict:
    """``agents/mpc_lp/agent.py``'s PARAMS, loaded by path (``agents`` is a loose folder, not an installed
    package, so a dotted ``import agents.mpc.agent`` is not reliable from a script run as ``examples/07_...py``).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("mpc_agent_defaults", MPC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PARAMS


def start_params() -> np.ndarray:
    DEFAULT = _load_defaults()
    weight = DEFAULT["msg_weight"]
    return np.array(
        [
            DEFAULT["warn_a"],
            DEFAULT["warn_b"],
            weight.get("0", 0.0),
            weight.get("1", 0.0),
            weight.get("2", 0.0),
            weight.get("3", 0.0),
            weight.get("4", 0.0),
            DEFAULT["msg_bump"],
            DEFAULT["msg_bump_weeks"],
            DEFAULT.get("holding_scale", 1.0),
            DEFAULT["closure_power"],
            DEFAULT["H"],
            DEFAULT.get("tariff_bump", 0.0),
            DEFAULT.get("tariff_weight", {}).get("0", 0.0),
            DEFAULT.get("tariff_weight", {}).get("1", 0.0),
            DEFAULT.get("tariff_weight", {}).get("2", 0.0),
        ]
    )


def write_candidate(params: np.ndarray, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(MPC, folder / "agent.py")
    clipped = np.clip(params, LOWER, UPPER)
    numbers = {
        "warn_a": round(float(clipped[0]), 4),
        "warn_b": round(float(clipped[1]), 4),
        "msg_weight": {str(i): round(float(clipped[2 + i]), 4) for i in range(5)},
        "msg_bump": round(float(clipped[7]), 4),
        "msg_bump_weeks": int(round(clipped[8])),
        "holding_scale": round(float(clipped[9]), 4),
        "closure_power": round(float(clipped[10]), 4),
        "H": int(round(clipped[11])),
        "tariff_bump": round(float(clipped[12]), 4),
        "tariff_weight": {str(i): round(float(clipped[13 + i]), 4) for i in range(3)},
    }
    (folder / "params.json").write_text(json.dumps(numbers) + "\n")
    return folder


def mutate(parents: list, rng: np.random.Generator, n: int, sigma: float = 0.15) -> list:
    out = []
    span = UPPER - LOWER
    for _ in range(n):
        p = parents[rng.integers(len(parents))]
        child = np.clip(p + rng.normal(0.0, sigma, p.shape) * span, LOWER, UPPER)
        out.append(child)
    return out


def main(
    task: str = "tiny",
    entropy: int = 20261004,
    train_episodes: int = 16,
    holdout: str = "dev",
    generations: int = 10,
    population: int = 12,
    elite: int = 3,
    sigma: float = 0.15,
    quick: bool = False,
    n_jobs: int = -1,
    seed: int = 0,
    out: str | None = None,
) -> None:
    if entropy == 0:
        raise ValueError("train on a root of your own (--entropy=...): root 0 holds the dev episodes of the check")
    out = Path(out or f"outputs/07_mpc_policy_search/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    rng = np.random.default_rng(seed)
    train = scoring.episode_set(task, train_episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
    held_out = scoring.episode_set(task, holdout, quick=quick, n_jobs=n_jobs)
    with tempfile.TemporaryDirectory(prefix="sbf-mpc-search-") as tmp:
        work = Path(tmp)

        def fitness(params: np.ndarray, name: str) -> float:
            score = train.score(str(write_candidate(params, work / name)), cpu_budget=True)
            return -float("inf") if score.rss is None else score.rss

        start = start_params()
        archive = [(fitness(start, "g0_start"), start)]
        print(f"{task}: training on {len(train.episodes)} episodes of root {entropy}")
        print(f"generation 0: current defaults, training {scoring.SCALE} {archive[0][0]:.4f}")
        for g in range(1, generations + 1):
            parents = [p for _s, p in sorted(archive, key=lambda x: -x[0])[:elite]]
            children = mutate(parents, rng, population, sigma=sigma)
            scored = [(fitness(c, f"g{g}_{i}"), c) for i, c in enumerate(children)]
            archive += scored
            best_score = max(s for s, _ in archive)
            print(f"generation {g}: best of {population} {max(s for s, _ in scored):.4f}; best so far {best_score:.4f}")
        best_score, best = max(archive, key=lambda x: x[0])
        print(f"the best candidate: training {scoring.SCALE} {best_score:.4f}")
        best_dir = write_candidate(best, work / "best")
        cmp = held_out.compare(str(best_dir), str(write_candidate(start, work / "start")), cpu_budget=True)
        cmp = replace(cmp, a=replace(cmp.a, agent="the best candidate"), b=replace(cmp.b, agent="current defaults"))
        print(f"held out, on {len(held_out.episodes)} dev episodes:\n{cmp}")
        if cmp.diff is not None and cmp.diff > 0:
            shutil.copytree(best_dir, out / "best", dirs_exist_ok=True)
            print(f"written {out / 'best'}: next, uv run sbf check {out / 'best'} --task={task}")
        else:
            print("not written: the best candidate does not beat the current defaults held out")


if __name__ == "__main__":
    fire.Fire(main)
