"""The validation benchmark behind ``sbf bench``: agents against the naive rule on fixed suites of episodes.

A suite (``benchmarks/suites.yaml``) is a network, a split (``val`` or ``stress``), scenario roots, a number of
episodes per root and one or more intensity rungs ``gamma``. Every episode is played by the naive rule and by each
agent on the same scenario; an agent's score on an episode is its saving ``(J_naive - J) / J_naive`` (naive 0, a plan
with no cost 1). No clairvoyant plan is solved, so these are not the leaderboard's numbers: they rank candidates.

- ``val`` suites sort episodes into four levels by naive's cost (its 50/80/95 % quantiles in the suite), report each
  level, and resample within levels for the interval.
- ``stress`` suites play harsher rungs and count all episodes alike.
- Per-episode results are cached under ``outputs/bench-cache/`` by package version, network, rung, agent content,
  root and episode: a rerun plays only agents that changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sbf_starter import ROOT, check_task


MANIFEST = ROOT / "benchmarks" / "suites.yaml"
CACHE = ROOT / "outputs" / "bench-cache"
NAIVE = "naive"
LEVEL_QUANTILES = (0.50, 0.80, 0.95)  # naive-cost cut points of the four val levels (the board's 50/30/15/5 % mix)
BOOTSTRAP, CONFIDENCE, SEED = 2000, 0.90, 0
FQ_REPLICATIONS = 1000  # the leaderboard's naive demand-model replications
# With the CPU budget on, every worker shares the machine: 24 workers on this 16-core hybrid CPU made mpc2 lose 3687 of
# 4160 Full weeks to naive, against 240 of 2080 alone (sbf evaluate). A third of the logical cores keeps the meter
# honest (8 workers: 165 of 1040, close to its 12 % alone).
METERED_WORKERS = max(1, (os.cpu_count() or 3) // 3)


# ----- manifest ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Suite:
    name: str
    task: str
    split: str
    roots: tuple[int, ...]
    episodes: int
    gammas: tuple[float | None, ...]

    def units(self) -> list[tuple[float | None, int, int]]:
        """(gamma, root, episode) of every episode the suite plays."""
        return [(g, r, n) for g in self.gammas for r in self.roots for n in range(self.episodes)]


def load_suites(path: str | Path = MANIFEST) -> dict[str, Suite]:
    """The manifest's suites, checked: known task and split, no reserved root (0 is dev), no root in two suites."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    reserved = set(data.get("reserved_roots", [])) | {0}
    suites, owner = {}, {}
    for name, s in data["suites"].items():
        suite = Suite(
            name,
            check_task(s["task"]),
            s["split"],
            tuple(int(r) for r in s["roots"]),
            int(s["episodes"]),
            tuple(None if g is None else float(g) for g in s.get("gamma", [None])),
        )
        if suite.split not in ("val", "stress"):
            raise ValueError(f"suite {name}: split must be val or stress, got {suite.split!r}")
        for root in suite.roots:
            if root in reserved:
                raise ValueError(f"suite {name}: root {root} is reserved (0 is the dev root; others were tuned on)")
            if root in owner:
                raise ValueError(f"suites {owner[root]} and {name} share root {root}")
            owner[root] = name
        suites[name] = suite
    return suites


# ----- agents --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Player:
    """An agent to play: its label, what ``load_agent_class`` loads (None for naive) and its content key."""

    label: str
    path: str | None
    key: str


def _digest(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(p.relative_to(folder).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def player(agent: str) -> Player:
    """``naive``, an agent's name, a folder, a zip (extracted under the cache) or an agent.py (its folder is hashed)."""
    from sbf_starter.agents import resolve

    if agent == NAIVE:
        return Player(NAIVE, None, NAIVE)
    path = resolve(agent)
    if path.is_file() and path.suffix == ".zip":
        key = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        folder = CACHE / "_agents" / key
        if not (folder / "agent.py").is_file():
            with zipfile.ZipFile(path) as z:
                z.extractall(folder)
        return Player(agent, str(folder), key)
    folder = path.parent if path.is_file() else path
    return Player(agent, str(path.resolve()), _digest(folder))


# ----- playing -------------------------------------------------------------------------------------------------------
def _version() -> str:
    import shockbench_flow

    return shockbench_flow.__version__


def _result_path(task: str, gamma: float | None, key: str, root: int, n: int) -> Path:
    rung = "board" if gamma is None else f"g{gamma:g}"
    return CACHE / _version() / f"{task}-{rung}" / key / f"{root}-{n}.json"


def _generator(task: str, gamma: float | None, n_jobs: int = 1):
    """The task's instance and generator at ``gamma``, with naive's quantiles from the shared disk cache."""
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import task_generator

    inst, params = task_generator(task, gamma)
    fq_quantiles(inst, params, FQ_REPLICATIONS, n_jobs=n_jobs, cache_dir=default_cache_dir())
    return inst, params


def play_one(path: str | None, task: str, gamma: float | None, root: int, n: int, cpu_budget: bool) -> dict:
    """One episode by naive (``path`` None) or by the agent at ``path``, metered at the task's budget when asked."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.tasks import split_label
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.omega.seeds import policy_seed
    from shockbench_flow.policies.naive_fq import anchor_policy, fallback_spec
    from shockbench_flow_agent.local_eval import ANCHOR_REGIME, NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import CPU_BUDGET_S, _metered_shim
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, params = _generator(task, gamma)
    label = split_label(root)
    omega = sample_omega(inst, params, root, n, label)
    marks = compute_marks(inst, omega)
    fallback = fallback_spec(inst, params, FQ_REPLICATIONS)
    seed = policy_seed(root, label, n, NO_ZIP_SHA256)
    start = time.perf_counter()
    if path is None:
        policy, regime = anchor_policy(inst, params, FQ_REPLICATIONS), ANCHOR_REGIME
        traj = rollout(inst, policy, omega, regime, seed, marks=marks, fallback=fallback)
        fell, error = 0, None
    else:
        shim = _metered_shim(
            load_agent_class(path, f"bench_{Path(path).stem}"), CPU_BUDGET_S[task] if cpu_budget else None
        )
        try:
            traj = rollout(inst, shim, omega, "standard", seed, marks=marks, fallback=fallback)
        finally:
            unload_agent()
        fell = sum(took_fallback(r) for r in traj.records)
        error = f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None
    return {
        "J_cents": int(traj.J_cents),
        "fallback_weeks": int(fell),
        "first_error": error,
        "weeks": int(inst.T),
        "seconds": round(time.perf_counter() - start, 3),
        "omega_hash": omega.hash,
    }


def _job(p: Player, task: str, gamma: float | None, root: int, n: int, cpu_budget: bool) -> dict:
    row = play_one(p.path, task, gamma, root, n, cpu_budget)
    out = _result_path(task, gamma, p.key, root, n)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row) + "\n")
    tmp.replace(out)  # an interrupted run keeps every finished episode
    return row


def results(players: list[Player], suites: list[Suite], *, cpu_budget: bool = True, n_jobs: int = -1, say=print):
    """{(agent key, suite, gamma, root, n): row}, playing what the cache lacks."""
    from joblib import Parallel, delayed

    found, todo = {}, []
    for s in suites:
        for g, r, n in s.units():
            for p in players:
                path = _result_path(s.task, g, p.key, r, n)
                if path.is_file():
                    found[(p.key, s.name, g, r, n)] = json.loads(path.read_text())
                else:
                    todo.append((p, s, g, r, n))
    if todo:
        for task, g in sorted({(s.task, g) for _p, s, g, _r, _n in todo}, key=str):
            _generator(task, g, n_jobs=n_jobs)  # naive's quantiles once, before workers read them from disk
        say(f"playing {len(todo)} episode(s) ({len(found)} cached) ...")
        t = time.perf_counter()
        rows = Parallel(n_jobs=n_jobs)(delayed(_job)(p, s.task, g, r, n, cpu_budget) for p, s, g, r, n in todo)
        for (p, s, g, r, n), row in zip(todo, rows):
            found[(p.key, s.name, g, r, n)] = row
        say(f"  done in {time.perf_counter() - t:.0f} s")
    return found


# ----- scores --------------------------------------------------------------------------------------------------------
def savings(agent: list[int], naive: list[int]) -> np.ndarray:
    """Per-episode ``(J_naive - J) / J_naive``."""
    a, b = np.asarray(agent, float), np.asarray(naive, float)
    return (b - a) / b


def levels(naive: list[int]) -> tuple[np.ndarray, list[float]]:
    """Level 1..4 of each episode by naive's cost, and the cut points (the suite's own quantiles)."""
    cuts = [float(c) for c in np.quantile(np.asarray(naive, float), LEVEL_QUANTILES)]
    return np.searchsorted(cuts, np.asarray(naive, float), side="right") + 1, cuts


def interval(diff: np.ndarray, groups: np.ndarray | None = None) -> tuple[float, float]:
    """Percentile bootstrap of the mean of ``diff``, resampled within ``groups`` when given."""
    rng = np.random.default_rng(SEED)
    groups = np.ones(len(diff), int) if groups is None else groups
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    means = np.empty(BOOTSTRAP)
    for b in range(BOOTSTRAP):
        means[b] = np.concatenate([diff[rng.choice(i, len(i))] for i in idx]).mean()
    tail = (1 - CONFIDENCE) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def table(players: list[Player], suites: list[Suite], found: dict, baseline: Player | None) -> list[dict]:
    """One row per (suite, gamma, agent): score, per level, paired difference with the baseline, fallback weeks."""
    rows = []
    for s in suites:
        for g in s.gammas:
            keys = [(r, n) for gg, r, n in s.units() if gg == g]
            naive = [found[(NAIVE, s.name, g, r, n)]["J_cents"] for r, n in keys]
            lv, cuts = levels(naive) if s.split == "val" else (None, None)
            base = None
            if baseline is not None:
                base = savings([found[(baseline.key, s.name, g, r, n)]["J_cents"] for r, n in keys], naive)
            for p in players:
                got = [found[(p.key, s.name, g, r, n)] for r, n in keys]
                sv = savings([x["J_cents"] for x in got], naive)
                row = {
                    "suite": s.name,
                    "task": s.task,
                    "split": s.split,
                    "gamma": g,
                    "agent": p.label,
                    "episodes": len(keys),
                    "score": float(sv.mean()),
                    "fallback_weeks": int(sum(x["fallback_weeks"] for x in got)),
                    "episodes_with_fallback": int(sum(x["fallback_weeks"] > 0 for x in got)),
                    "first_error": next((x["first_error"] for x in got if x["first_error"]), None),
                    "seconds_per_episode": float(np.mean([x["seconds"] for x in got])),
                }
                if lv is not None:
                    row["by_level"] = {str(k): float(sv[lv == k].mean()) for k in np.unique(lv)}
                    row["level_counts"] = {str(k): int((lv == k).sum()) for k in np.unique(lv)}
                    row["level_cuts_usd"] = [c / 100 for c in cuts]
                if base is not None and p.key != baseline.key:
                    diff = sv - base
                    lo, hi = interval(diff, lv)
                    row["vs_baseline"] = {
                        "baseline": baseline.label,
                        "diff": float(diff.mean()),
                        "low": lo,
                        "high": hi,
                        "distinguishable": not (lo <= 0 <= hi),
                    }
                rows.append(row)
    return rows


def markdown(rows: list[dict], cpu_budget: bool) -> str:
    head = (
        "Saving vs the naive rule, (J_naive - J) / J_naive per episode, averaged (0 = naive). Not the leaderboard's "
        f"RSS: no clairvoyant plan. CPU budget {'on' if cpu_budget else 'off'}; 90 % paired bootstrap intervals.\n"
    )
    lines = [
        head,
        "| suite | gamma | agent | n | score | by level (1..4) | vs baseline [90 %] | fallback weeks (episodes) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lv = " / ".join(f"{v:+.3f}" for v in r.get("by_level", {}).values()) or "-"
        vs = "-"
        if "vs_baseline" in r:
            v = r["vs_baseline"]
            mark = "" if v["distinguishable"] else " ~"
            vs = f"{v['diff']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}]{mark}"
        g = "0.62" if r["gamma"] is None else f"{r['gamma']:g}"
        fb = f"{r['fallback_weeks']} ({r['episodes_with_fallback']})"
        lines.append(
            f"| {r['suite']} | {g} | {r['agent']} | {r['episodes']} | {r['score']:+.4f} | {lv} | {vs} | {fb} |"
        )
    lines.append("\n`~`: the interval holds 0, these episodes cannot tell the agent from the baseline.")
    return "\n".join(lines) + "\n"


def run(
    agents: list[str],
    *,
    baseline: str | None = None,
    suites: list[str] | None = None,
    manifest: str | Path = MANIFEST,
    cpu_budget: bool = True,
    n_jobs: int = -1,
    out: str | Path | None = None,
    say=print,
) -> list[dict]:
    """Benchmark ``agents`` (and ``baseline``) on ``suites`` (all of the manifest by default); writes the report."""
    from sbf_starter.check import import_warnings

    known = load_suites(manifest)
    chosen = list(known) if not suites else suites
    unknown = [s for s in chosen if s not in known]
    if unknown:
        raise ValueError(f"unknown suite(s) {unknown}; the manifest has {list(known)}")
    names = list(dict.fromkeys([*agents, *([baseline] if baseline else [])]))
    players = [player(a) for a in names]
    for p in players:
        if p.path is not None:
            folder = Path(p.path) if Path(p.path).is_dir() else Path(p.path).parent
            for w in import_warnings(folder):
                say(f"warning ({p.label}): {w}")
    base = next((p for p in players if p.label == baseline), None)
    every = [*([] if any(p.key == NAIVE for p in players) else [player(NAIVE)]), *players]
    picked = [known[s] for s in chosen]
    if cpu_budget and n_jobs == -1:
        n_jobs = METERED_WORKERS
        say(f"CPU budget on: {n_jobs} workers, so that the agents do not slow each other down (--n_jobs to change)")
    found = results(every, picked, cpu_budget=cpu_budget, n_jobs=n_jobs, say=say)
    rows = table(players, picked, found, base)
    text = markdown(rows, cpu_budget)
    out = Path(out or ROOT / "outputs" / "bench" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / "report.md").write_text(text)
    say(text)
    say(f"written {out / 'report.md'} and report.json")
    return rows
