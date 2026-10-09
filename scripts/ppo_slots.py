"""PPO for agents/ppo_slots: a per-slot policy that scales the naive rule's flows (log multiplier per action slot).

    uv run python scripts/ppo_slots.py --task=small --iterations=300
    uv run python scripts/ppo_slots.py --task=full --iterations=100 --episodes=16

Each iteration plays ``episodes`` new episodes (root 9601 small / 9701 full: apart from the bench suites, dev and
tuning) with sampled actions, one process each, and takes PPO steps on them: reward -cost of the week / 1e10 USD,
GAE over the weeks, the joint log-probability of a week's actions summed over its live slots (open, naive > 0).
Every ``eval_every`` iterations the mean policy plays 16 small-val search episodes (7101 n 0..15) against naive's
cached ones. The best evaluated policy is written to agents/ppo_slots/policy_<task>.pt.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fire
import numpy as np
import torch
from joblib import Parallel, delayed

from sbf_starter import ROOT, bench


AGENT = ROOT / "agents" / "ppo_slots"
sys.path.insert(0, str(AGENT))
from policy import SIZES, SlotPolicy, WeekValue  # noqa: E402


SCALE = 1e10  # USD per reward unit
ROOTS = {"small": 9601, "full": 9701}
EVAL = [(7101, n) for n in range(16)]


def _episode(path, task, root, n):
    """Play one episode with the training agent at ``path``: its policy seed and the weekly costs."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.tasks import split_label
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.omega.seeds import policy_seed
    from shockbench_flow.policies.naive_fq import fallback_spec
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, params = bench._generator(task, None)
    label = split_label(root)
    omega = sample_omega(inst, params, root, n, label)
    seed = policy_seed(root, label, n, NO_ZIP_SHA256)
    cls = _metered_shim(load_agent_class(path, f"ppo_{n}"), None)
    try:
        traj = rollout(
            inst,
            cls,
            omega,
            "standard",
            seed,
            marks=compute_marks(inst, omega),
            fallback=fallback_spec(inst, params, bench.FQ_REPLICATIONS),
        )
    finally:
        unload_agent()
    return seed, np.array([r.costs.total() for r in traj.records])


def _save(path, pol, task, n_features):
    hidden, layers = SIZES[task]
    torch.save({"state": pol.state_dict(), "n_features": n_features, "hidden": hidden, "layers": layers}, path)


def _gae(rewards, values, lam):
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + nxt - values[t]
        last = delta + lam * last
        adv[t] = last
    return adv


def _evaluate(folder, task):
    p = bench.Player("ppo", str(folder), bench._digest(folder))
    rows = Parallel(n_jobs=bench.METERED_WORKERS)(delayed(bench._job)(p, task, None, r, n, True) for r, n in EVAL)
    naive = [json.loads(bench._result_path(task, None, "naive", r, n).read_text()) for r, n in EVAL]
    return float(np.mean(bench.savings([r["J_cents"] for r in rows], [r["J_cents"] for r in naive])))


def main(
    task: str = "small",
    iterations: int = 300,
    episodes: int = 32,
    epochs: int = 4,
    lr: float = 3e-4,
    clip: float = 0.2,
    lam: float = 0.95,
    eval_every: int = 10,
    n_jobs: int = 16,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    bench._generator(task, None, n_jobs=n_jobs)
    if task == "small":  # naive's eval episodes, cached by sbf bench
        p = bench.Player(bench.NAIVE, None, bench.NAIVE)
        Parallel(n_jobs=n_jobs)(
            delayed(bench._job)(p, task, None, r, n, True)
            for r, n in EVAL
            if not bench._result_path(task, None, "naive", r, n).is_file()
        )
    hidden, layers = SIZES[task]
    sample = _feature_sample(task)
    pol = SlotPolicy(sample.shape[1], hidden, layers)
    val = WeekValue(sample.shape[1], hidden)
    for net in (pol, val):
        net.mean.copy_(torch.tensor(sample.mean(0)))
        net.std.copy_(torch.tensor(sample.std(0)).clamp_min(1e-3))
    opt = torch.optim.Adam(list(pol.parameters()) + list(val.parameters()), lr=lr)
    best = -np.inf
    log = ROOT / "outputs" / "ppo_slots" / f"{task}-{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ppo-") as tmp:
        folder = Path(tmp) / "agent"
        shutil.copytree(AGENT, folder, ignore=shutil.ignore_patterns("__pycache__", "policy_*.pt"))
        data = Path(tmp) / "data"
        data.mkdir()
        src = (folder / "agent.py").read_text().replace("TRAIN = None", f"TRAIN = {{'out': {str(data)!r}}}")
        (folder / "agent.py").write_text(src)
        weights = folder / f"policy_{task}.pt"
        for it in range(iterations):
            t0 = time.perf_counter()
            _save(weights, pol, task, pol.mean.numel())
            for f in data.glob("*.npz"):
                f.unlink()
            base = ROOTS[task]
            res = Parallel(n_jobs=n_jobs)(
                delayed(_episode)(str(folder), task, base, it * episodes + i) for i in range(episodes)
            )
            eps = []
            for s, costs in res:
                f = data / f"{s}.npz"
                if f.is_file():
                    d = np.load(f)
                    eps.append((d, -costs[: len(d["a"])] / SCALE))
            _update(pol, val, opt, eps, epochs, clip, lam)
            row = {
                "it": it,
                "seconds": round(time.perf_counter() - t0, 1),
                "episodes": len(eps),
                "J": float(np.mean([-r.sum() for _, r in eps])) * SCALE,
                "std": float(torch.exp(pol.log_std)),
            }
            if task == "small" and (it + 1) % eval_every == 0:
                _save(weights, pol, task, pol.mean.numel())
                ev = folder.parent / "eval"
                if ev.exists():
                    shutil.rmtree(ev)
                shutil.copytree(folder, ev, ignore=shutil.ignore_patterns("__pycache__"))
                (ev / "agent.py").write_text((AGENT / "agent.py").read_text())
                row["eval_saving"] = _evaluate(ev, task)
                if row["eval_saving"] > best:
                    best = row["eval_saving"]
                    _save(AGENT / f"policy_{task}.pt", pol, task, pol.mean.numel())
            if task != "small":
                _save(AGENT / f"policy_{task}.pt", pol, task, pol.mean.numel())
            print(json.dumps(row), flush=True)
            with log.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
    print(f"best eval saving {best:.4f}; log {log}")


def _feature_sample(task):
    """Features of recorded weeks (scripts/nn_collect.py's data) for the networks' input normalisation."""
    files = sorted((ROOT / "outputs" / "nn_data" / task).glob("*.npz"))[:20]
    if not files:
        raise SystemExit("run scripts/nn_collect.py once: its features set the input normalisation")
    X = np.concatenate([np.load(f)["X"].reshape(-1, np.load(f)["X"].shape[-1]) for f in files])
    return X


def _update(pol, val, opt, eps, epochs, clip, lam):
    std_old = float(torch.exp(pol.log_std).detach())
    batch = []
    with torch.no_grad():
        for d, r in eps:
            X = torch.from_numpy(d["X"])
            m = torch.from_numpy(d["mask"])
            v = val(X, m).numpy()
            adv = _gae(r, v, lam)
            ret = adv + v
            mu_old = torch.from_numpy(d["mu"])
            a = torch.from_numpy(d["a"])
            lp_old = (-0.5 * ((a - mu_old) / std_old) ** 2 - np.log(std_old)) * m
            batch.append((X, m, a, lp_old.sum(-1), torch.from_numpy(adv).float(), torch.from_numpy(ret).float()))
    X = torch.cat([b[0] for b in batch])
    M = torch.cat([b[1] for b in batch])
    A = torch.cat([b[2] for b in batch])
    LP = torch.cat([b[3] for b in batch])
    ADV = torch.cat([b[4] for b in batch])
    RET = torch.cat([b[5] for b in batch])
    ADV = (ADV - ADV.mean()) / ADV.std().clamp_min(1e-8)
    n = len(X)
    for _ in range(epochs):
        for idx in torch.randperm(n).split(256):
            mu = pol(X[idx])
            std = torch.exp(pol.log_std)
            lp = ((-0.5 * ((A[idx] - mu) / std) ** 2 - torch.log(std)) * M[idx]).sum(-1)
            ratio = torch.exp(torch.clamp(lp - LP[idx], -20, 20))
            pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx]).mean()
            vl = ((val(X[idx], M[idx]) - RET[idx]) ** 2).mean()
            loss = pg + 0.5 * vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pol.parameters()) + list(val.parameters()), 0.5)
            opt.step()


if __name__ == "__main__":
    fire.Fire(main)
