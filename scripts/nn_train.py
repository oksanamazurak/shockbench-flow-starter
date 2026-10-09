"""Train agents/nn_mpc's per-slot network on mpc2's flows (outputs/nn_data/<task>/, from scripts/nn_collect.py).

    uv run python scripts/nn_train.py --task=small
    uv run python scripts/nn_train.py --task=full

The target is log1p(mpc2's flow / (SCALE u0)), u0 the edge's nominal capacity, on the slots open this week;
10 % of the episodes are held out. Writes agents/nn_mpc/weights_<task>.pt (the state dict and the sizes).
"""

import sys

import fire
import numpy as np
import torch

from sbf_starter import ROOT


sys.path.insert(0, str(ROOT / "agents" / "nn_mpc"))
from features import U0_COL  # noqa: E402
from model import SCALE, SIZES, SlotNet  # noqa: E402


def _load(task):
    files = sorted((ROOT / "outputs" / "nn_data" / task).glob("*.npz"))
    if not files:
        raise SystemExit(f"no data in outputs/nn_data/{task}: run scripts/nn_collect.py first")
    eps = []
    for f in files:
        d = np.load(f)
        X, y, m = d["X"], d["flows"], d["mask"]
        u0 = np.exp(X[..., U0_COL] * 12) - 1  # features.py stores log1p(u0) / 12
        eps.append((X[m > 0], np.log1p(y / np.maximum(SCALE * u0, 1e-9))[m > 0]))
    return eps


def main(task: str = "small", epochs: int = 30, lr: float = 2e-3, batch: int = 4096, seed: int = 0) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    eps = _load(task)
    order = rng.permutation(len(eps))
    n_val = max(1, len(eps) // 10)
    val = [eps[i] for i in order[:n_val]]
    train = [eps[i] for i in order[n_val:]]
    Xt = torch.tensor(np.concatenate([x for x, _ in train]))
    yt = torch.tensor(np.concatenate([y for _, y in train]))
    Xv = torch.tensor(np.concatenate([x for x, _ in val]))
    yv = torch.tensor(np.concatenate([y for _, y in val]))
    hidden, layers = SIZES[task]
    net = SlotNet(Xt.shape[1], hidden, layers)
    net.mean.copy_(Xt.mean(0))
    net.std.copy_(Xt.std(0).clamp_min(1e-3))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"{task}: {len(train)} train / {len(val)} val episodes, {len(Xt):,} rows, {Xt.shape[1]} features, "
          f"net {hidden}x{layers} ({sum(p.numel() for p in net.parameters()):,} parameters)")
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            j = perm[i : i + batch]
            loss = torch.nn.functional.smooth_l1_loss(net(Xt[j]), yt[j], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = torch.nn.functional.smooth_l1_loss(net(Xv), yv, beta=0.1).item()
            mae = (net(Xv) - yv).abs().mean().item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train {loss.item():.4f}  val {v:.4f}  val MAE {mae:.4f} (log units)")
    out = ROOT / "agents" / "nn_mpc" / f"weights_{task}.pt"
    torch.save({"state": best_state, "n_features": Xt.shape[1], "hidden": hidden, "layers": layers}, out)
    print(f"best val {best:.4f} -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
