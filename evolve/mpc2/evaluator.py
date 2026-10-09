"""Fitness of an mpc2 + rules candidate: mean saving vs naive on 24 small-val episodes, CPU budget on.

The search episodes are small-val's first 12 of each root (7101 and 7102 n 0..11); the other 56 stay for
confirmation with ``sbf bench``. Naive's and every candidate's episodes are cached in outputs/bench-cache/, so a
candidate equal to one seen before costs nothing. A crash or a week played by naive counts as it would on the board.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from sbf_starter import ROOT, bench


BASE = ROOT / "agents" / "mpc2"
UNITS = [(root, n) for root in (7101, 7102) for n in range(12)]


def _folder(program_path, tmp):
    folder = Path(tmp) / "candidate"
    shutil.copytree(BASE, folder, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    (folder / "agent.py").rename(folder / "mpc2_base.py")
    shutil.copyfile(program_path, folder / "agent.py")
    return folder


def _rows(p):
    def one(root, n):
        path = bench._result_path("small", None, p.key, root, n)
        if path.is_file():
            return json.loads(path.read_text())
        return bench._job(p, "small", None, root, n, True)

    return Parallel(n_jobs=bench.METERED_WORKERS)(delayed(one)(r, n) for r, n in UNITS)


def evaluate(program_path):
    """Score in a fresh Python process: joblib cannot start its workers inside OpenEvolve's forked worker."""
    try:
        out = subprocess.run(
            [sys.executable, __file__, str(program_path)], capture_output=True, text=True, timeout=1200, cwd=ROOT
        )
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        print(f"evaluator: {type(exc).__name__}: {exc}")
        return {"combined_score": -1.0, "score": -1.0, "fallback_weeks": -1.0}


def _evaluate(program_path):
    try:
        with tempfile.TemporaryDirectory(prefix="sbf-evolve-") as tmp:
            folder = _folder(program_path, tmp)
            cand = bench.Player("candidate", str(folder), bench._digest(folder))
            rows = _rows(cand)
        naive = _rows(bench.Player(bench.NAIVE, None, bench.NAIVE))
        saving = bench.savings([r["J_cents"] for r in rows], [r["J_cents"] for r in naive])
        fell = sum(r["fallback_weeks"] for r in rows)
        score = float(np.mean(saving))
        return {"combined_score": score, "score": score, "fallback_weeks": float(fell)}
    except Exception as exc:  # noqa: BLE001
        print(f"evaluator: {type(exc).__name__}: {exc}")
        return {"combined_score": -1.0, "score": -1.0, "fallback_weeks": -1.0}


if __name__ == "__main__":
    print(json.dumps(_evaluate(sys.argv[1])))
