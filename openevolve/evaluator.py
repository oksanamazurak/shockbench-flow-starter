"""OpenEvolve evaluator: scores a candidate agent.py with ShockBench-Flow.

Runs on the tuning root (``entropy=12345``), never the dev episodes (root 0, ``entropy=0``) that AGENTS.md
reserves for confirming the final result — a search that only sees the dev episodes fits them.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TASK = "tiny"
EPISODES = 8
ENTROPY = 12345


def evaluate(program_path):
    from sbf_starter.scoring import evaluate as sbf_evaluate

    try:
        result = sbf_evaluate(
            program_path,
            task=TASK,
            episodes=EPISODES,
            entropy=ENTROPY,
            cpu_budget=False,
            n_jobs=1,
            verbose=False,
        )
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}

    return {"combined_score": float(result.rss), "rss": float(result.rss)}
