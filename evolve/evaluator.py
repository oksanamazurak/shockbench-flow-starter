"""Fitness: eight full Small episodes. Higher is better. A crash scores -1."""

import shutil
import tempfile
from pathlib import Path

from sbf_starter import scoring


def evaluate(program_path):
    """Mean score on 8 Small episodes of root 1. Not the public dev episodes."""
    try:
        with tempfile.TemporaryDirectory(prefix="sbf-evolve-") as tmp:
            folder = Path(tmp) / "candidate"
            folder.mkdir()
            shutil.copyfile(program_path, folder / "agent.py")
            result = scoring.evaluate(
                str(folder),
                task="small",
                episodes=8,
                entropy=1,
                n_jobs=1,
                verbose=False,
            )
        score = -1.0 if result.rss is None else float(result.rss)
    except Exception:
        score = -1.0
    return {"combined_score": score, "score": score}
