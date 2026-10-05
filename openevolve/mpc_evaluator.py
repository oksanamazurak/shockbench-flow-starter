"""OpenEvolve evaluator for the mpc agent's two EVOLVE-BLOCK regions (forecast_open, Agent.act's hybrid
criterion). Scores on the same tuning root used by examples/07_mpc_policy_search.py (entropy=20261004),
never the dev episodes (root 0) reserved for the final sbf compare.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TASK = "small"
EPISODES = 6
ENTROPY = 20261004
PARAMS_SOURCE = Path(__file__).resolve().parent / "params.json"


def evaluate(program_path):
    from sbf_starter.scoring import evaluate as sbf_evaluate

    # OpenEvolve writes each candidate to a NamedTemporaryFile (typically /tmp), not beside this file, so the
    # candidate's own ``HERE / "params.json"`` lookup would silently miss our calibrated numbers and fall back
    # to the hardcoded PARAMS defaults in its EVOLVE-BLOCK-free code. Drop our calibrated params.json next to
    # it so every candidate is scored holding the same calibrated numbers -- the structural search should be
    # judged with the numbers we already trust, not against a candidate re-rolling PARAMS defaults by accident.
    candidate_params = Path(program_path).parent / "params.json"
    if PARAMS_SOURCE.is_file():  # always: a stale file from an earlier run would silently win
        shutil.copy(PARAMS_SOURCE, candidate_params)

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
