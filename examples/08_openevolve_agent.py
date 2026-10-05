"""OpenEvolve search with gpt-6-luna.

    export OPENAI_API_KEY=sk-...
    .venv-linux/bin/python examples/08_openevolve_agent.py

200 edits of evolve/initial_agent.py (see evolve/config.yaml). Each candidate
plays 8 Small episodes. The best file is written to
outputs/08_openevolve_agent/<time>/champion/agent.py.
"""

import json
import os
import time
from pathlib import Path

import fire

from sbf_starter import ROOT


def main(iterations: int | None = None, out: str | None = None) -> None:
    """Run the search in evolve/config.yaml and save the best agent.py.

    Pass --iterations only to override max_iterations from the config.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first")

    run_dir = Path(out or f"outputs/08_openevolve_agent/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    from openevolve import run_evolution

    result = run_evolution(
        initial_program=ROOT / "evolve" / "initial_agent.py",
        evaluator=ROOT / "evolve" / "evaluator.py",
        config=ROOT / "evolve" / "config.yaml",
        iterations=iterations,
        output_dir=str(run_dir / "openevolve"),
        cleanup=False,
    )

    champion = run_dir / "champion"
    champion.mkdir(parents=True, exist_ok=True)
    (champion / "agent.py").write_text(result.best_code or "", encoding="utf-8")
    summary = {"iterations": iterations, "best_score": result.best_score, "metrics": result.metrics}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"best score: {result.best_score:.4f}")
    print(f"champion: {champion / 'agent.py'}")


if __name__ == "__main__":
    fire.Fire(main)
