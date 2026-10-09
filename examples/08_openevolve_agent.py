"""OpenEvolve search with Claude Sonnet through Claude Code (`claude -p`, the CLI's login; no API key).

    uv run --extra evolve python examples/08_openevolve_agent.py                # evolve/: a heuristic seed
    uv run --extra evolve python examples/08_openevolve_agent.py --setup=mpc2   # evolve/mpc2/: rules over mpc2

evolve/ edits a small heuristic agent (8 Small episodes per candidate). evolve/mpc2/ edits the ``adjust`` rules that
correct agents/mpc2's planned flows (24 small-val episodes per candidate, CPU budget on). The best program is written
to outputs/08_openevolve_agent/<time>/champion/; for mpc2 the folder is a full submission (agent.py, mpc2_base.py,
sbfv/, params.json).
"""

import json
import shutil
import time
from pathlib import Path

import fire

from sbf_starter import ROOT


def main(setup: str = "heuristic", iterations: int | None = None, out: str | None = None) -> None:
    """Run the search in evolve/config.yaml (or evolve/mpc2/ with --setup=mpc2) and save the champion.

    Pass --iterations only to override max_iterations from the config.
    """
    if shutil.which("claude") is None:
        raise RuntimeError("The Claude Code CLI (`claude`) is not on PATH; log in with `claude` first")
    folder = ROOT / "evolve" / ("mpc2" if setup == "mpc2" else "")

    run_dir = Path(out or f"outputs/08_openevolve_agent/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    from openevolve import run_evolution

    result = run_evolution(
        initial_program=folder / "initial_agent.py",
        evaluator=folder / "evaluator.py",
        config=folder / "config.yaml",
        iterations=iterations,
        output_dir=str(run_dir / "openevolve"),
        cleanup=False,
    )

    champion = run_dir / "champion"
    if setup == "mpc2":
        shutil.copytree(ROOT / "agents" / "mpc2", champion, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
        (champion / "agent.py").rename(champion / "mpc2_base.py")
    champion.mkdir(parents=True, exist_ok=True)
    (champion / "agent.py").write_text(result.best_code or "", encoding="utf-8")
    summary = {"setup": setup, "iterations": iterations, "best_score": result.best_score, "metrics": result.metrics}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"best score: {result.best_score:.4f}")
    print(f"champion: {champion / 'agent.py'}")


if __name__ == "__main__":
    fire.Fire(main)
