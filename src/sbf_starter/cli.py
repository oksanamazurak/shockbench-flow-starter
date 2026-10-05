"""The ``sbf`` command line (``uv run sbf --help``). Commands print their results; the exit status says pass or fail.

An AGENT is a name (``mine`` means ``agents/mine/``), a submission folder, a zip, or (evaluate, compare) an agent.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import fire

from sbf_starter import DEFAULT_TASK, check_task, cpu_budget_s
from sbf_starter.agents import resolve


REFUSED, FAILED = 2, 1
OUT_DIR = Path("outputs")


def _say(*parts) -> None:
    print(*parts, flush=True)


def _indent(text: str) -> str:
    return "  " + text.strip().replace("\n", "\n  ")


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _path(path: str) -> Path:
    """A named agent's folder, else the path as given (a missing one is then reported as the scorer would)."""
    try:
        return resolve(path)
    except FileNotFoundError:
        return Path(path)


def _as_zip(path: Path, work: Path) -> Path:
    from shockbench_flow_agent.submission import build_submission

    if path.is_dir():
        return Path(build_submission(path, work / f"{path.resolve().name or 'submission'}.zip"))
    if path.is_file():
        return path
    raise SystemExit(f"{path}: no such folder or zip, and no agent of that name")


def _validate(zip_path: Path):
    """The scorer's own zip check; prints the verdict; None when refused."""
    from shockbench_flow_agent.submission import REFUSAL_FIXES, SubmissionError, check_zip

    try:
        sub = check_zip(zip_path)
    except SubmissionError as err:
        _say(f"REFUSED: {err}")
        if err.code in REFUSAL_FIXES:
            _say(f"  what to fix: {REFUSAL_FIXES[err.code]}")
        return None
    _say(
        f"OK: the scorer's checks pass: {len(sub.files)} file(s), {sub.zip_bytes:,} bytes zipped, "
        f"{sub.total_bytes:,} unpacked"
    )
    _say(f"  sha256 {sub.sha256} (the submission id; it salts config['policy_seed'])")
    return sub


def _static(root: Path) -> tuple[list[str], list[str]]:
    """(warnings, fatal ones). Only an import agent.py itself lacks is fatal: naive then plays every week."""
    from shockbench_flow_agent.submission import agent_warnings

    from sbf_starter.check import import_warnings

    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    own = [w for w in agent_warnings((root / "agent.py").read_bytes(), files) if not w.startswith("agent.py imports")]
    imports = import_warnings(root)
    return own + imports, [w for w in imports if w.startswith("agent.py imports")]


def check(
    path: str,
    task: str = DEFAULT_TASK,
    episodes: int = 1,
    timing: bool = True,
    docker: bool = False,
    docker_episodes: int = 1,
) -> None:
    """Check a submission as the scorer does, then time it in a process holding only the server's packages.

    Args:
        path: an agent's name, a submission folder or a zip.
        task: tiny, small (the public board's) or full (the private board's).
        episodes: dev episodes of the timed run.
        timing: run the timed episodes (False: the static checks only).
        docker: also play in a local copy of the scoring container, with its CPU meter (needs Docker).
        docker_episodes: episodes of the container run.

    Exit status: 0 all passed, 2 the scorer would refuse the zip, 1 the agent would not run there.

    """
    from shockbench_flow_agent.submission import extract_submission

    from sbf_starter import check as checks

    check_task(task)
    budget = cpu_budget_s(task)
    with tempfile.TemporaryDirectory(prefix="sbf-check-") as tmp:
        work = Path(tmp)
        zip_path = _as_zip(_path(path), work)
        _say(f"1. the scorer's checks of {zip_path.name}, and every file's imports (nothing is imported)")
        if _validate(zip_path) is None:
            sys.exit(REFUSED)
        root = extract_submission(zip_path, work / "sub").root
        warnings, fatal = _static(root)
        for w in warnings:
            _say(f"WARNING: {w}")
        ok = not (fatal and not timing)
        if timing:
            _say(
                f"2. a timed run in an isolated process (only the submission and the scoring image's packages): "
                f"{episodes} dev episode(s) of {task}, CPU seconds per week (budget {budget:g} s)"
            )
            rows = checks.timed_run(root, task, episodes)
            if rows and rows[0]["missing"]:
                _say(f"  NOTE: not installed here, so not importable in this run: {rows[0]['missing']} (--extra rl)")
            for r in rows:
                if not r["imported"]:
                    _say("FAILED: agent.py does not import with the scoring image's packages: on the server naive")
                    _say("  plays every week and the score is 0. The agent's error:")
                    _say(_indent(checks.last_error(r["stderr"])))
                    ok = False
                    break
                s = checks.summary(r)
                over = f", over the budget in weeks {s['over']}" if s["over"] else ""
                _say(
                    f"  episode {r['episode']}: week 1 (Agent(config) + first act) {s['week1_s']:.3f} s, median act "
                    f"{s['median_s']:.4f} s, max {s['max_s']:.3f} s{over}"
                )
                if s["over"]:
                    _say(
                        "  WARNING: over the budget on this machine; the scorer meters its own (a Linux x86_64 server, "
                        "one CPU): check with --docker, and keep a margin"
                    )
                if r["substitutions"]:
                    ok = False
                    subs = r["substitutions"]
                    _say(f"FAILED: {len(subs)} of {r['weeks']} weeks would be played by naive: {subs[:10]}")
                    _say("  (action: an exception or a malformed action; timeout: no reply in 10 s). The error:")
                    _say(_indent(checks.last_error(r["stderr"])))
        if docker and ok:
            from sbf_starter import container

            _say(f"3. the scoring container, {docker_episodes} dev episode(s) of {container.docker_task(task)}")
            rows = container.play(root, container.docker_task(task), docker_episodes, echo=lambda s: _say("  " + s))
            for r in rows:
                subs, s = r["substitutions"], checks.summary(r)
                _say(
                    f"  episode {r['episode']} ({r['task']}, {r['weeks']} weeks): ready in {r['ready_s'] or 0:.1f} s, "
                    f"week 1 {s['week1_s']:.3f} s CPU, median {s['median_s']:.3f} s, max {s['max_s']:.3f} s "
                    f"(budget {r['cpu_budget_s']:g} s); {len(subs)} week(s) the scorer would give to naive"
                    + (f": {subs[:10]}" if subs else "")
                )
                if r["meter_errors"]:
                    _say(f"  NOTE: the CPU meter could not read the container: {r['meter_errors'][:2]}")
                ok = ok and not subs
        if not ok:
            _say("a check failed: see above")
            sys.exit(FAILED)
        _say(f"all checks passed; next: uv run sbf upload {path}")


def pack(folder: str, out: str | None = None, compress: bool = False) -> str:
    """Zip a submission folder with repeatable bytes, then check the zip.

    Args:
        folder: an agent's name or a submission folder.
        out: the zip to write (default: outputs/<name>.zip).
        compress: deflate the files (for large weights).

    """
    from shockbench_flow_agent.submission import build_submission

    src = _path(folder)
    if not (src / "agent.py").is_file():
        raise SystemExit(f"{src}/agent.py does not exist: the zip needs agent.py at its root")
    dest = Path(out) if out else OUT_DIR / f"{src.resolve().name}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    build_submission(src, dest, compress=compress)
    _say(f"written {dest}")
    if _validate(dest) is None:
        sys.exit(REFUSED)
    _say(f"next: uv run sbf upload {_shown(dest)} (or upload it on the competition page)")
    return str(dest)


def _scored(result, out: str | None) -> None:
    import json

    from sbf_starter.scoring import as_dict

    _say(str(result))  # a quick result's report says so itself
    if out:
        Path(out).write_text(json.dumps(as_dict(result), indent=1) + "\n")
        _say(f"written {out}")


def evaluate(
    path: str,
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool = False,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """The local score (0 = naive rule, 1 = clairvoyant plan) with a 90 % interval.

    The first run on a network computes the references and caches them: a minute or two on tiny, longer on small and
    full.

    Args:
        path: an agent's name, a submission folder or zip, or an agent.py (played in this process: a debugger works).
        task: tiny, small (the public board's) or full (the private board's).
        episodes: dev (20 episodes, 5 per harm level), a count k (episodes 0..k-1) or a list.
        quick: seconds, not the leaderboard's numbers (a rough naive rule, no harm levels, 4 episodes).
        entropy: 0 for the public dev episodes; any other integer for scenarios of your own.
        cpu_budget: a week over the task's CPU budget is played by the naive rule, as on the server.
        n_jobs: workers of a first run's reference computation (-1: all cores).
        out: also write the result as JSON there.

    """
    from sbf_starter.scoring import evaluate as score

    _scored(score(path, task, episodes, quick=quick, entropy=entropy, cpu_budget=cpu_budget, n_jobs=n_jobs), out)


def compare(
    a: str,
    b: str,
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool = False,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """A's score minus B's on the same episodes, with a paired 90 % interval (the other arguments are evaluate's).

    When the interval holds 0, these episodes cannot tell the two apart.
    """
    from sbf_starter.scoring import compare as cmp

    result = cmp(a, b, task, episodes, quick=quick, entropy=entropy, cpu_budget=cpu_budget, n_jobs=n_jobs)
    _scored(result, out)


def bench(
    *agents: str,
    baseline: str | None = None,
    suites: str | list[str] | None = None,
    cpu_budget: bool = True,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """Agents against the naive rule on the validation benchmark (benchmarks/suites.yaml), not the leaderboard's RSS.

    Each episode scores the saving (J_naive - J) / J_naive; val suites add four levels by naive's cost, stress suites
    play harsher disruption rungs. Results are cached per episode under outputs/bench-cache/: a rerun plays only the
    agents that changed. The dev episodes (root 0) stay for sbf evaluate / sbf compare.

    Args:
        agents: names, folders, zips or agent.py files; "naive" is the naive rule itself.
        baseline: the agent the others are compared with (a paired 90 % interval per suite).
        suites: comma-separated suite names (default: every suite of the manifest).
        cpu_budget: a week over the task's CPU budget is played by the naive rule, as on the server.
        n_jobs: joblib workers (-1: all cores).
        out: the report's folder (default outputs/bench/<date_time>/).

    """
    from sbf_starter.bench import run

    if not agents:
        raise SystemExit("name at least one agent: uv run sbf bench mpc_lp heuristic --baseline=heuristic")
    if isinstance(suites, str):
        suites = [s for s in suites.split(",") if s]
    elif isinstance(suites, tuple):
        suites = list(suites)
    run(list(agents), baseline=baseline, suites=suites, cpu_budget=cpu_budget, n_jobs=n_jobs, out=out, say=_say)


def token(competition: str | None = None) -> None:
    """Get your Codabench API token from your username and password, and save it as CODABENCH_TOKEN in .env.

    Codabench's pages do not show the token. The password is not echoed, printed or stored. An account made with
    "Sign in with GitHub" needs a password first (Codabench's password reset).

    Args:
        competition: the competition's URL, which names the server (default: CODABENCH_COMPETITION).

    """
    import getpass

    from dotenv import find_dotenv, set_key

    from sbf_starter import ROOT, codabench

    base = codabench.server_url(competition or os.environ.get("CODABENCH_COMPETITION"))
    _say(f"your account on {base} (the password is not shown, not printed and not stored)")
    username = input("username or email: ").strip()
    password = getpass.getpass("password: ")
    try:
        value = codabench.get_token(base, username, password)
    except codabench.CodabenchError as err:
        _say(f"ERROR: {err}")
        sys.exit(FAILED)
    path = Path(find_dotenv(usecwd=True) or ROOT / ".env")
    if not path.exists():
        example = ROOT / ".env.example"
        path.write_text(example.read_text() if example.is_file() else "")
        path.chmod(0o600)
    set_key(path, "CODABENCH_TOKEN", value, quote_mode="auto")
    _say(f"CODABENCH_TOKEN saved in {_shown(path)} (gitignored)")
    if not (competition or os.environ.get("CODABENCH_COMPETITION")):
        _say(f"next: set CODABENCH_COMPETITION in {_shown(path)} to the competition's URL")
    _say("next: uv run sbf upload <agent> --dry_run")


def upload(
    agent: str,
    competition: str | None = None,
    phase: str | None = None,
    dry_run: bool = False,
    wait: bool = False,
    poll_s: float = 30.0,
) -> None:
    """Pack an agent and submit it to the competition. Each upload spends one of 3 daily submissions.

    Only the scorer's static checks run here: run ``sbf check`` first.

    Args:
        agent: an agent's name, a submission folder (packed to outputs/<name>.zip) or a zip.
        competition: the competition's URL (default: CODABENCH_COMPETITION).
        phase: the phase's name (default: the one open now).
        dry_run: check the zip, token, registration, phase and daily slots; upload nothing.
        wait: wait for the score and print it.
        poll_s: seconds between two status reads while waiting (at least 10).

    """
    from shockbench_flow_agent.submission import build_submission, extract_submission

    from sbf_starter import codabench

    path = _path(agent)
    if path.is_dir():
        zip_path = OUT_DIR / f"{path.resolve().name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        build_submission(path, zip_path)
    elif path.is_file() and path.suffix == ".zip":
        zip_path = path
    else:
        raise SystemExit(f"{path}: not an agent's name, a submission folder or a zip")
    _say(f"1. the scorer's checks of {_shown(zip_path)}")
    if _validate(zip_path) is None:
        sys.exit(REFUSED)
    with tempfile.TemporaryDirectory(prefix="sbf-upload-") as tmp:
        _warnings, fatal = _static(extract_submission(zip_path, Path(tmp)).root)
    if fatal:
        for w in fatal:
            _say(f"FAILED: {w}")
        _say(f"  nothing uploaded; see uv run sbf check {agent}")
        sys.exit(FAILED)
    try:
        client, pk, secret = codabench.from_env(competition)
        _say(f"2. Codabench at {client.base_url}, competition {pk}")
        comp, ph = codabench.preflight(client, pk, secret, phase)
        used = ph.get("used_submissions_per_day")
        limit = ph.get("max_submissions_per_day")
        _say(f"  phase {ph.get('name')!r} (id {ph['id']}, {ph.get('status')}); submissions today: {used} of {limit}")
        if dry_run:
            _say("dry run: nothing uploaded")
            return
        _say("3. uploading")
        sub = codabench.upload(client, zip_path, comp, ph)
        _say(f"  submission {sub.get('id')} created, status {sub.get('status')}")
        if wait:
            done = codabench.wait(client, sub["id"], poll_s, echo=lambda s: _say("  " + s))
            _print_submission(done)
        else:
            _say(f"follow it: uv run sbf status {sub.get('id')} --wait")
    except codabench.CodabenchError as err:
        _say(f"ERROR: {err}")
        sys.exit(FAILED)


def _print_submission(sub: dict) -> None:
    from sbf_starter import codabench

    sc = codabench.scores(sub)
    shown = ", ".join(f"{k} {v:.4f}" if "rss" in k else f"{k} {v:g}" for k, v in sc.items()) or "no score"
    when = sub.get("created_when") or sub.get("submitted_at") or ""
    _say(f"  {sub.get('id')}: {sub.get('status')} {when} {sub.get('filename') or ''}  {shown}".rstrip())
    if sub.get("status") == "Failed" and sub.get("status_details"):
        _say(f"    {sub['status_details']}")


def status(
    submission: int | None = None,
    competition: str | None = None,
    phase: str | None = None,
    wait: bool = False,
    poll_s: float = 30.0,
) -> None:
    """One submission's status and score, or all of yours in the phase.

    Args:
        submission: a submission id (upload prints it).
        competition: the competition's URL (default: CODABENCH_COMPETITION).
        phase: the phase's name, for the list (default: the one open now).
        wait: follow the submission until it is Finished, Failed or Cancelled.
        poll_s: seconds between two reads while waiting (at least 10).

    """
    from sbf_starter import codabench

    try:
        client, pk, secret = codabench.from_env(competition)
        if submission is not None:
            sub = codabench.wait(client, int(submission), poll_s) if wait else client.submission(int(submission))
            _print_submission(sub)
            return
        ph = codabench.pick_phase(client.competition(pk, secret), phase)
        subs = client.submissions(ph["id"])
        _say(f"phase {ph.get('name')!r}: {len(subs)} submission(s)")
        for s in subs:
            _print_submission(s)
    except codabench.CodabenchError as err:
        _say(f"ERROR: {err}")
        sys.exit(FAILED)


COMMANDS = {
    "evaluate": evaluate,
    "compare": compare,
    "check": check,
    "pack": pack,
    "bench": bench,
    "token": token,
    "upload": upload,
    "status": status,
}


def main() -> None:
    """The ``sbf`` console script: reads ``.env`` (from the working directory up), then runs Fire."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=True)
    fire.Fire(COMMANDS, name="sbf", serialize=lambda result: None)  # commands print their own results


if __name__ == "__main__":
    main()
