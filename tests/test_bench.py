"""The validation benchmark (``sbf bench``): the manifest's checks, the scores, and a tiny run against naive."""

import json

import numpy as np
import pytest

from sbf_starter import bench
from tests.conftest import ROOT, write_agent


def _manifest(tmp_path, text: str):
    path = tmp_path / "suites.yaml"
    path.write_text(text)
    return path


def test_shipped_manifest_loads():
    suites = bench.load_suites()
    assert {s.task for s in suites.values()} == {"tiny", "small", "full"}
    assert {s.split for s in suites.values()} == {"val", "stress"}


def test_dev_root_is_rejected(tmp_path):
    path = _manifest(tmp_path, "suites:\n  a: {task: tiny, split: val, roots: [0], episodes: 2}\n")
    with pytest.raises(ValueError, match="suite a: root 0"):
        bench.load_suites(path)


def test_shared_root_is_rejected(tmp_path):
    path = _manifest(
        tmp_path,
        "suites:\n"
        "  a: {task: tiny, split: val, roots: [5], episodes: 2}\n"
        "  b: {task: tiny, split: stress, roots: [5], episodes: 2, gamma: [0.95]}\n",
    )
    with pytest.raises(ValueError, match="a and b share root 5"):
        bench.load_suites(path)


def test_savings_levels_and_interval():
    assert bench.savings([100, 50], [100, 100]).tolist() == [0.0, 0.5]
    lv, cuts = bench.levels(list(range(1, 101)))
    assert np.bincount(lv)[1:].tolist() == [50, 30, 15, 5] and len(cuts) == 3
    lo, hi = bench.interval(np.zeros(10))
    assert lo == hi == 0.0


SEND_NOTHING = """
    import numpy as np

    class Agent:
        def __init__(self, config):
            self.n = config["spaces"]["action"]["flows"]["shape"][0]

        def act(self, observation):
            return {"flows": np.zeros(self.n)}
"""


@pytest.fixture
def private_bench_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "CACHE", tmp_path / "bench-cache")
    monkeypatch.setattr(bench, "FQ_REPLICATIONS", 2)  # naive's quantiles in seconds; one process (n_jobs=1)
    return tmp_path / "bench-cache"


def test_tiny_run_naive_zero_and_cache(tmp_path, private_bench_cache):
    agent = write_agent(tmp_path / "nothing", SEND_NOTHING)
    path = _manifest(tmp_path, "suites:\n  t: {task: tiny, split: val, roots: [9101], episodes: 2}\n")
    said = []
    rows = bench.run(
        [str(agent), "naive"], baseline="naive", manifest=path, n_jobs=1, out=tmp_path / "r", say=said.append
    )
    by = {r["agent"]: r for r in rows}
    assert by["naive"]["score"] == 0.0
    assert by[str(agent)]["episodes"] == 2 and "vs_baseline" in by[str(agent)]
    assert json.loads((tmp_path / "r" / "report.json").read_text()) == rows
    assert any(s.startswith("playing 4 episode") for s in said)

    said.clear()  # unchanged: read from the cache, same numbers
    again = bench.run(
        [str(agent), "naive"], baseline="naive", manifest=path, n_jobs=1, out=tmp_path / "r2", say=said.append
    )
    assert again == rows and not any(s.startswith("playing") for s in said)

    (agent / "agent.py").write_text((agent / "agent.py").read_text() + "\n# changed\n")
    said.clear()  # changed: only the agent is played again
    bench.run([str(agent), "naive"], manifest=path, n_jobs=1, out=tmp_path / "r3", say=said.append)
    assert any(s.startswith("playing 2 episode") for s in said)


def test_default_suites_never_touch_dev_or_tuning_roots():
    roots = [r for s in bench.load_suites().values() for r in s.roots]
    assert not {0, 12345, 20261002, 20261004} & set(roots)
    assert (ROOT / "benchmarks" / "suites.yaml").is_file()
