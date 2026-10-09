"""Collect (features, naive, mpc2 flows) per week from agents/mpc2 for training agents/nn_mpc.

    uv run python scripts/nn_collect.py --task=small --root=9101 --episodes=200
    uv run python scripts/nn_collect.py --task=full --root=9201 --episodes=60

Roots 9101 / 9201 are kept apart from the bench suites (7xxx), the dev episodes (0) and the tuning root (12345).
One file per episode in outputs/nn_data/<task>/<root>-<n>.npz: X (weeks, slots, features), naive, flows, mask.
"""

import shutil
import tempfile
from pathlib import Path

import fire
from joblib import Parallel, delayed

from sbf_starter import ROOT, bench


COLLECTOR = '''
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from features import Features  # noqa: E402
from mpc2_base import Agent as Base  # noqa: E402

OUT = Path({out!r})


class Agent(Base):
    def __init__(self, config):
        super().__init__(config)
        self.feat = Features(config)
        self.name = OUT / f"{{config['policy_seed']}}.npz"
        self.rows = {{"X": [], "naive": [], "flows": [], "mask": []}}

    def act(self, observation):
        flat = super().act(observation)
        mask = np.asarray(observation["action_mask"], dtype=float)
        naive = self._flat(self.policy._fallback.act(self.decoder.decode(observation)))["flows"] * mask
        self.rows["X"].append(self.feat.week(observation, naive))
        self.rows["naive"].append(naive.astype(np.float32))
        self.rows["flows"].append(np.asarray(flat["flows"], dtype=np.float32))
        self.rows["mask"].append(mask.astype(np.float32))
        if int(np.asarray(observation["week"]).reshape(-1)[0]) >= self.feat.T:
            np.savez_compressed(self.name, **{{k: np.stack(v) for k, v in self.rows.items()}})
        return flat
'''




def main(task: str = "small", root: int = 9101, episodes: int = 200, n_jobs: int = 12) -> None:
    out = ROOT / "outputs" / "nn_data" / task
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nn-collect-") as tmp:
        folder = Path(tmp) / "collector"
        shutil.copytree(ROOT / "agents" / "mpc2", folder, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
        (folder / "agent.py").rename(folder / "mpc2_base.py")
        shutil.copy(ROOT / "agents" / "nn_mpc" / "features.py", folder / "features.py")
        (folder / "agent.py").write_text(COLLECTOR.format(out=str(out)))
        bench._generator(task, None, n_jobs=n_jobs)
        rows = Parallel(n_jobs=n_jobs)(
            delayed(bench.play_one)(str(folder), task, None, root, n, False) for n in range(episodes)
        )
    print(f"{len(rows)} episodes -> {out} ({len(list(out.glob('*.npz')))} files)")


if __name__ == "__main__":
    fire.Fire(main)
