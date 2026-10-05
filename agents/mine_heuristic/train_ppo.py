"""Train PPO (Stable-Baselines3) for ``agents/ppo`` and export it. Needs the rl extra: ``uv sync --extra rl``.

    uv run python agents/mine_heuristic/train_ppo.py
    uv run python agents/mine_heuristic/train_ppo.py --task=small --total_timesteps=2000000 --n_envs=8 --n_scenarios=512
    uv run python agents/mine_heuristic/train_ppo.py --net_arch=[256,256] --activation=relu

This is this participant's own copy of ``examples/05_train_ppo.py``'s approach (that file is the shared starter
kit and is not edited); it writes its submission to ``agents/ppo`` instead of ``examples/ppo_agent.py``.

The observation adds a ``warning_summary`` feature (``ppo_features.summarize_warnings``, in ``agents/ppo``):
counts and nearest-week distances from ``messages``/``pending_prohibitions``/``closure_end``, the padded lists
the public example drops entirely. The same function runs at inference in ``agents/ppo/agent.py``. ``summary.json``
also reports ``holdout_mean_return``, the policy's mean return on a held-out scenario pool (``eval_entropy``,
distinct from the training root), to catch overfitting to the training pool that the training-pool learning curve
alone would hide.

The server has torch but not SB3, so the policy and its observation normaliser are exported as TorchScript
(``policy.pt``, beside ``agents/ppo/agent.py`` and ``agents/ppo/ppo_features.py``) and checked against SB3. It
fits only the network it was trained on: use ``--task=small`` for a submission.
"""

import copy
import csv
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
from gymnasium.wrappers import FilterObservation, FlattenObservation, RescaleAction
from shockbench_flow_agent import load_agent_class
from shockbench_flow_agent.submission import agent_warnings, build_submission, check_zip
from shockbench_flow_gym import agent_config_from_reset
from shockbench_flow_gym.wrappers import CapacityFractionAction, ScaleReward, ScenarioPool, SlimInfo, draw_scenarios

from sbf_starter import env_id


PPO_DIR = Path(__file__).resolve().parent.parent / "ppo"  # agents/ppo: not a sibling of this script (agents/mine_heuristic)
sys.path.insert(0, str(PPO_DIR))
from ppo_features import summarize_warnings  # noqa: E402 - needs the sys.path insert above


try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
except ModuleNotFoundError as err:
    raise SystemExit(f"{err.name} is missing: install the rl extra first, uv sync --extra rl") from None


# padded variable-length lists, left out of the flat vector except for warning_summary (about 62,000 of Tiny's
# ~62,800 numbers): messages/pending_prohibitions/closure_end are summarized by AddWarningSummary instead of dropped
DROP_PREFIXES = ("pipeline.", "queue_lots.", "wip.", "messages.", "pending_prohibitions.", "closure_end.")
ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
}
ACTION_ATOL = 1e-4  # allowed gap between the exported and the SB3 flows, as a fraction of capacity
AGENT_FILE = PPO_DIR / "agent.py"
FEATURES_FILE = PPO_DIR / "ppo_features.py"
WEIGHTS = "policy.pt"


def observation_keys(env: gym.Env) -> list[str]:
    """The fields of the flat vector, in the order FlattenObservation concatenates them (sorted Dict keys)."""
    kept = [k for k in env.observation_space.spaces if not k.startswith(DROP_PREFIXES)]
    return list(FilterObservation(env, kept).observation_space.spaces)


class LastDictObservation(gym.Wrapper):
    """Keeps the last Dict observation, for ``check_submission``."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dict = obs
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self.last_dict = out[0]
        return out


class AddWarningSummary(gym.ObservationWrapper):
    """Adds the ``warning_summary`` key (``ppo_features.summarize_warnings``) to the Dict observation.

    Placed after ``CapacityFractionAction`` and before ``FlattenObservation``/``FilterObservation``: the key is
    new, so ``observation_keys()`` (computed after this wrapper is in the stack) keeps it automatically, with no
    change to ``DROP_PREFIXES``. It must stay outside ``LastDictObservation`` (the ``check=True`` branch), which
    captures the server-shaped Dict the real ``Agent.act`` receives - that Dict never has this key; ``agent.py``
    (``agents/ppo``) computes it itself at inference, from the same ``summarize_warnings``.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        inst = env.unwrapped.instance
        self.action_edges = np.array([e for e, _k, _lane in inst.action_slots])
        self.capacity = np.array([inst.edges[e].u0 for e in self.action_edges], dtype=np.float64)
        self.horizon = int(inst.T)
        self.observation_space = gym.spaces.Dict(
            {**env.observation_space.spaces, "warning_summary": gym.spaces.Box(-np.inf, np.inf, (10,), np.float64)}
        )

    def observation(self, observation: dict) -> dict:
        week = float(np.asarray(observation["week"]).reshape(()))
        out = dict(observation)
        out["warning_summary"] = summarize_warnings(observation, self.action_edges, self.capacity, week, self.horizon)
        return out


def make_env(task: str, scenarios: list[Path], entropy: int, regime: str, *, check: bool = False):
    def thunk() -> gym.Env:
        env = ScenarioPool(gym.make(env_id(task), regime=regime), len(scenarios), entropy)
        if check:
            env = LastDictObservation(env)
        env = AddWarningSummary(ScaleReward(CapacityFractionAction(env)))
        env = FlattenObservation(FilterObservation(env, observation_keys(env)))
        n = env.action_space.shape
        env = RescaleAction(env, np.full(n, -1.0, np.float32), np.full(n, 1.0, np.float32))
        return env if check else Monitor(SlimInfo(env))  # SlimInfo: a small info dict through the subprocess pipes

    return thunk


class StopAfter(BaseCallback):
    def __init__(self, minutes: float | None) -> None:
        super().__init__()
        self.deadline = None if minutes is None else time.monotonic() + 60 * minutes

    def _on_step(self) -> bool:
        return self.deadline is None or time.monotonic() < self.deadline


class ExportedPolicy(torch.nn.Module):
    """Flat observation (float64) -> fraction of capacity in [0, 1], computed exactly as SB3 does.

    The normalisation runs in float64 like SB3's numpy code, the layers in float32 like SB3's torch code.
    """

    obs_keys: list[str]

    def __init__(self, model: PPO, venv: VecNormalize, keys: list[str]) -> None:
        super().__init__()
        policy = model.policy
        if not isinstance(getattr(policy.features_extractor, "flatten", None), torch.nn.Flatten):
            raise ValueError("the export expects the flat observation (SB3's FlattenExtractor)")
        layers = [*policy.mlp_extractor.policy_net, policy.action_net]
        self.net = torch.nn.Sequential(*[copy.deepcopy(m).cpu() for m in layers]).eval()
        space = model.action_space
        self.register_buffer("mean", torch.as_tensor(venv.obs_rms.mean, dtype=torch.float64))
        self.register_buffer("std", torch.sqrt(torch.as_tensor(venv.obs_rms.var, dtype=torch.float64) + venv.epsilon))
        self.register_buffer("low", torch.as_tensor(space.low, dtype=torch.float64))
        self.register_buffer("high", torch.as_tensor(space.high, dtype=torch.float64))
        self.clip_obs = float(venv.clip_obs)
        self.obs_keys = list(keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.clamp((x - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        a = self.net(z.to(torch.float32)).to(torch.float64)
        a = torch.minimum(torch.maximum(a, self.low), self.high)
        return torch.clamp((a - self.low) / (self.high - self.low), 0.0, 1.0)


def export(model: PPO, venv: VecNormalize, keys: list[str], path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch 2.14 marks TorchScript deprecated
        torch.jit.save(torch.jit.script(ExportedPolicy(model, venv, keys)), str(path))
    return path


def write_submission(weights: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(AGENT_FILE, folder / "agent.py")
    shutil.copy(FEATURES_FILE, folder / "ppo_features.py")
    shutil.copy(weights, folder / WEIGHTS)
    return build_submission(folder, folder.with_suffix(".zip"))


def _inner(env: gym.Env, cls: type) -> gym.Env:
    while not isinstance(env, cls):
        env = env.env
    return env


def check_submission(task, model, venv, scenarios, entropy, regime, folder: Path, episodes: int) -> float:
    """The largest gap between the exported agent's and SB3's flows, each week, on the same Dict observation."""
    agent_class = load_agent_class(folder, "agent")
    env = make_env(task, scenarios, entropy, regime, check=True)()
    keep, fraction = _inner(env, LastDictObservation), _inner(env, CapacityFractionAction)
    summarized = _inner(env, AddWarningSummary)
    keys = observation_keys(summarized)  # includes warning_summary; keep.observation_space does not (upstream of it)
    worst = 0.0
    for i in range(episodes):
        flat, info = env.reset(seed=i, options={"pool_index": i % len(scenarios)})
        agent = agent_class(agent_config_from_reset(env, keep.last_dict, info))
        done = False
        while not done:
            raw = keep.last_dict  # the server-shaped Dict: no warning_summary key, same as Agent.act gets in production
            obs = summarized.observation(raw)  # adds warning_summary, matching what FlattenObservation used for flat
            if not np.array_equal(flat, np.concatenate([np.asarray(obs[k], np.float64).ravel() for k in keys])):
                raise RuntimeError("the flat observation is not the concatenation of the fields the agent reads")
            action, _ = model.predict(venv.normalize_obs(flat[None]), deterministic=True)
            theirs = fraction.action(env.action(action[0]))["flows"]
            mine = agent.act(raw)["flows"]
            worst = max(worst, float(np.max(np.abs(mine - theirs) / fraction.capacity)))
            flat, _reward, done, _truncated, _info = env.step(action[0])
    return worst


def learning_curve(progress: Path, points: int = 12) -> list[list[float]]:
    """About ``points`` rows of [timesteps, mean episode return] from SB3's progress.csv."""
    with progress.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("rollout/ep_rew_mean")]
    keep = sorted({round(i * (len(rows) - 1) / max(1, points - 1)) for i in range(points)}) if rows else []
    return [
        [int(float(rows[i]["time/total_timesteps"])), round(float(rows[i]["rollout/ep_rew_mean"]), 4)] for i in keep
    ]


def main(
    task: str = "tiny",
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    n_scenarios: int = 64,
    entropy: int = 20260928,
    regime: str = "standard",
    check_episodes: int = 2,
    net_arch: list[int] = (64, 64),
    activation: str = "tanh",
    n_steps: int | None = None,
    batch_size: int | None = None,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    eval_entropy: int | None = None,
    eval_episodes: int = 20,
    max_minutes: float | None = None,
    torch_threads: int = 1,
    seed: int = 0,
    out: str | None = None,
) -> None:
    """Train, export and check the agents/ppo submission; write a run folder and summary.json, and update policy.pt.

    Args:
        task: tiny, small or full.
        total_timesteps: environment weeks of training (an episode is 26 on Tiny; scale up for small/full - e.g.
            2,000,000+ timesteps and 256+ n_scenarios for a small submission).
        n_envs: parallel environments (subprocesses when above 1).
        n_scenarios: training scenarios drawn from ``entropy``.
        entropy: your training root: any integer but 0 (the dev episodes).
        regime: the information regime; standard is the scored one.
        check_episodes: episodes on which the export must match the SB3 policy.
        net_arch: hidden layers of the policy and the value network.
        activation: tanh, relu, elu, leaky_relu, gelu or silu.
        n_steps: steps per environment per rollout (default: 8 episodes).
        batch_size: PPO's minibatch (default: n_steps).
        learning_rate: PPO's learning rate.
        gamma: the discount factor.
        gae_lambda: PPO's GAE lambda.
        n_epochs: PPO's number of passes over each rollout.
        clip_range: PPO's policy clip range.
        ent_coef: PPO's entropy bonus coefficient (exploration).
        eval_entropy: held-out root for the generalization check (default: ``entropy + 1``); must differ from
            ``entropy`` and from 0 (the reserved dev root).
        eval_episodes: episodes of ``eval_entropy`` the held-out check averages over.
        max_minutes: stop training after this much wall time.
        torch_threads: keep 1: more threads compete with the environment workers.
        seed: PPO's seed.
        out: the run folder (default: outputs/train_ppo_mine/<date_time>).

    """
    settings = dict(locals())
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {list(ACTIVATIONS)}, got {activation!r}")
    eval_entropy = entropy + 1 if eval_entropy is None else eval_entropy
    if eval_entropy in (entropy, 0):
        raise ValueError(f"eval_entropy must differ from entropy ({entropy}) and from 0 (the dev root)")
    settings["eval_entropy"] = eval_entropy
    out = Path(out or f"outputs/train_ppo_mine/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"run folder: {out}")
    T = gym.make(env_id(task)).unwrapped.instance.T
    n_steps = n_steps or 8 * T
    torch.set_num_threads(torch_threads)
    start = time.monotonic()
    scenarios = draw_scenarios(task, n_scenarios, entropy, n_jobs=max(1, n_envs))
    print(f"{n_scenarios} training scenarios of root {entropy}: {scenarios[0].parent}")
    thunks = [make_env(task, scenarios, entropy, regime) for _ in range(n_envs)]
    venv = VecNormalize(
        SubprocVecEnv(thunks) if n_envs > 1 else DummyVecEnv(thunks), norm_obs=True, norm_reward=False, gamma=gamma
    )
    arch = list(net_arch)
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=n_steps,
        batch_size=batch_size or n_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_epochs=n_epochs,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": {"pi": arch, "vf": arch}, "activation_fn": ACTIVATIONS[activation]},
        seed=seed,
    )
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    model.learn(total_timesteps=total_timesteps, callback=StopAfter(max_minutes))
    train_seconds = time.monotonic() - start
    venv.close()
    venv.training = False  # freeze the normaliser's statistics
    model.save(out / f"ppo_{task}.zip")
    venv.save(out / "vecnormalize.pkl")
    keys = observation_keys(AddWarningSummary(gym.make(env_id(task), regime=regime)))
    weights = export(model, venv, keys, out / WEIGHTS)
    folder = out / "submission"
    zip_path = write_submission(weights, folder)
    checked = check_zip(zip_path)
    worst = check_submission(task, model, venv, scenarios, entropy, regime, folder, check_episodes)

    eval_scenarios = draw_scenarios(task, eval_episodes, eval_entropy)
    eval_venv = VecNormalize(
        DummyVecEnv([make_env(task, eval_scenarios, eval_entropy, regime)]),
        norm_obs=True,
        norm_reward=False,
        gamma=gamma,
        training=False,
    )
    eval_venv.obs_rms = copy.deepcopy(venv.obs_rms)
    holdout_mean_return, holdout_std_return = evaluate_policy(
        model, eval_venv, n_eval_episodes=eval_episodes, deterministic=True
    )
    eval_venv.close()

    summary = {
        "settings": settings,
        "timesteps": int(model.num_timesteps),
        "train_seconds": round(train_seconds, 1),
        "observation_size": len(venv.obs_rms.mean),
        "learning_curve": learning_curve(out / "progress.csv"),
        "holdout_mean_return": round(float(holdout_mean_return), 4),
        "holdout_std_return": round(float(holdout_std_return), 4),
        "model": str(out / f"ppo_{task}.zip"),
        "submission": str(folder),
        "zip": str(zip_path),
        "zip_sha256": checked.sha256,
        "agent_warnings": agent_warnings(
            (folder / "agent.py").read_bytes(), files=[name for name, _size in checked.files]
        ),
        "max_flow_difference": worst,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    if worst > ACTION_ATOL:
        raise RuntimeError(f"the exported agent's flows differ from the SB3 policy's by {worst:.2e} of a capacity")
    shutil.copy(weights, PPO_DIR / WEIGHTS)  # the committed, canonical agents/ppo/policy.pt - only once checks pass
    print(f"updated {PPO_DIR / WEIGHTS}")
    print(f"next: uv run sbf check {PPO_DIR} --task={task}")


if __name__ == "__main__":
    fire.Fire(main)
