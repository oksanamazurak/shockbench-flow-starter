"""Pre-generate agents/mpc2's scenario library (future disruption draws of the public generator) for one board.

    uv run python scripts/scenario_library.py --task=small --n=8

A draw costs about 1.6 s to sample, too much for a week's CPU budget, but a board's instance is the same in every
episode and the draws are deterministic (scenarios.scenario_library, tag TAG_MPC_SCEN), so they ship with the agent:
agents/mpc2/scenarios_<task>.pkl holds the instance's content digest and the draws; the agent uses them only when
the digest matches its own instance.
"""

import pickle
import time

import fire
import shockbench_flow_gym as g
from shockbench_flow_agent.shim import load_agent_class

from sbf_starter import ROOT


def main(task: str = "small", n: int = 8) -> None:
    env = g.make_env(task)
    obs, info = env.reset(seed=0)
    agent = load_agent_class(str(ROOT / "agents" / "mpc2"), "lib")(g.agent_config_from_reset(env, obs, info))
    agent.act(obs)
    pol = agent.policy
    import sys

    S_ = sys.modules["sbfv.policies.scenarios"]
    start = time.perf_counter()
    lib = S_.scenario_library(pol._inst, pol._gen, S_.TAG_MPC_SCEN, n)
    out = ROOT / "agents" / "mpc2" / f"scenarios_{task}.pkl"
    out.write_bytes(pickle.dumps({"digest": pol._inst.content_digest, "draws": lib}))
    print(f"{n} draws in {time.perf_counter() - start:.0f} s -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    fire.Fire(main)
