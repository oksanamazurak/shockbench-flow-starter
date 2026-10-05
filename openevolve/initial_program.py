"""Seed program for OpenEvolve: a copy of agents/heuristic/agent.py.

Only the EVOLVE-BLOCK region is mutated. ``Agent.__init__(config)`` and ``Agent.act(observation)`` must keep
their signatures: evaluator.py loads this file exactly as the scorer loads a submission's agent.py.
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMS = {
    "fraction": 1.0,  # share of each slot's capacity: one number, or a list with one per slot
    "closure_power": 1.0,  # a lane through a strait ships (its open fraction) ** closure_power; 0 ignores closures
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        u0 = static["edges"]["u0"]  # each edge's nominal capacity per week
        slots = static["action_slots"]
        self.cap = np.array([u0[e] for e in slots["edge"]], dtype=float) * np.asarray(PARAMS["fraction"], dtype=float)
        self.power = float(PARAMS["closure_power"])
        position = {node: i for i, node in enumerate(layout["chokepoints"])}  # strait -> its index in graph_now.open
        lane_chokepoints = static["lanes"]["chokepoints"]
        # per slot, the indices of the straits its lane passes ([] off any lane)
        self.through = [
            [position[c] for c in lane_chokepoints[lane]] if lane is not None else [] for lane in slots["lane"]
        ]

    def act(self, observation):
        # EVOLVE-BLOCK-START
        flows = self.cap * observation["action_mask"]
        open_now = observation["graph_now.open"]  # 1 open .. 0 closed
        seen = observation["graph_now.open.observed"] == 1
        for s, chokepoints in enumerate(self.through):
            for c in chokepoints:
                if seen[c]:
                    flows[s] *= max(float(open_now[c]), 0.0) ** self.power
        return {"flows": flows}
        # EVOLVE-BLOCK-END
