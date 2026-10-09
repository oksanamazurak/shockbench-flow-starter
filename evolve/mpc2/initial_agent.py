"""agents/mpc2 plus evolved rules: the LP plans, ``adjust`` corrects its flows before they ship.

The evaluator places this file as agent.py beside a copy of agents/mpc2 (mpc2_base.py, sbfv/, params.json).
"""

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mpc2_base import Agent as Base  # noqa: E402


class Info:
    """Per action slot facts, built once from the instance (lengths come from the config, never hard-coded).

    slot_k[i]        commodity of slot i
    slot_tail[i]     node type the route leaves ("source", "material", "terminal", "fab", "osat", "chokepoint", ...)
    slot_head[i]     node type the route enters (same names; "grid" and "sink" included)
    slot_mode[i]     "sea", "air", "pipeline", ...
    to_grid[i]       True when slot i delivers a fuel to a power grid
    grid_of[i]       grid ordinal for those slots, else -1
    fuel_days[i]     days of cover the grid keeps of that fuel (0 when not a grid fuel)
    fuel_share[i]    share of the grid's generation from that fuel (0 when not a grid fuel)
    to_fab[i]        True when slot i delivers a fab's input (wafers)
    to_sink[i]       True when slot i ends at a demand sink
    n_grids, n_fabs, T
    """

    def __init__(self, inst):
        n = len(inst.action_slots)
        self.T = int(inst.T)
        self.n_grids, self.n_fabs = len(inst.grids), len(inst.fabs)
        self.slot_k = np.zeros(n, dtype=int)
        self.slot_tail, self.slot_head, self.slot_mode = [], [], []
        self.to_grid = np.zeros(n, dtype=bool)
        self.grid_of = np.full(n, -1)
        self.fuel_days = np.zeros(n)
        self.fuel_share = np.zeros(n)
        self.to_fab = np.zeros(n, dtype=bool)
        self.to_sink = np.zeros(n, dtype=bool)
        for i, slot in enumerate(inst.action_slots):
            e, k = int(slot[0]), int(slot[1])
            edge = inst.edges[e]
            tail, head = inst.nodes[edge.tail], inst.nodes[edge.head]
            self.slot_k[i] = k
            self.slot_tail.append(tail.type)
            self.slot_head.append(head.type)
            self.slot_mode.append(edge.mode)
            if head.type == "grid":
                self.to_grid[i] = True
                self.grid_of[i] = inst.grids.index(edge.head)
                self.fuel_days[i] = float(head.grid.days_cover.get(k, 0.0))
                self.fuel_share[i] = float(head.grid.shares.get(k, 0.0))
            elif head.type == "fab" and k == head.fab.input:
                self.to_fab[i] = True
            elif head.type == "sink":
                self.to_sink[i] = True


# EVOLVE-BLOCK-START
def adjust(flows, naive, info, week, mem):
    """Correct mpc2's planned flows for this week. Returns the flows to ship (same length).

    flows: mpc2's requested quantity per action slot (already masked, >= 0)
    naive: the naive rule's quantity per action slot this week (same length)
    info:  ``Info`` above; week: 1..info.T; mem: a dict kept across weeks of the episode
    """
    return flows


# EVOLVE-BLOCK-END


class Agent(Base):
    def __init__(self, config):
        super().__init__(config)
        self.info = None
        self.mem = {}

    def act(self, observation):
        flat = super().act(observation)
        if self.info is None:
            self.info = Info(self.policy._inst)
        mask = np.asarray(observation["action_mask"], dtype=float)
        naive = self._flat(self.policy._fallback.act(self.decoder.decode(observation)))["flows"] * mask
        week = int(np.asarray(observation["week"]).reshape(-1)[0])
        out = adjust(flat["flows"].copy(), naive, self.info, week, self.mem)
        flat["flows"] = np.nan_to_num(np.maximum(np.asarray(out, dtype=float), 0.0)) * mask
        return flat
