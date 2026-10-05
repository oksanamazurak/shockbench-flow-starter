"""Seed policy. OpenEvolve rewrites only the marked block.

``Signals`` turns each week's observation into arrays (closures, pending sanctions, threats, tariffs,
warnings, stock, backlog, forecast). The evolved ``act`` can use them without parsing the padded lists.
Only the standard library and numpy may be imported.
"""

import math

import numpy as np


MID_THREAT, TIES_THREAT = 5, 4
TARIFF_CHANNELS = (0, 1, 2)
TARGET_CHOKEPOINT, TARGET_EDGE = 0, 1


class Signals:
    """Network tables, read once per episode, and ``read(obs)``: this week's signals as arrays."""

    def __init__(self, config):
        static, layout = config["static"], config["layout"]
        slots, edges = static["action_slots"], static["edges"]
        self.T = int(config["T"])
        self.n_slots = len(slots["edge"])
        self.slot_edge = np.array(slots["edge"], dtype=int)
        self.slot_k = np.array(slots["k"], dtype=int)
        self.slot_lane = list(slots["lane"])
        self.cap0 = np.array([edges["u0"][e] for e in slots["edge"]], dtype=float)

        self.chokepoint_nodes = list(layout["chokepoints"])
        position = {node: i for i, node in enumerate(self.chokepoint_nodes)}
        self.n_chokepoints = len(self.chokepoint_nodes)
        lanes = static["lanes"]
        self.route_edges = [
            list(lanes["edges"][lane]) if lane is not None else [int(edge)]
            for edge, lane in zip(slots["edge"], slots["lane"])
        ]
        self.route_chokepoints = [
            [position[name] for name in lanes["chokepoints"][lane]] if lane is not None else []
            for lane in slots["lane"]
        ]
        self.edge_slots = {}
        for slot, route in enumerate(self.route_edges):
            for edge in route:
                self.edge_slots.setdefault(int(edge), []).append(slot)

        self.warning_chokepoints = [
            (i, position[node])
            for i, (kind, node) in enumerate(layout["warning_units"])
            if kind == "chokepoint"
        ]

        # Which demand a route delivers to, or -1. Indexed like flows, so it can cap them.
        heads = static["edges"]["head"]
        demand_index = {tuple(pair): i for i, pair in enumerate(layout["demands"])}
        self.demand_of_slot = np.array(
            [
                demand_index.get((int(heads[route[-1]]), int(k)), -1)
                for route, k in zip(self.route_edges, self.slot_k)
            ],
            dtype=int,
        )

    def read(self, obs):
        """This week's signals. Arrays are per action slot unless the name says strait."""
        week = int(obs["week"][0])
        open_frac = np.where(obs["graph_now.open.observed"] == 1, obs["graph_now.open"], 1.0)
        capacity = np.where(obs["graph_now.u.observed"] == 1, obs["graph_now.u"], np.nan)
        cap_now = np.array(
            [
                capacity[edge] if not math.isnan(capacity[edge]) else self.cap0[slot]
                for slot, edge in enumerate(self.slot_edge)
            ]
        )
        tariff = obs["graph_now.tariff"]
        slot_tariff = np.array(
            [max(tariff[edge, k] for edge in route) for route, k in zip(self.route_edges, self.slot_k)]
        )

        ban_in = np.full(self.n_slots, np.inf)
        live = obs["pending_prohibitions.edge.observed"] == 1
        pending = zip(
            obs["pending_prohibitions.edge"][live],
            obs["pending_prohibitions.k"][live],
            obs["pending_prohibitions.effective_week"][live],
        )
        for edge, k, effective in pending:
            for slot in self.edge_slots.get(int(edge), []):
                if self.slot_k[slot] == k:
                    ban_in[slot] = min(ban_in[slot], max(int(effective) - week, 0))

        strait_threat = np.zeros(self.n_chokepoints)
        slot_threat = np.zeros(self.n_slots)
        tariff_in = np.full(self.n_slots, np.inf)
        position = {node: i for i, node in enumerate(self.chokepoint_nodes)}
        for i in np.flatnonzero(obs["messages.msg_id.observed"] == 1):
            channel = int(obs["messages.channel"][i])
            kind = int(obs["messages.target_kind"][i])
            target = int(obs["messages.target"][i])
            if channel == MID_THREAT and kind == TARGET_CHOKEPOINT and target in position:
                strait_threat[position[target]] += 1.0
            elif channel == TIES_THREAT and kind == TARGET_EDGE:
                for slot in self.edge_slots.get(target, []):
                    slot_threat[slot] += 1.0
            elif channel in TARIFF_CHANNELS and obs["messages.stated_effective_week.observed"][i]:
                weeks = max(int(obs["messages.stated_effective_week"][i]) - week, 0)
                names_good = obs["messages.k.observed"][i] == 1
                for slot in range(self.n_slots):
                    if names_good and self.slot_k[slot] != obs["messages.k"][i]:
                        continue
                    if kind == TARGET_EDGE and target not in self.route_edges[slot]:
                        continue
                    tariff_in[slot] = min(tariff_in[slot], weeks)

        warning = np.where(obs["warning.score.observed"] == 1, obs["warning.score"], 0.0)
        strait_warning = np.zeros(self.n_chokepoints)
        for index, choke in self.warning_chokepoints:
            strait_warning[choke] = warning[index]

        def over_route(per_strait, worst, default):
            values = []
            for chokes in self.route_chokepoints:
                values.append(worst(per_strait[choke] for choke in chokes) if chokes else default)
            return np.array(values)

        # Same length as flows. inf: this route does not end at a demand sink.
        forecast = np.asarray(obs["demand_forecast.qty"], dtype=float)
        backlog = np.asarray(obs["backlog.qty"], dtype=float)
        need = np.full(self.n_slots, np.inf)
        for slot, demand in enumerate(self.demand_of_slot):
            if demand >= 0:
                need[slot] = float(forecast[demand, 0] + backlog[demand])

        return {
            "week": week,
            "weeks_left": self.T - week + 1,
            "mask": obs["action_mask"].astype(float),
            "cap_now": cap_now,
            "open_strait": open_frac,
            "open": over_route(open_frac, min, 1.0),
            "threat_strait": strait_threat,
            "threat": over_route(strait_threat, max, 0.0) + slot_threat,
            "warning_strait": strait_warning,
            "warning": over_route(strait_warning, max, 0.0),
            "ban_in": ban_in,
            "tariff": slot_tariff,
            "tariff_in": tariff_in,
            "need": need,
            "stock": obs["stock.qty"],
            "backlog": backlog,
            "forecast": forecast,
            "clip_requested": obs["last_week.clip.requested"],
            "clip_executed": obs["last_week.clip.executed"],
            "costs": obs["last_week.cost_components"],
        }


# EVOLVE-BLOCK-START
RUSH = 1.0  # a coming ban or tariff cancels the risk cut, so the route stays full
RISK = 0.35  # cut a route as warning + threat grows
HORIZON = 6  # weeks ahead that still count as soon
PREBUILD = 2.0  # extra weeks of need allowed through a route that is about to close


class Agent:
    def __init__(self, config=None):
        self.sig = Signals(config)
        self.rng = np.random.default_rng(config["policy_seed"])

    def act(self, observation):
        s = self.sig.read(observation)
        soon = np.minimum(s["ban_in"], s["tariff_in"])
        rush = np.zeros(self.sig.n_slots, dtype=float)
        live = np.isfinite(soon) & (soon < HORIZON)
        rush[live] = (HORIZON - soon[live]) / HORIZON
        risk = s["warning"] + s["threat"]
        open_frac = np.maximum(s["open"], 0.0)
        room = s["cap_now"] * s["mask"]
        # Use a route while it is still open. A live warning cuts it only when no ban or tariff is close.
        calm = np.exp(-RISK * risk * (1.0 - rush))
        flows = room * open_frac * calm

        # Routes that end at the same sink share one budget. Safer ones, and ones about to close, get more of it.
        demand = self.sig.demand_of_slot
        weight = open_frac * calm * s["mask"]
        for d in range(int(np.max(demand)) + 1):
            slots = np.flatnonzero(demand == d)
            if slots.size == 0:
                continue
            w = np.maximum(weight[slots], 0.0)
            if slots.size >= 2 and float(w.sum()) > 0.0:
                total = float(flows[slots].sum())
                flows[slots] = np.minimum(total * w / float(w.sum()), room[slots])
            target = float(s["need"][slots[0]])
            if np.isfinite(target) and target >= 0.0:
                budget = target * (1.0 + PREBUILD * float(np.max(rush[slots])))
                total = float(flows[slots].sum())
                if total > budget:
                    flows[slots] *= budget / total

        flows = np.minimum(np.maximum(flows, 0.0), room)
        return {"flows": np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)}


# EVOLVE-BLOCK-END
