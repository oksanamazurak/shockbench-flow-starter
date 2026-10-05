"""Rolling-horizon LP planner (the package's mpc_det) behind the flat Dict interface.

Each week the flat observation is turned back into the protocol's lists, the planner solves the window LP on a
persistence forecast with SciPy's HiGHS, and its week-1 action is turned into flows, override quantities and release
modes. The planner's modules ship beside this file as ``sbfv``.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sbfv.policies import lp_common as L  # noqa: E402
from sbfv.policies.mpc_det import MpcDet  # noqa: E402
from sbfv.policies.registry import PolicyContext  # noqa: E402

WAR_RISK = ("none", "red_sea", "hormuz_2026")
TARIFF_CHANNELS = (0, 1, 2)
MID_THREAT = 5
TARGET_CHOKEPOINT, TARGET_EDGE = 0, 1

DEFAULTS = {
    "H_extra": 0,  # weeks added to the planner's window L
    "demand_scale": 1.0,  # plan for this multiple of the forecast demand
    "throughput_scale": 1.0,  # plan for this multiple of the observed strait throughput
    "closure_end": False,  # reopen a closed strait from its announced end week
    "warn_gain": 0.0,  # cut a strait's planned open fraction by gain x its warning score
    "warn_lag": 1,  # first window week the warning cut applies to
    "threat_gain": 0.0,  # cut per live military threat naming the strait
    "tariff_rate": 0.0,  # rate assumed from an announced tariff's stated week (0: ignore announcements)
}
PARAMS_FILE = HERE / "params.json"
PARAMS = {**DEFAULTS, **(json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.is_file() else {})}


def _seen(obs, key):
    return obs[f"{key}.observed"] == 1


class Decoder:
    """Flat Dict observation -> the protocol observation the planner reads (only the blocks it uses)."""

    def __init__(self, config):
        static, layout = config["static"], config["layout"]
        self.stock_slots = [tuple(s) for s in layout["stock_slots"]]
        self.supply_slots = [tuple(s) for s in layout["supply_slots"]]
        self.demands = [tuple(d) for d in layout["demands"]]
        self.chokepoints = list(layout["chokepoints"])
        self.fabs, self.grids, self.osats = layout["fabs"], layout["grids"], layout["osats"]
        self.lot_keys = [tuple(k) for k in layout.get("lot_keys", [])]
        heads, taus = static["edges"]["head"], static["edges"]["tau0"] if "tau0" in static["edges"] else None
        lanes = static["lanes"]["edges"]
        self.entry_edge = {}
        for c, k, lane, nxt in self.lot_keys:
            into = [e for e in lanes[lane] if heads[e] == c]
            self.entry_edge[(c, k, lane, nxt)] = int(into[0]) if into else int(nxt)
        self.taus = taus

    def decode(self, obs):
        week = int(obs["week"][0])
        stock = {
            "node": [n for n, _ in self.stock_slots],
            "k": [k for _, k in self.stock_slots],
            "qty": [float(q) for q in obs["stock.qty"]],
        }
        live = _seen(obs, "pipeline.qty")
        lane_seen = _seen(obs, "pipeline.lane")
        pipeline = {
            "edge": [int(x) for x in obs["pipeline.edge"][live]],
            "k": [int(x) for x in obs["pipeline.k"][live]],
            "lane": [int(x) if s else None for x, s in zip(obs["pipeline.lane"][live], lane_seen[live])],
            "qty": [float(x) for x in obs["pipeline.qty"][live]],
            "arrival_week": [int(x) for x in obs["pipeline.arrival_week"][live]],
        }
        cols = {f: [] for f in ("chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week",
                                "entry_edge")}
        qty, seen = obs["queue_lots.qty"], obs["queue_lots.qty.observed"]
        for i, j in zip(*np.nonzero(seen == 1)):
            c, k, lane, nxt = self.lot_keys[i]
            cols["chokepoint"].append(c)
            cols["k"].append(k)
            cols["qty"].append(float(qty[i, j]))
            cols["lane"].append(lane)
            cols["next_edge"].append(nxt)
            cols["arrival_week"].append(int(j) + 1)
            cols["dispatch_week"].append(int(j) + 1)
            cols["entry_edge"].append(self.entry_edge[(c, k, lane, nxt)])
        live = _seen(obs, "wip.qty")
        wip = {
            "node": [int(x) for x in obs["wip.node"][live]],
            "k": [int(x) for x in obs["wip.k"][live]],
            "qty": [float(x) for x in obs["wip.qty"][live]],
            "out_week": [int(x) for x in obs["wip.out_week"][live]],
        }
        backlog = {
            "node": [n for n, _ in self.demands],
            "k": [k for _, k in self.demands],
            "qty": [float(q) for q in obs["backlog.qty"]],
        }
        fc_seen = obs["demand_forecast.qty.observed"]
        fc = {"node": [], "k": [], "h": [], "qty": []}
        for d, h in zip(*np.nonzero(fc_seen == 1)):
            fc["node"].append(self.demands[d][0])
            fc["k"].append(self.demands[d][1])
            fc["h"].append(int(h))
            fc["qty"].append(float(obs["demand_forecast.qty"][d, h]))
        live = _seen(obs, "pending_prohibitions.edge")
        pending = {
            "edge": [int(x) for x in obs["pending_prohibitions.edge"][live]],
            "k": [int(x) for x in obs["pending_prohibitions.k"][live]],
            "effective_week": [int(x) for x in obs["pending_prohibitions.effective_week"][live]],
        }
        return {
            "week": week,
            "stock": stock,
            "pipeline": pipeline,
            "queue_lots": cols,
            "wip": wip,
            "backlog": backlog,
            "demand_forecast": fc if fc["node"] else None,
            "pending_prohibitions": pending,
            "graph_now": self._graph(obs),
        }

    def _graph(self, obs):
        if not (_seen(obs, "graph_now.open").any() or _seen(obs, "graph_now.c").any()):
            return None

        def col(key, cast=float):
            return [cast(v) if s else None for v, s in zip(obs[key], _seen(obs, key))]

        pro = np.argwhere((obs["graph_now.prohibited"] == 1) & _seen(obs, "graph_now.prohibited"))
        tar_seen = _seen(obs, "graph_now.tariff")
        tar = np.argwhere((obs["graph_now.tariff"] != 0) & tar_seen)
        wr = [WAR_RISK[int(v)] if s else None for v, s in zip(obs["graph_now.war_risk"], _seen(obs, "graph_now.war_risk"))]
        return {
            "u": col("graph_now.u"),
            "c": col("graph_now.c"),
            "open": col("graph_now.open"),
            "kappa": {"tb": col("graph_now.kappa.tb"), "ct": col("graph_now.kappa.ct")},
            "war_risk": wr,
            "supply": {
                "node": [n for n, _ in self.supply_slots],
                "k": [k for _, k in self.supply_slots],
                "avail": col("graph_now.supply.avail"),
            },
            "fab": {"node": list(self.fabs), "R": col("graph_now.fab.R"), "alpha_bar": col("graph_now.fab.alpha_bar")},
            "grid": {"node": list(self.grids), "G_bar": col("graph_now.grid.G_bar"), "y_bar": col("graph_now.grid.y_bar")},
            "osat": {"node": list(self.osats), "R": col("graph_now.osat.R")},
            "prohibited": {"edge": [int(e) for e, _ in pro], "k": [int(k) for _, k in pro]},
            "tariff": {
                "edge": [int(e) for e, _ in tar],
                "k": [int(k) for _, k in tar],
                "rate": [float(obs["graph_now.tariff"][e, k]) for e, k in tar],
            },
        }


class SignalMpc(MpcDet):
    """mpc_det whose window forecast also reads the early signals: closure ends, warnings, threats, tariffs."""

    def __init__(self, params, config):
        super().__init__(context=PolicyContext())
        self.p = params
        layout = config["layout"]
        self.choke_pos = {node: i for i, node in enumerate(layout["chokepoints"])}
        self.warn_index = [
            (i, self.choke_pos[unit]) for i, (kind, unit) in enumerate(layout["warning_units"]) if kind == "chokepoint"
        ]
        self.flat = None

    def reset(self, static, obs, policy_seed):
        super().reset(static, obs, policy_seed)
        self._kappa0 = L.ObservedGraph.nominal(self._inst).values["kappa"].copy()

    def _horizon(self, inst, plan):
        return super()._horizon(inst, plan) + int(self.p["H_extra"])

    def _window_arrays(self, inst, obs, H_t):
        base = super()._window_arrays(inst, obs, H_t)
        p, flat, t = self.p, self.flat, int(obs["week"])
        arrays = {k: np.array(v) for k, v in base.items()}
        o, kappa = arrays["o"], arrays["kappa"]
        weeks = t + np.arange(H_t)

        if p["closure_end"]:
            live = flat["closure_end.chokepoint.observed"] == 1
            ends = flat["closure_end.end_week.observed"] == 1
            for c, end, known in zip(flat["closure_end.chokepoint"][live], flat["closure_end.end_week"][live], ends[live]):
                if known and int(c) in self.choke_pos:
                    j = self.choke_pos[int(c)]
                    after = weeks >= int(end)
                    o[after, j] = 1.0
                    kappa[after, j] = np.maximum(kappa[after, j], self._kappa0[j])

        cut = np.ones(o.shape[1])
        if p["warn_gain"]:
            score = np.where(flat["warning.score.observed"] == 1, flat["warning.score"], 0.0)
            for i, j in self.warn_index:
                cut[j] *= max(0.0, 1.0 - p["warn_gain"] * float(score[i]))
        msgs = np.flatnonzero(flat["messages.msg_id.observed"] == 1)
        if p["threat_gain"]:
            for i in msgs:
                if flat["messages.channel"][i] == MID_THREAT and flat["messages.target_kind"][i] == TARGET_CHOKEPOINT:
                    j = self.choke_pos.get(int(flat["messages.target"][i]))
                    if j is not None:
                        cut[j] *= max(0.0, 1.0 - p["threat_gain"])
        lag = int(p["warn_lag"])
        if lag < H_t:
            o[lag:] *= cut
        if p["throughput_scale"] != 1.0:
            kappa *= p["throughput_scale"]
        if p["demand_scale"] != 1.0:
            arrays["demand"] *= p["demand_scale"]

        if p["tariff_rate"]:
            tariff = arrays["tariff"]
            seen_week = flat["messages.stated_effective_week.observed"]
            for i in msgs:
                if flat["messages.channel"][i] not in TARIFF_CHANNELS or not seen_week[i]:
                    continue
                if flat["messages.target_kind"][i] != TARGET_EDGE:
                    continue
                e = int(flat["messages.target"][i])
                start = max(int(flat["messages.stated_effective_week"][i]) - t, 0)
                if start >= H_t:
                    continue
                ks = [int(flat["messages.k"][i])] if flat["messages.k.observed"][i] else range(tariff.shape[2])
                for k in ks:
                    tariff[start:, e, k] = np.maximum(tariff[start:, e, k], p["tariff_rate"])
        return L.read_only(L.with_now(arrays))


class Agent:
    def __init__(self, config=None, params=None):
        self.params = PARAMS if params is None else {**DEFAULTS, **params}
        self.config = config
        static = config["static"]
        self.decoder = Decoder(config)
        self.n_slots = len(static["action_slots"]["edge"])
        ov = static["override_slots"]
        self.n_override = len(ov["chokepoint"])
        pairs = {tuple(p): i for i, p in enumerate(config["layout"]["release_pairs"])}
        self.n_pairs = len(pairs)
        self.pair_index = pairs
        self.override_pair = [pairs[(c, k)] for c, k in zip(ov["chokepoint"], ov["k"])]
        self.policy = self._make_policy()
        self.static = static
        self.seed = int(config["policy_seed"])
        self.started = False

    def _make_policy(self):
        return SignalMpc(self.params, self.config)

    def act(self, observation):
        obs = self.decoder.decode(observation)
        self.policy.flat = observation
        if not self.started:
            self.policy.reset(self.static, obs, self.seed)
            self.started = True
        action = self.policy.act(obs)
        return self._flat(action)

    def _flat(self, action):
        flows = np.zeros(self.n_slots)
        f = action.get("flows") or {"slot": [], "qty": []}
        for s, q in zip(f["slot"], f["qty"]):
            flows[int(s)] = max(float(q or 0.0), 0.0)
        override_qty = np.zeros(self.n_override)
        release = np.zeros(self.n_pairs, dtype=np.int64)
        ov = action.get("overrides")
        if ov:
            for o, q in zip(ov["slot"], ov["qty"]):
                override_qty[int(o)] = max(float(q or 0.0), 0.0)
                release[self.override_pair[int(o)]] = 1
        hold = action.get("hold")
        if hold:
            for c, k in zip(hold["chokepoint"], hold["k"]):
                release[self.pair_index[(int(c), int(k))]] = 2
        return {"flows": flows, "override_qty": override_qty, "release_mode": release}
