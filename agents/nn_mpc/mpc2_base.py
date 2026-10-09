"""Rolling-horizon LP planner behind the flat Dict interface, with impairments that end.

Each week the flat observation is turned back into the protocol's lists and the planner solves window LPs with SciPy's
HiGHS; its week-1 action is turned into flows, override quantities and release modes. Where the package's mpc_det
holds every observed impairment (a closed strait, a tariff, an energy shock) for the whole window, this planner ends
each one after a quantile of its residual duration under the public generator's duration law, conditioned on its
observed age; several quantiles make a two-stage SAA sharing the week-1 action. The early signals (announced closure
ends, warnings, threats, tariff notices) adjust every window. The planner's modules ship beside this file as ``sbfv``.
"""

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sbfv import marks as M  # noqa: E402
from sbfv.omega.container import EVENT_FIELDS  # noqa: E402
from sbfv.oracle import lp as LP  # noqa: E402
from sbfv.policies import lp_common as L  # noqa: E402
from sbfv.policies import scenarios as S_  # noqa: E402
from sbfv.policies.base import StepTelemetry  # noqa: E402
from sbfv.policies.mpc_det import MpcDet  # noqa: E402
from sbfv.policies.registry import PolicyContext  # noqa: E402

WAR_RISK = ("none", "red_sea", "hormuz_2026")
TARIFF_CHANNELS = (0, 1, 2)
MID_THREAT = 5
TARGET_CHOKEPOINT, TARGET_EDGE = 0, 1

DEFAULTS = {
    "H_extra": 8,  # weeks added to the planner's window L (L+8 of the package sweep)
    "quantiles": [0.55],  # residual-duration quantiles; [] would keep impairments for the whole window
    "future_draws": 0,  # public-generator future onsets (0: only the observed impairments end)
    "demand_scale": 1.0,  # 1: trust the forecast; extra demand was hurting RSS on Small
    "throughput_scale": 1.0,  # plan for this multiple of the observed strait throughput
    "closure_end": True,  # persist a closure until its announced end week, then reopen
    "closure_hold": True,  # if True, keep the observed open fraction until the announced end
    "warn_gain": 0.0,  # warning cuts over-rerouted on Small; 0 trusts residual duration instead
    "warn_lag": 1,  # first window week the warning cut applies to
    "warn_weeks": 8,  # how many window weeks a warning cut lasts (0: the rest of the window)
    "threat_gain": 0.0,  # military-threat cuts; 0 on Small (noisy decoys)
    "threat_weeks": 12,  # how many window weeks a military-threat cut lasts
    "tariff_rate": 0.0,  # announced-tariff rate; 0 ignores noisy notices
    "tariff_proposal_scale": 0.4,  # proposals / informal notices use this fraction of tariff_rate
    "fab_energy_cap": True,  # window LP: a base_first grid's fabs get at most max(0, G_bar - y_bar) in every week
    "base_first_fix": True,  # re-solve the window until no week sheds base load to power a fab (``_solve``)
    "bf_passes": 1,  # at most this many re-solves per week
    "cpu_limit": 1.2,  # drop extra scenarios / passes when last week used more than this many seconds
    "fuel_mult": 1.0,  # ask this multiple of the planned flow on grid-fuel slots (agents/mpc_lp's lever against shed)
    "long_fuel_days": 0,  # grid fuels with at least this many days of cover get at least naive's flow (0: off)
    "bf_adapt": 0.0,  # > 0: on-week share per grid = bf_adapt * (1 - planned shed share), spread evenly
    "bf_cycle": 0,  # > 0: at a grid that sheds, fabs run every bf_cycle-th week (shed priced there), off between
}
PARAMS_FILE = HERE / "params.json"
PARAMS = {**DEFAULTS, **(json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.is_file() else {})}
# Tiny's LP is cheap and the window covers most of the episode: two residual quantiles and the early
# signals help there. They hurt on Small once CPU and decoys enter, so they apply only when T is Tiny's.
TINY_OVERRIDES = {
    "quantiles": [0.4, 0.7],
    "demand_scale": 1.05,
    "warn_gain": 0.35,
    "warn_weeks": 8,
    "threat_gain": 0.5,
    "threat_weeks": 16,
    "tariff_rate": 0.15,
    "tariff_proposal_scale": 0.4,
    "closure_hold": True,
    "base_first_fix": True,
    "bf_passes": 2,
    "cpu_limit": 1.55,
}

# Small: a grid short of fuel sheds every week, so base_first keeps its fabs off all episode. Running them in a share
# 0.6 * (1 - planned shed share) of the weeks, with shed priced there, lets the fuel build up for them (``_cycle``):
# small-val (80 episodes) +0.0038 [-0.0000, +0.0075] (0.5: +0.0034, 0.8: -0.0043, 1.0: -0.0162), dev 0.7776 -> 0.7887.
SMALL_OVERRIDES = {
    "bf_adapt": 0.6,
}

# Full's LP is big: the base-first re-solves took a median 2.3 s a week and up to 12 s, so about 12 % of Full's weeks
# went over the 4 s budget and were played by naive. Without them the median is 0.32 s and dev Full scores 0.449
# against 0.407 (20 episodes, CPU budget on); Tiny and Small keep their settings.
FULL_OVERRIDES = {
    "bf_passes": 0,
    # Full's grids shed 425 B$ an episode more than agents/mpc_lp's, which asks 10x the fuel; here full-val (40
    # episodes, sbf bench) gains +0.0051 [+0.0010, +0.0089] at 10 (+0.0030 at 1.5, +0.0036 at 3); Small: none.
    "fuel_mult": 10.0,
    # A grid's nuclear fuel holds a year of cover and arrives 8 weeks after dispatch: the window sees plenty and never
    # reorders, so on Full (104 weeks) the stock runs out near week 64 and the grid sheds to the end (grid_us: 157k
    # units an episode against naive's 17k). Naive's flow on these slots keeps the stock up.
    "long_fuel_days": 90,
}


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
        heads = static["edges"]["head"]
        lanes = static["lanes"]["edges"]
        self.entry_edge = {}
        for c, k, lane, nxt in self.lot_keys:
            into = [e for e in lanes[lane] if heads[e] == c]
            self.entry_edge[(c, k, lane, nxt)] = int(into[0]) if into else int(nxt)

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
        if self.lot_keys and np.ndim(qty) == 2:
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
        else:
            live = seen == 1
            for i in np.flatnonzero(live):
                cols["chokepoint"].append(int(obs["queue_lots.chokepoint"][i]))
                cols["k"].append(int(obs["queue_lots.k"][i]))
                cols["qty"].append(float(qty[i]))
                lane_ok = bool(obs["queue_lots.lane.observed"][i])
                cols["lane"].append(int(obs["queue_lots.lane"][i]) if lane_ok else None)
                cols["next_edge"].append(int(obs["queue_lots.next_edge"][i]))
                cols["arrival_week"].append(int(obs["queue_lots.arrival_week"][i]))
                cols["dispatch_week"].append(int(obs["queue_lots.dispatch_week"][i]))
                cols["entry_edge"].append(int(obs["queue_lots.entry_edge"][i]))
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


def _base_first_price(inst, T, go):
    """The LP builder's week-1 price on shed base load (``lp.base_first_price``): V / min e_f over the grid's fabs."""
    es = [inst.nodes[inst.fabs[fo]].fab.e for fo in inst.grid_fabs[go]]
    es = [e for e in es if e > 0]
    if not es:
        return 0.0
    V = max(
        max((d.pi * (T if d.backlog else 1) for d in inst.demands), default=0.0),
        max((st.salvage for st in inst.stock_slots), default=0.0),
    )
    return V / min(es)


def _empty_draw(gid):
    """A scenario draw without events: its future layer is the nominal network."""
    events = {f"ev_{name}": np.zeros(0, dtype=dtype) for name, dtype in EVENT_FIELDS}
    events["ev_key_ptr"] = np.zeros(1, dtype=np.int64)
    events["ev_key"] = np.zeros(0, dtype=np.int64)
    return S_.ScenarioDraw(index=-1, entropy_sha256="none", generator_id=gid, events=events)


class SignalMpc(MpcDet):
    """mpc_det whose windows end the observed impairments at residual quantiles and read the early signals."""

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
        inst = self._inst
        self._kappa0 = L.ObservedGraph.nominal(inst).values["kappa"].copy()
        self._ages = S_.ImpairmentAges()
        self._quantiles = [float(q) for q in self.p["quantiles"]]
        if self._quantiles:
            self._gen = S_.generator_params(inst, None)
            empty = _empty_draw(S_.generator_id(self._gen, inst))
            self._nominal = S_.future_marks(inst, self._gen, empty, inst.T + 1)
            n = int(self.p["future_draws"])
            self._library = S_.scenario_library(inst, self._gen, S_.TAG_MPC_SCEN, n) if n else ()
            self._residuals = {}

    def _horizon(self, inst, plan):
        return super()._horizon(inst, plan) + int(self.p["H_extra"])

    def _cpu_tight(self):
        """Drop extra scenarios when CPU is scarce. Tiny's LP is cheap: always keep the full quantile set."""
        if int(self._inst.T) <= 30:
            return False
        if not self.telemetry:
            return True
        return self.telemetry[-1].seconds > 0.55 * float(self.p["cpu_limit"])

    def act(self, obs):
        start = time.perf_counter()
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        self._ages.update(inst, obs, self._memory)
        self._t = t
        H_t = L.window_length(self._H, t, inst.T)
        qs = self._quantiles
        if qs and self._cpu_tight():
            qs = [min(qs, key=lambda q: abs(q - 0.5))]
        saved = self._quantiles
        self._quantiles = qs
        try:
            if self._quantiles:
                windows = self._quantile_windows(inst, obs, H_t)
            else:
                windows = [(L.persistence_arrays(inst, obs, self._memory, H_t), ())]
        finally:
            self._quantiles = saved
        windows = [(self._adjust(arrays, t, H_t), hits) for arrays, hits in windows]
        window = L.rolled_window(inst, obs, H_t)
        models = [L.rolled_lp(inst, obs, a, H_t, fab_hits=h, window=window, planning_rules=True) for a, h in windows]
        res = self._solve(inst, models, [a for a, _ in windows])
        if res.ok:
            x0 = res.x[: len(models[0].lb)]
            action = L.week1_action(inst, models[0], x0, obs, L.prohibited_now(self._memory, t))
        else:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, res.iterations, res.trail, fallback=not res.ok)
        )
        return action

    # ----- the base-first fix: no window week may shed base load to power a fab --------------------------------------
    def _lp(self, inst, models):
        if len(models) == 1:
            return L.to_highs_lp(models[0]), L.WindowShape.of(models[0])
        return L.saa_lp(models, L.action_columns(inst, models[0]))

    def _energy_cells(self, inst, model):
        """Per window week and base_first grid with energy-drawing fabs: (t, go, ysh column, E columns)."""
        cells = []
        for go, g in enumerate(inst.grids):
            if inst.nodes[g].grid.priority != "base_first":
                continue
            fos = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            if not fos:
                continue
            for t in range(2, model.T + 1):  # week 1 carries the package's own price already
                cells.append((t, go, model.index[("ysh", t, go)], [model.index[("E", t, fo)] for fo in fos]))
        return cells

    def _solve(self, inst, models, arrays):
        """Solve the window; with ``base_first_fix`` re-solve until no week both sheds base load and powers a fab.

        The relaxed LP may plan to shed base load at a grid and give the energy to its fabs, which a base_first grid of
        the simulator never does. Each pass prices shed at the offending (week, grid) cells (the fabs must then run on
        energy beyond the base load, found by moving fuel) and, where shed stays, turns the cells' fab energy off
        instead (the fabs stop that week, as in the simulator). A few passes approximate the exact (mixed-integer)
        rule within the CPU budget; the week-1 action is read from the last solution.
        """
        lp, shape = self._lp(inst, models)
        res = self._session.solve(lp, shape)
        if (int(self.p.get("bf_cycle", 0)) > 0 or float(self.p.get("bf_adapt", 0.0)) > 0) and res.ok:
            cyc = self._cycle(inst, models, arrays, res)
            if cyc is not None:
                return cyc
        passes = int(self.p["bf_passes"]) if self.p["base_first_fix"] else 0
        if passes and self._cpu_tight():
            passes = 1
        if not passes or not res.ok:
            return res
        n = len(models[0].lb)
        cells = self._energy_cells(inst, models[0])
        if not cells:
            return res
        price = {go: _base_first_price(inst, models[0].T, go) for go in {c[1] for c in cells}}
        z1 = [set() for _ in models]  # cells priced: shed there must go
        z0 = [set() for _ in models]  # cells whose fabs are off
        for _ in range(passes):
            changed = False
            for b, model in enumerate(models):
                x = res.x[b * n : (b + 1) * n]
                y_bar, G_bar = arrays[b]["y_bar"], arrays[b]["G_bar"]
                for i, (t, go, jsh, jE) in enumerate(cells):
                    shed = x[jsh] > 1e-6 * max(1.0, float(y_bar[t - 1, go]))
                    if i in z0[b]:
                        continue
                    if i in z1[b]:
                        if shed:  # shed could not be avoided: the fabs are off that week
                            z1[b].discard(i)
                            z0[b].add(i)
                            changed = True
                        continue
                    if shed and sum(x[j] for j in jE) > 1e-6 * max(1.0, float(G_bar[t - 1, go])):
                        z1[b].add(i)
                        changed = True
            if not changed:
                break
            fixed = []
            for b, model in enumerate(models):
                prio = model.meta.get("priority")
                prio = np.zeros(n) if prio is None else np.array(prio, dtype=float, copy=True)
                ub = np.array(model.ub, dtype=float, copy=True)
                for i in z1[b]:
                    t, go, jsh, _ = cells[i]
                    prio[jsh] = price[go]
                for i in z0[b]:
                    _, _, _, jE = cells[i]
                    ub[jE] = 0.0
                fixed.append(replace(model, ub=ub, meta={**model.meta, "priority": prio}))
            lp, _ = self._lp(inst, fixed)
            new = self._session.solve(lp, None)  # the kept basis as it is: the same week, the same shape
            self._session._shape = shape  # so that next week's solve shifts the basis as usual
            if not new.ok:
                break
            res = new
        return res

    def _cycle(self, inst, models, arrays, res):
        """Fuel is storable and shed costs per unit: a grid short of fuel can shed more in some weeks and none in the
        others, where its fabs run (base_first gives fabs energy only when no base load is shed). At every grid whose
        window plan sheds, week w runs its fabs when w % bf_cycle == 0 (shed priced as in ``_solve``) and keeps them
        off otherwise. One re-solve; None when no grid sheds or the re-solve fails.
        """
        model, P, t0 = models[0], int(self.p.get("bf_cycle", 0)), self._t
        adapt = float(self.p.get("bf_adapt", 0.0))
        n = len(model.lb)
        cells = []
        for go, g in enumerate(inst.grids):
            if inst.nodes[g].grid.priority != "base_first":
                continue
            fos = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            if fos:
                cells += [(t, go, model.index[("ysh", t, go)], [model.index[("E", t, fo)] for fo in fos])
                          for t in range(1, model.T + 1)]
        shedding = set()
        for b in range(len(models)):
            x, y_bar = res.x[b * n : (b + 1) * n], arrays[b]["y_bar"]
            shedding |= {go for t, go, jsh, _ in cells if x[jsh] > 1e-3 * max(1.0, float(y_bar[t - 1, go]))}
        if not shedding:
            return None
        price = {go: _base_first_price(inst, model.T, go) for go in shedding}
        share = {}
        if adapt > 0:
            x, y_bar = res.x[:n], arrays[0]["y_bar"]
            for go in shedding:
                shed = sum(x[jsh] for t, g, jsh, _ in cells if g == go)
                load = sum(float(y_bar[t - 1, g]) for t, g, _, _ in cells if g == go)
                share[go] = min(1.0, max(0.0, adapt * (1.0 - shed / max(load, 1e-9))))

        def on(go, w):
            if adapt > 0:
                f = share[go]
                return math.floor((w + 1) * f) > math.floor(w * f)
            return w % P == 0

        fixed = []
        for m in models:
            prio = m.meta.get("priority")
            prio = np.zeros(n) if prio is None else np.array(prio, dtype=float, copy=True)
            ub = np.array(m.ub, dtype=float, copy=True)
            for t, go, jsh, jE in cells:
                if go not in shedding:
                    continue
                if on(go, t0 + t - 1):
                    prio[jsh] = price[go]
                else:
                    ub[jE] = 0.0
            fixed.append(replace(m, ub=ub, meta={**m.meta, "priority": prio}))
        lp, shape = self._lp(inst, fixed)
        new = self._session.solve(lp, None)
        self._session._shape = shape
        return new if new.ok else None

    def _residual(self, t, el, q):
        age, seen = self._ages.age(t, el)
        key = (el.type_code, age, seen, q)
        if key not in self._residuals:
            self._residuals[key] = S_.residual_duration(self._inst, self._gen, el.type_code, q, age=age, start_seen=seen)
        return self._residuals[key]

    def _quantile_windows(self, inst, obs, H):
        """One window per quantile: ``scenarios.scenario_windows`` with each impairment's residual at that quantile."""
        T, t = inst.T, int(obs["week"])
        v = self._memory.values
        sl = slice(t - 1, t - 1 + H)
        weeks = np.arange(t, t + H)
        elements = S_.present_elements(inst, self._memory)
        demand = L.point_demand(inst, obs, H)
        pending = L.pending_mask(self._memory, t, H, v["prohibited"].shape)
        k_mu = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints], dtype=np.float64).reshape(-1, 2)
        K = len(inst.commodities)
        out = []
        for i, q in enumerate(self._quantiles):
            if self._library:
                fut = S_.future_marks(inst, self._gen, self._library[i % len(self._library)], t)
            else:
                fut = self._nominal
            u, o, supply = fut.u[sl].copy(), fut.o[sl].copy(), fut.supply[sl].copy()
            G, R, R_osat = fut.G_bar[sl].copy(), fut.R[sl].copy(), fut.R_osat[sl].copy()
            c, tariff = fut.c[sl].copy(), fut.tariff[sl].copy()
            prohibited = fut.prohibited[sl] | pending
            wr = fut.wr_class[sl].copy()
            target = {"closure": o, "capacity": u, "fab": R, "osat": R_osat, "grid": G, "supply": supply}
            for el in elements:
                r = self._residual(t, el, q)
                if el.kind in S_._CONTINUOUS:
                    sev = 1.0 - el.value / el.nominal
                    target[el.kind][:, el.element] *= M.open_fraction(T, [(t - 1.0, t - 1.0 + r, sev)])[sl]
                    continue
                on = weeks < math.ceil(t - 1.0 + r) + 1
                if el.kind == "prohibition":
                    prohibited[on, el.element // K, el.element % K] = True
                elif el.kind == "tariff":
                    tariff[on, el.element // K, el.element % K] += el.value
                elif el.kind == "war_risk":
                    wr[on, el.element] = np.maximum(wr[on, el.element], np.int8(el.value))
                else:
                    c[on, el.element] *= el.value
            hq, cwr = L.window_queue_and_transit(inst, wr)
            arrays = {
                "u": u,
                "c": c,
                "o": o,
                "kappa": k_mu[None, :, :] * o[:, :, None],
                "supply": supply,
                "G_bar": G,
                "y_bar": np.broadcast_to(v["grid_y"], (H, len(inst.grids))).copy(),
                "R": R,
                "alpha_bar": np.maximum(fut.alpha_bar[sl], v["fab_alpha"][None, :]),
                "sigma_scr": fut.sigma_scr[sl].copy(),
                "R_osat": R_osat,
                "demand": demand.copy(),
                "prohibited": prohibited,
                "tariff": tariff,
                "wr_class": wr,
                "h_queue": hq,
                "c_wr": cwr,
            }
            hits = tuple(
                M.FabHit(fab=h.fab, onset=h.onset - (t - 1), severity=h.severity)
                for h in fut.fab_hits
                if h.onset > t - 1 and t <= h.onset_week <= t + H - 1
            ) if self._library else ()
            out.append((arrays, hits))
        return out

    def _apply_cut(self, o, kappa, cut, start, length, H_t):
        """Multiply open fraction and throughput by ``cut`` on window weeks [start, start + length)."""
        if start >= H_t or length == 0 or not (cut < 1.0).any():
            return
        stop = H_t if length < 0 else min(H_t, start + max(int(length), 0))
        if stop <= start:
            return
        o[start:stop] *= cut
        kappa[start:stop] *= cut[None, :, None]

    def _adjust(self, base, t, H_t):
        """The early signals on one window's arrays (warnings, threats, closure ends, tariff notices)."""
        p, flat = self.p, self.flat
        arrays = {k: np.array(v) for k, v in base.items()}
        o, kappa = arrays["o"], arrays["kappa"]
        weeks = t + np.arange(H_t)
        lag = int(p["warn_lag"])
        n_c = o.shape[1]
        msgs = np.flatnonzero(flat["messages.msg_id.observed"] == 1) if "messages.msg_id" in flat else np.array([], int)

        if p["closure_end"] and "closure_end.chokepoint" in flat:
            live = flat["closure_end.chokepoint.observed"] == 1
            ends = flat["closure_end.end_week.observed"] == 1
            hold = bool(p.get("closure_hold", True))
            for c, end, known in zip(flat["closure_end.chokepoint"][live], flat["closure_end.end_week"][live], ends[live]):
                if not known or int(c) not in self.choke_pos:
                    continue
                j = self.choke_pos[int(c)]
                before = weeks < int(end)
                after = weeks >= int(end)
                if hold:
                    o[before, j] = o[0, j]
                    kappa[before, j] = kappa[0, j]
                o[after, j] = 1.0
                kappa[after, j] = np.maximum(kappa[after, j], self._kappa0[j])

        warn_cut = np.ones(n_c)
        if p["warn_gain"] and "warning.score" in flat:
            score = np.where(flat["warning.score.observed"] == 1, flat["warning.score"], 0.0)
            for i, j in self.warn_index:
                warn_cut[j] *= max(0.0, 1.0 - p["warn_gain"] * float(score[i]))
        self._apply_cut(o, kappa, warn_cut, lag, int(p["warn_weeks"]) or -1, H_t)

        threat_cut = np.ones(n_c)
        if p["threat_gain"]:
            for i in msgs:
                if flat["messages.channel"][i] == MID_THREAT and flat["messages.target_kind"][i] == TARGET_CHOKEPOINT:
                    j = self.choke_pos.get(int(flat["messages.target"][i]))
                    if j is not None:
                        threat_cut[j] *= max(0.0, 1.0 - p["threat_gain"])
        self._apply_cut(o, kappa, threat_cut, lag, int(p["threat_weeks"]) or -1, H_t)

        if p["throughput_scale"] != 1.0:
            kappa *= p["throughput_scale"]
        if p["demand_scale"] != 1.0:
            arrays["demand"] *= p["demand_scale"]

        if p["tariff_rate"] and len(msgs):
            tariff = arrays["tariff"]
            seen_week = flat["messages.stated_effective_week.observed"]
            kind_seen = flat["messages.kind.observed"] if "messages.kind" in flat else np.ones(len(flat["messages.channel"]))
            kinds = flat["messages.kind"] if "messages.kind" in flat else np.zeros(len(flat["messages.channel"]))
            proposal_scale = float(p.get("tariff_proposal_scale", 0.4))
            for i in msgs:
                if flat["messages.channel"][i] not in TARIFF_CHANNELS or not seen_week[i]:
                    continue
                if flat["messages.target_kind"][i] != TARGET_EDGE:
                    continue
                e = int(flat["messages.target"][i])
                start = max(int(flat["messages.stated_effective_week"][i]) - t, 0)
                if start >= H_t:
                    continue
                rate = float(p["tariff_rate"])
                ch = int(flat["messages.channel"][i])
                if ch != 2:  # not tariff_final
                    rate *= proposal_scale
                if kind_seen[i] and int(kinds[i]) == 0:  # proposal
                    rate *= proposal_scale
                ks = [int(flat["messages.k"][i])] if flat["messages.k.observed"][i] else range(tariff.shape[2])
                for k in ks:
                    tariff[start:, e, k] = np.maximum(tariff[start:, e, k], rate)
        return L.read_only(L.with_now(arrays))


class Agent:
    def __init__(self, config=None, params=None):
        if params is None:
            self.params = {**PARAMS}
            T = int((config or {}).get("T") or 52)
            if T <= 30:
                self.params.update(TINY_OVERRIDES)
            elif T > 60:
                self.params.update(FULL_OVERRIDES)
            else:
                self.params.update(SMALL_OVERRIDES)
        else:
            self.params = {**DEFAULTS, **params}
        LP.FAB_ENERGY_CAP = bool(self.params["fab_energy_cap"])
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
        self.policy = SignalMpc(self.params, config)
        self.static = static
        self.seed = int(config["policy_seed"])
        self.started = False
        # grid fuels: what a grid stocks; the environment clips an oversized request to capacity and stock
        grids = set(config["layout"]["grids"])
        fuels = {int(k) for node, k in config["layout"]["stock_slots"] if node in grids}
        self.fuel_slots = np.array([int(k) in fuels for k in static["action_slots"]["k"]], dtype=bool)
        self.long_slots = None

    def _long_slots(self, days):
        """Action slots into a grid of a fuel the grid keeps at least ``days`` of cover of."""
        inst = self.policy._inst
        slots = self.static["action_slots"]
        out = np.zeros(self.n_slots, dtype=bool)
        for i, (e, k) in enumerate(zip(slots["edge"], slots["k"])):
            head = inst.nodes[inst.edges[int(e)].head]
            grid = getattr(head, "grid", None)
            if grid is not None and float(grid.days_cover.get(int(k), 0.0)) >= days:
                out[i] = True
        return out

    def act(self, observation):
        obs = self.decoder.decode(observation)
        self.policy.flat = observation
        if not self.started:
            self.policy.reset(self.static, obs, self.seed)
            self.started = True
        action = self.policy.act(obs)
        flat = self._flat(action)
        mask = np.asarray(observation["action_mask"], dtype=float)
        seen = observation.get("action_mask.observed")
        if seen is not None and np.asarray(seen).reshape(-1)[0] == 0:
            mask = np.ones_like(flat["flows"])
        flat["flows"] = np.maximum(flat["flows"], 0.0) * mask
        days = float(self.params.get("long_fuel_days", 0))
        if days > 0:
            if self.long_slots is None:
                self.long_slots = self._long_slots(days)
            if self.long_slots.any():
                naive = self._flat(self.policy._fallback.act(obs))["flows"] * mask
                flat["flows"][self.long_slots] = np.maximum(flat["flows"], naive)[self.long_slots]
        mult = float(self.params.get("fuel_mult", 1.0))
        if mult != 1.0:
            flat["flows"][self.fuel_slots] *= mult
        return flat

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
