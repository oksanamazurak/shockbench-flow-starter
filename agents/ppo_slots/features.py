"""Per action slot features for the nn_mpc network: numpy only, every size read from the config.

``Features(config)`` precomputes what does not change in an episode (the slot's route, its endpoints, its commodity);
``Features.week(obs, naive)`` returns an (n_slots, N_FEATURES) float32 matrix for this week. The network is shared by
every slot, so the same weights would run on any board; Small and Full still get their own weights (``agent.py``).
"""

import numpy as np


NODE_TYPES = ("source", "material", "terminal", "chokepoint", "fab", "osat", "grid", "sink")
MODES = ("sea", "air", "pipeline", "rail", "road")
U0_COL = 2 * len(NODE_TYPES) + len(MODES) + 3  # the column of log1p(u0) / 12 (after c0, v, tau0)


def _get(obs, key, default=0.0):
    """A field with its unobserved entries set to ``default`` (fields without an .observed mask are taken as is)."""
    v = np.asarray(obs[key], dtype=np.float64)
    seen = obs.get(f"{key}.observed")
    if seen is not None:
        seen = np.asarray(seen).astype(bool)
        if seen.shape == v.shape:
            v = np.where(seen, v, default)
        elif seen.size == 1 and not seen.reshape(-1)[0]:
            v = np.full_like(v, default)
    return np.nan_to_num(v, nan=default, posinf=default, neginf=default)


class Features:
    def __init__(self, config):
        st, lay = config["static"], config["layout"]
        self.T = int(config["T"])
        slots = st["action_slots"]
        self.edge = np.asarray(slots["edge"], dtype=int)
        self.k = np.asarray(slots["k"], dtype=int)
        lane = slots["lane"]
        self.lane = np.array([-1 if x is None else int(x) for x in lane])
        n = len(self.edge)
        edges, nodes = st["edges"], st["nodes"]
        tail = np.asarray(edges["tail"], dtype=int)[self.edge]
        head = np.asarray(edges["head"], dtype=int)[self.edge]
        self.tail, self.head = tail, head
        ntype = list(nodes["type"])
        mode = [edges["mode"][e] for e in self.edge]
        c0 = np.asarray(edges["c0"], dtype=float)[self.edge]
        self.u0 = np.maximum(np.asarray(edges["u0"], dtype=float)[self.edge], 1e-6)
        tau0 = np.asarray(edges["tau0"], dtype=float)[self.edge]
        v = np.asarray(st["commodities"]["v"], dtype=float)[self.k]
        pool_tb = np.array([st["commodities"]["pool"][k] == "tb" for k in self.k], dtype=float)
        static = [
            np.array([[ntype[t] == x for x in NODE_TYPES] for t in tail], dtype=float),
            np.array([[ntype[h] == x for x in NODE_TYPES] for h in head], dtype=float),
            np.array([[m == x for x in MODES] for m in mode], dtype=float),
            np.log1p(c0)[:, None] / 10,
            np.log1p(v)[:, None] / 12,
            tau0[:, None] / 10,
            np.log1p(self.u0)[:, None] / 12,
            pool_tb[:, None],
            (self.lane >= 0).astype(float)[:, None],
        ]
        self.static = np.concatenate(static, axis=1)
        # stock slots at the route's ends, with their storage
        ss = [tuple(map(int, s)) for s in lay["stock_slots"]]
        index = {s: i for i, s in enumerate(ss)}
        self.tail_slot = np.array([index.get((int(t), int(k)), -1) for t, k in zip(tail, self.k)])
        self.head_slot = np.array([index.get((int(h), int(k)), -1) for h, k in zip(head, self.k)])
        inst_nodes = st["instance"]["nodes"]
        kid = list(st["commodities"]["id"])
        storage = np.ones(len(ss))
        for i, (node, k) in enumerate(ss):
            s = (inst_nodes[node].get("stock") or {}).get(kid[k]) or {}
            storage[i] = max(float(s.get("storage") or 0.0), 1.0)
        self.storage = storage
        # chokepoints on the slot's lane, as positions in layout["chokepoints"]
        chk = list(map(int, lay["chokepoints"]))
        pos = {c: i for i, c in enumerate(chk)}
        lanes = st["lanes"]["chokepoints"]
        self.lane_chk = [[pos[c] for c in lanes[ln] if c in pos] if ln >= 0 else [] for ln in self.lane]
        warn = lay["warning_units"]
        self.warn_chk = {pos[int(u)]: i for i, (kind, u) in enumerate(warn) if kind == "chokepoint" and int(u) in pos}
        # demand rows of a sink head, grid / fab ordinals of the head
        dem = [tuple(map(int, d)) for d in lay["demands"]]
        pairs = [(int(h), int(k)) for h, k in zip(head, self.k)]
        self.dem_of = np.array([dem.index(p) if p in dem else -1 for p in pairs])
        grids = list(map(int, lay["grids"]))
        self.grid_of = np.array([grids.index(int(h)) if int(h) in grids else -1 for h in head])
        fabs = list(map(int, lay["fabs"]))
        self.fab_of = np.array([fabs.index(int(h)) if int(h) in fabs else -1 for h in head])
        self.n = n

    def week(self, obs, naive):
        n, T = self.n, self.T
        w = float(np.asarray(obs["week"]).reshape(-1)[0])
        u = _get(obs, "graph_now.u", np.nan)
        u = np.where(np.isnan(u), 0.0, u)[self.edge] if u.size else np.zeros(n)
        prohib = _get(obs, "graph_now.prohibited")[self.edge, self.k]
        tariff = _get(obs, "graph_now.tariff")[self.edge, self.k]
        open_ = _get(obs, "graph_now.open", 1.0)
        war = _get(obs, "graph_now.war_risk")
        warn = _get(obs, "warning.score")
        lane_open, lane_war, lane_warn = np.ones(n), np.zeros(n), np.zeros(n)
        for i, cs in enumerate(self.lane_chk):
            if cs:
                lane_open[i] = min(open_[c] for c in cs)
                lane_war[i] = max(war[c] for c in cs)
                lane_warn[i] = max((warn[self.warn_chk[c]] for c in cs if c in self.warn_chk), default=0.0)
        # announced prohibitions on (edge, k): weeks until effective (capped)
        pe, pk, pw = (_get(obs, f"pending_prohibitions.{f}", -1) for f in ("edge", "k", "effective_week"))
        soon = np.full(n, 99.0)
        key = {(int(e), int(k)): i for i, (e, k) in enumerate(zip(self.edge, self.k))}
        for e, k, ew in zip(pe, pk, pw):
            if e >= 0 and ew > 0:
                i = key.get((int(e), int(k)))
                if i is not None:
                    soon[i] = min(soon[i], ew - w)
        stock = _get(obs, "stock.qty")
        ti, hi = np.maximum(self.tail_slot, 0), np.maximum(self.head_slot, 0)
        st_tail = np.where(self.tail_slot >= 0, stock[ti] / self.storage[ti], 0.0)
        st_head = np.where(self.head_slot >= 0, stock[hi] / self.storage[hi], 0.0)
        fc = _get(obs, "demand_forecast.qty")
        backlog = _get(obs, "backlog.qty")
        need = np.zeros(n)
        has = self.dem_of >= 0
        if fc.size:
            need[has] = (fc[self.dem_of[has], :2].sum(axis=1) + backlog[self.dem_of[has]]) / self.u0[has]
        G, y = _get(obs, "graph_now.grid.G_bar"), _get(obs, "graph_now.grid.y_bar")
        shed = _get(obs, "last_week.shed.qty")
        gh = self.grid_of >= 0
        grid_gap, grid_shed = np.zeros(n), np.zeros(n)
        if G.size:
            grid_gap[gh] = (G[self.grid_of[gh]] - y[self.grid_of[gh]]) / np.maximum(y[self.grid_of[gh]], 1.0)
            grid_shed[gh] = shed[self.grid_of[gh]] / np.maximum(y[self.grid_of[gh]], 1.0)
        R = _get(obs, "graph_now.fab.R", 1.0)
        fh = self.fab_of >= 0
        fab_r = np.ones(n)
        if R.size:
            fab_r[fh] = R[self.fab_of[fh]]
        req, exe = _get(obs, "last_week.clip.requested"), _get(obs, "last_week.clip.executed")
        clip = np.where(req > 1e-9, exe / np.maximum(req, 1e-9), 1.0) if req.size == n else np.ones(n)
        mask = np.asarray(obs["action_mask"], dtype=float).reshape(-1)
        dyn = np.stack(
            [
                np.full(n, w / T),
                np.full(n, (T - w) / T),
                np.full(n, min(T - w, 8) / 8),
                u / self.u0,
                prohib,
                np.minimum(tariff, 1.0),
                lane_open,
                lane_war,
                lane_warn,
                np.clip(soon, -1, 20) / 20,
                np.minimum(st_tail, 5.0),
                np.minimum(st_head, 5.0),
                np.minimum(need, 20.0),
                np.clip(grid_gap, -1, 1),
                np.minimum(grid_shed, 1.0),
                fab_r,
                np.clip(clip, 0, 2),
                mask,
                np.minimum(naive / self.u0, 5.0),
                (naive > 0).astype(float),
            ],
            axis=1,
        )
        return np.concatenate([self.static, dyn], axis=1).astype(np.float32)
