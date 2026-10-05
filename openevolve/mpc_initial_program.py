"""Seed program for OpenEvolve: agents/mpc_lp/agent.py, with three EVOLVE-BLOCK regions (forecast, hybrid, fuel).

Only these two regions are mutated: ``forecast_open``'s probability combination (how warning.score and
messages.* turn into a forecast open fraction) and ``Agent.act``'s hybrid criterion (which slots the LP
decides versus which fall back to the plain heuristic rule). ``build_lp``'s index arithmetic is NOT a
mutation target -- an LLM diff there is far more likely to silently corrupt the LP than to improve it.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent

PARAMS = {
    "H": 6,  # rolling horizon, in weeks
    "closure_power": 1.0,  # a lane through a forecast-derated chokepoint ships (open fraction) ** this
    "warn_a": 1.0,  # sigmoid slope on warning.score
    "warn_b": 2.0,  # sigmoid offset: p_warn = sigmoid(warn_a * score - warn_b)
    "msg_weight": {"0": 0.0, "1": 0.3, "2": 0.15, "3": 0.5, "4": 0.0},  # message kind -> closure weight
    "msg_bump": 0.05,  # flat weekly risk bump when a live message names no stated_effective_week
    "msg_bump_weeks": 3,  # how many weeks ahead the flat bump applies
    "holding_scale": 1.0,
    "lp_scope": "all_but_grid",
    # terminal value: what a unit still in the chain at the horizon's end is worth, as a share of the commodity's
    # published value v_k -- without it a wafer shipped now (fab 8 weeks + OSAT 2 + lane, past H) is pure cost
    "terminal_scale": 0.25,
    # model grid power in the LP (burn of each fuel, served base load, shed at VOLL). Off: shed is ~60% of the cost
    # on small, but letting the LP ship fuel did worse than the heuristic's "order the nominal, let the environment
    # clip it" on every variant tried (small train rss 0.53-0.60 vs 0.61 off; tiny ~equal), so fuel stays heuristic
    "model_grids": False,
    # reroute queued tanker cargo (lng, crude) at a strait around a closed strait further down its lanes, through
    # override slots whose remaining route is open (release_mode 1); off: the default release
    # grid fuel slots ask for this many times the heuristic's nominal request; the environment clips to capacity
    # and stock. The heuristic's request (u0) is what binds on small -- fuel piles up at sources while grids shed
    # (98% of shed is fuel-short weeks): dev small 0.558 -> 0.610 at 10; tiny loses 0.02 at 10, so per network
    "fuel_mult": {"tiny": 1.0, "small": 10.0, "full": 10.0},
    "reroute": False,
    "reroute_open": 0.5,
    "reroute_forecast": False,  # judge straits ahead by the forecast (True) or this week's observed openness  # a strait counts as open at or above this (forecast) open fraction
    # LP-decided upstream slots ship at least this share of the heuristic, per network (static["instance"]["kind"]):
    # on tiny the LP alone starves the chain ahead of shocks (train rss 0.50 at 0, 0.78 at 0.8); on small the
    # LP alone is best (0.61 at 0, falling to 0.56 at 0.7). A number applies to every network.
    "upstream_floor": {"tiny": 0.8, "small": 0.0, "full": 0.0},  # which slots the LP decides: "all_but_grid" or "demand" (Stage 1's rule)  # multiplier on static["instance"]["nodes"]'s real per-(node, k) holding_cost
    # expected tariff increase per live announcement: measured on a training root (24 tiny episodes, entropy
    # 20261004), every announcement targets a region; the import tariff rose in ~45% of cases by ~0.2, so
    # ~0.085 in expectation, alike for all three channels (formal 0.079, informal 0.088, final 0.082)
    "tariff_bump": 0.085,
    "tariff_weight": {"0": 1.0, "1": 1.0, "2": 1.0},  # channel (tariff_formal, informal, final) -> weight
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())

CLOSURE_CHANNELS = (3, 4, 5)  # sanction_legal, ties_threat, mid_threat (tariff channels excluded: Stage 1)
MESSAGE_TARGET_CHOKEPOINT = 0
TARIFF_CHANNELS = (0, 1, 2)  # tariff_formal, tariff_informal, tariff_final
TARGET_EDGE, TARGET_NODE, TARGET_REGION = 1, 2, 3


def build_topology(static: dict, layout: dict) -> dict:
    """Static, per-episode network shape the LP needs every week. Computed once, in ``Agent.__init__``."""
    edges = static["edges"]
    lanes = static["lanes"]
    action_slots = static["action_slots"]
    n_slots = len(action_slots["edge"])

    slot_edges, slot_dest, slot_tail, slot_chokepoints = [], [], [], []
    for s in range(n_slots):
        lane = action_slots["lane"][s]
        chain = list(lanes["edges"][lane]) if lane is not None else [action_slots["edge"][s]]
        slot_edges.append(chain)
        slot_dest.append(edges["head"][chain[-1]])
        slot_tail.append(edges["tail"][chain[0]])
        slot_chokepoints.append(list(lanes["chokepoints"][lane]) if lane is not None else [])
    slot_k = list(action_slots["k"])

    chokepoint_pos = {node: i for i, node in enumerate(layout["chokepoints"])}
    chokepoint_warn_row = {}
    for node in layout["chokepoints"]:
        for row, unit in enumerate(layout["warning_units"]):
            if unit[0] == "chokepoint" and unit[1] == node:
                chokepoint_warn_row[node] = row
                break

    stock_slots = [tuple(p) for p in layout["stock_slots"]]
    stock_index = {p: i for i, p in enumerate(stock_slots)}
    supply_list = [tuple(p) for p in layout["supply_slots"]]
    supply_set = set(supply_list)

    demands = [
        {
            "node": static["sinks"]["node"][i],
            "k": static["sinks"]["k"][i],
            "pi": static["sinks"]["pi"][i],
            "backlog": static["sinks"]["backlog"][i],
        }
        for i in range(len(static["sinks"]["node"]))
    ]
    demand_index = {(d["node"], d["k"]): i for i, d in enumerate(demands)}
    demand_set = set(demand_index)

    outgoing: dict = {}
    incoming: dict = {}
    for s in range(n_slots):
        outgoing.setdefault((slot_tail[s], slot_k[s]), []).append(s)
        incoming.setdefault((slot_dest[s], slot_k[s]), []).append(s)

    # Stage 2: static["instance"] (public, in config["static"]) publishes each node's holding cost and storage
    # cap per commodity, each commodity's disposal cost, and each fab/OSAT's input, output and processing lead
    # time. Production gets its own LP variables (one per fab, one per OSAT package; an OSAT's packages share
    # its throughput), so the LP sees a reward for feeding the chain, not just for the last leg to a sink.
    inst = static["instance"]
    commodity_index = {name: i for i, name in enumerate(static["commodities"]["id"])}
    disposal_cost = [0.0] * len(commodity_index)
    for c in inst["commodities"]:
        disposal_cost[commodity_index[c["id"]]] = float(c.get("disposal_cost", 0.0))
    holding_cost, storage = {}, {}
    for node_idx, nd in enumerate(inst["nodes"]):
        for cname, info in nd.get("stock", {}).items():
            q = (node_idx, commodity_index[cname])
            holding_cost[q] = float(info.get("holding_cost", 0.0))
            storage[q] = info.get("storage")
    procs, cap_groups = [], []  # a process: node, k_in, k_out, tau, group; a group: ("fab" | "osat", row)
    for row, node in enumerate(layout["fabs"]):
        nd = inst["nodes"][node]["fab"]
        cap_groups.append(("fab", row))
        procs.append({"node": node, "k_in": commodity_index[nd["input"]], "k_out": commodity_index[nd["product"]],
                      "tau": int(nd["tau"]), "group": len(cap_groups) - 1})
    for row, node in enumerate(layout["osats"]):
        nd = inst["nodes"][node]["osat"]
        cap_groups.append(("osat", row))
        for k_in_name, k_out_name in nd["packages"].items():
            procs.append({"node": node, "k_in": commodity_index[k_in_name], "k_out": commodity_index[k_out_name],
                          "tau": int(nd["tau"]), "group": len(cap_groups) - 1})
    proc_in, proc_out = {}, {}
    for j, pr in enumerate(procs):
        proc_in.setdefault((pr["node"], pr["k_in"]), []).append(j)
        proc_out.setdefault((pr["node"], pr["k_out"]), []).append(j)
    supply_nodes = {node for node, _k in supply_list}
    chokepoint_nodes = set(layout["chokepoints"])
    grid_set = set(layout["grids"])
    # a grid's fuels (what it stocks: lng on tiny; lng, crude, nucfuel on small) -- their consumption, power, is
    # not modeled, so the LP prices them at nothing wherever they travel; on small they reach a grid through a
    # terminal, so "the lane ends at a grid" does not catch them, the commodity does
    grid_fuels = {k for (node, k) in stock_slots if node in grid_set}
    # grids (static["instance"]: shares per fuel, the unmodelled share, the rationed fuel, VOLL); the simulator
    # runs every grid base_first: fuel k gives up to share_k G-bar (and no more than its stock), the unmodelled
    # share always; base load y-bar is served first, the rest is shed at VOLL
    grids = []
    for row, node in enumerate(layout["grids"]):
        gd = inst["nodes"][node]["grid"]
        fuels = [(commodity_index[name], float(sh)) for name, sh in gd["shares"].items() if name != "unmodelled"]
        grids.append({
            "node": node, "row": row, "fuels": fuels,
            "share_null": float(gd["shares"].get("unmodelled", 0.0)),
            "voll": float(gd["voll"]),
            "rationed": commodity_index.get(gd.get("rationed")) if gd.get("rationed") else None,
            "ibar": {commodity_index[name]: float(v) for name, v in gd.get("ibar", {}).items()},
        })
    psi = float(inst.get("params", {}).get("psi", 0.0))

    tracked, cumulative_cap = [], []
    for p in stock_slots:
        if p in supply_set or p in incoming or p in proc_out or p in proc_in:
            tracked.append(p)
        elif p in outgoing:
            cumulative_cap.append(p)

    return {
        "n_slots": n_slots,
        "slot_edges": slot_edges,
        "slot_dest": slot_dest,
        "slot_tail": slot_tail,
        "slot_chokepoints": slot_chokepoints,
        "slot_k": slot_k,
        "chokepoint_pos": chokepoint_pos,
        "chokepoint_warn_row": chokepoint_warn_row,
        "stock_index": stock_index,
        "supply_list": supply_list,
        "supply_set": supply_set,
        "demands": demands,
        "demand_index": demand_index,
        "demand_set": demand_set,
        "outgoing": outgoing,
        "incoming": incoming,
        "tracked": tracked,
        "cumulative_cap": cumulative_cap,
        "edges_head": list(edges["head"]),
        "edges_tail": list(edges["tail"]),
        "holding_cost": holding_cost,
        "storage": storage,
        "disposal_cost": disposal_cost,
        "procs": procs,
        "cap_groups": cap_groups,
        "proc_in": proc_in,
        "proc_out": proc_out,
        "supply_nodes": supply_nodes,
        "chokepoint_nodes": chokepoint_nodes,
        "grid_set": grid_set,
        "grid_fuels": grid_fuels,
        "grids": grids,
        "psi": psi,
        "node_region": list(static["nodes"]["region"]),
    }


def forecast_open(obs: dict, topology: dict, H: int, params: dict) -> dict:
    """Forecast open fraction of every chokepoint, ``h = 0..H-1``; ``h=0`` is the observed value exactly."""
    msg_weight = {int(k): float(v) for k, v in params["msg_weight"].items()}
    a, b = float(params["warn_a"]), float(params["warn_b"])
    bump, bump_weeks = float(params["msg_bump"]), int(params["msg_bump_weeks"])
    week = int(obs["week"][0])

    msg_live = obs["messages.msg_id.observed"].astype(bool) & (obs["messages.target_kind"] == MESSAGE_TARGET_CHOKEPOINT)
    msg_live &= np.isin(obs["messages.channel"], CLOSURE_CHANNELS)
    msg_target = obs["messages.target"]
    msg_kind = obs["messages.kind"]
    msg_eff_week = obs["messages.stated_effective_week"]
    msg_eff_observed = obs["messages.stated_effective_week.observed"].astype(bool)

    out = {}
    for node, pos in topology["chokepoint_pos"].items():
        open_now = float(obs["graph_now.open"][pos])
        row = topology["chokepoint_warn_row"].get(node)
        score = float(obs["warning.score"][row]) if row is not None else 0.0

        mine = msg_live & (msg_target == node)
        msg_kinds_here = msg_kind[mine]
        msg_eff_week_here = msg_eff_week[mine]
        msg_eff_observed_here = msg_eff_observed[mine]

        # EVOLVE-BLOCK-START
        # Combine the warning score and this chokepoint's live messages into a closure-probability forecast
        # for each of the next H weeks. ``p_warn`` is a single scalar (the sigmoid baseline risk); ``p_msg``
        # is per-week, built from messages with (weight keyed by kind) or without (a flat few-week bump) a
        # stated effective week. The result must be a length-H array of probabilities in [0, 1].
        p_warn = 1.0 / (1.0 + np.exp(-(a * score - b)))
        p_msg = np.zeros(H, dtype=float)
        for kind, eff_week, eff_observed in zip(msg_kinds_here, msg_eff_week_here, msg_eff_observed_here):
            weight = msg_weight.get(int(kind), 0.0)
            if weight <= 0.0:
                continue
            if eff_observed:
                h = int(eff_week) - week
                if 0 <= h < H:
                    p_msg[h] = min(1.0, p_msg[h] + weight)
            else:
                for h in range(min(bump_weeks, H)):
                    p_msg[h] = min(1.0, p_msg[h] + bump * weight)
        # noisy-OR: avoids additive double-counting/saturation when both warning score and message
        # risk are high at the same horizon; stays in [0, 1] without needing an explicit clip.
        p_close = 1.0 - (1.0 - p_warn) * (1.0 - p_msg)
        # EVOLVE-BLOCK-END

        forecast = open_now * (1.0 - p_close)
        forecast[0] = open_now
        out[node] = forecast
    return out


def tariff_forecast(obs: dict, topology: dict, H: int, params: dict) -> np.ndarray:
    """Tariff rate per (edge, k, h): this week's persisted, plus ``tariff_bump * weight[channel]`` from the week a
    live tariff announcement says it takes effect, on the edges it names (an edge; edges into or out of a node;
    edges into a region). The announced rate itself is not in the observation, hence one tunable bump.
    """
    now = np.asarray(obs["graph_now.tariff"], dtype=float)
    out = np.repeat(now[:, :, None], H, axis=2)
    bump = float(params.get("tariff_bump", 0.0))
    if bump == 0.0:
        return out
    weights = {int(c): float(w) for c, w in params.get("tariff_weight", {}).items()}
    week = int(obs["week"][0])
    live = obs["messages.msg_id.observed"].astype(bool) & np.isin(obs["messages.channel"], TARIFF_CHANNELS)
    live &= obs["messages.stated_effective_week.observed"].astype(bool)
    k_obs = obs["messages.k.observed"].astype(bool)
    head, tail, region = topology["edges_head"], topology["edges_tail"], topology["node_region"]
    n_edges, n_k = now.shape
    for i in np.nonzero(live)[0]:
        h0 = max(int(obs["messages.stated_effective_week"][i]) - week, 0)
        if h0 >= H:
            continue
        w = weights.get(int(obs["messages.channel"][i]), 0.0)
        if w <= 0.0:
            continue
        kind, target = int(obs["messages.target_kind"][i]), int(obs["messages.target"][i])
        if kind == TARGET_EDGE:
            es = [target] if 0 <= target < n_edges else []
        elif kind == TARGET_NODE:
            es = [e for e in range(n_edges) if head[e] == target or tail[e] == target]
        elif kind == TARGET_REGION:  # a region's import tariff: edges into it from outside
            es = [e for e in range(n_edges) if region[head[e]] == target and region[tail[e]] != target]
        else:
            es = []
        ks = [int(obs["messages.k"][i])] if k_obs[i] else range(n_k)
        for e in es:
            for k in ks:
                out[e, k, h0:] += bump * w
    return out


def slot_tau_total(obs: dict, topology: dict) -> np.ndarray:
    """Each slot's lead time this week, persisted: the sum of ``graph_now.tau`` over its edge chain."""
    tau = obs["graph_now.tau"]
    return np.array([sum(int(tau[e]) for e in chain) for chain in topology["slot_edges"]])


def slot_capacity(obs: dict, topology: dict, H: int, params: dict, open_forecast: dict) -> np.ndarray:
    """Upper bound on ``x[s, h]``: persisted edge capacity, derated by forecast chokepoint openness, zero where
    ``graph_now.prohibited`` (persisted) or a ``pending_prohibitions`` entry bans the slot's chain by week ``t+h``.
    """
    n = topology["n_slots"]
    cap = np.zeros((n, H), dtype=float)
    u = obs["graph_now.u"]
    prohibited = obs["graph_now.prohibited"]
    week = int(obs["week"][0])
    power = float(params["closure_power"])

    pend_obs = obs["pending_prohibitions.edge.observed"].astype(bool)
    pend_edge = obs["pending_prohibitions.edge"]
    pend_week = obs["pending_prohibitions.effective_week"]
    pend_map: dict = {}
    for i in np.nonzero(pend_obs)[0]:
        e, w = int(pend_edge[i]), int(pend_week[i])
        if e not in pend_map or w < pend_map[e]:
            pend_map[e] = w

    for s in range(n):
        chain = topology["slot_edges"][s]
        k = topology["slot_k"][s]
        base_u = min(float(u[e]) for e in chain)
        chokepoints = topology["slot_chokepoints"][s]
        for h in range(H):
            week_h = week + h
            if any(prohibited[e, k] for e in chain):
                continue
            if any(e in pend_map and week_h >= pend_map[e] for e in chain):
                continue
            value = base_u
            for node in chokepoints:
                if node in open_forecast:
                    value *= max(float(open_forecast[node][h]), 0.0) ** power
            cap[s, h] = max(value, 0.0)
    return cap


def build_lp(obs: dict, topology: dict, params: dict, commodities_v: np.ndarray) -> dict:
    """This week's rolling-horizon LP: freight + tariff + holding + shortage, minimized over ``H`` weeks."""
    H = int(params["H"])
    n = topology["n_slots"]
    tau = slot_tau_total(obs, topology)
    forecast = forecast_open(obs, topology, H, params)
    cap = slot_capacity(obs, topology, H, params, forecast)

    tracked = topology["tracked"]
    demands = topology["demands"]
    backlog_demands = [d for d, dem in enumerate(demands) if dem["backlog"]]
    backlog_pos = {d: bi for bi, d in enumerate(backlog_demands)}

    n_x = n * H
    n_stock = len(tracked) * H
    n_served_new = len(demands) * H
    n_backlog = len(backlog_demands) * H
    n_served_old = len(backlog_demands) * H
    procs = topology["procs"]
    grids = topology["grids"] if params.get("model_grids", False) else []
    n_proc = len(procs) * H
    # spill: what a tracked slot loses above its storage cap -- free at a supply node (its supply simply stops,
    # as the simulator does), disposal_cost[k] elsewhere; no cap at chokepoints (the simulator has none) or at
    # grids (their lng draw is not modeled, so a cap there would only ever be breached by the model itself)
    spill_slots = [
        i for i, (node, k) in enumerate(tracked)
        if topology["storage"].get((node, k)) is not None
        and node not in topology["chokepoint_nodes"]
        and (node not in topology["grid_set"] or grids)
    ]
    spill_pos = {i: j for j, i in enumerate(spill_slots)}
    n_spill = len(spill_slots) * H

    off_x = 0
    off_stock = off_x + n_x
    off_served_new = off_stock + n_stock
    off_backlog = off_served_new + n_served_new
    off_served_old = off_backlog + n_backlog
    burns = [(g, k, share) for g, gd in enumerate(grids) for (k, share) in gd["fuels"]]
    burn_at = {}
    for b, (g, k, _share) in enumerate(burns):
        burn_at.setdefault((grids[g]["node"], k), []).append(b)
    n_burn = len(burns) * H
    n_gserved = len(grids) * H
    off_proc = off_served_old + n_served_old
    off_spill = off_proc + n_proc
    off_burn = off_spill + n_spill
    off_gserved = off_burn + n_burn
    n_vars = off_gserved + n_gserved

    def iburn(b, h):
        return off_burn + b * H + h

    def igserved(g, h):
        return off_gserved + g * H + h

    def iproc(j, h):
        return off_proc + j * H + h

    def ispill(i, h):  # h in 1..H, like stock
        return off_spill + spill_pos[i] * H + (h - 1)

    def ix(s, h):
        return off_x + h * n + s

    def istock(i, h):
        return off_stock + i * H + (h - 1)

    def iserved_new(d, h):
        return off_served_new + d * H + h

    def ibacklog(bi, h):
        return off_backlog + bi * H + (h - 1)

    def iserved_old(bi, h):
        return off_served_old + bi * H + h

    c = np.zeros(n_vars)
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    freight = obs["graph_now.c"]
    tariff = tariff_forecast(obs, topology, H, params)
    for s in range(n):
        e0 = topology["slot_edges"][s][0]
        k = topology["slot_k"][s]
        for h in range(H):
            c[ix(s, h)] += float(freight[e0]) + float(tariff[e0, k, h]) * float(commodities_v[k])
            ub[ix(s, h)] = cap[s, h]

    holding_scale = float(params["holding_scale"])
    for i, (node, k) in enumerate(tracked):
        rate = holding_scale * topology["holding_cost"].get((node, k), 0.0)
        for h in range(1, H + 1):
            c[istock(i, h)] += rate
        if i in spill_pos:
            cap_qty = float(topology["storage"][(node, k)])
            unit = 0.0 if node in topology["supply_nodes"] else topology["disposal_cost"][k]
            for h in range(1, H + 1):
                ub[istock(i, h)] = cap_qty
                c[ispill(i, h)] += unit

    demand_h = []
    for d, dem in enumerate(demands):
        qty = obs["demand_forecast.qty"][d]
        arr = np.array([qty[h] if h < len(qty) else qty[-1] for h in range(H)])
        demand_h.append(arr)
        for h in range(H):
            ub[iserved_new(d, h)] = max(float(arr[h]), 0.0)
            c[iserved_new(d, h)] += -float(dem["pi"])
        if dem["backlog"]:
            bi = backlog_pos[d]
            ub[iserved_old(bi, 0)] = max(float(obs["backlog.qty"][d]), 0.0)


    A_eq_rows, b_eq = [], []
    A_ub_rows, b_ub = [], []

    stock0 = {p: float(obs["stock.qty"][topology["stock_index"][p]]) for p in tracked}
    supply_val = {p: float(obs["graph_now.supply.avail"][i]) for i, p in enumerate(topology["supply_list"])}

    pipeline_live = obs["pipeline.qty.observed"].astype(bool)
    pipeline_edge = obs["pipeline.edge"]
    pipeline_k = obs["pipeline.k"]
    pipeline_qty = obs["pipeline.qty"]
    pipeline_arrival = obs["pipeline.arrival_week"]
    pipeline_head = np.array([topology["edges_head"][e] for e in pipeline_edge])
    week = int(obs["week"][0])

    for i, (node, k) in enumerate(tracked):
        d_index = topology["demand_index"].get((node, k))
        bi = backlog_pos.get(d_index)
        for h in range(1, H + 1):
            row = np.zeros(n_vars)
            row[istock(i, h)] = 1.0
            rhs = 0.0
            if h == 1:
                rhs += stock0[(node, k)]
            else:
                row[istock(i, h - 1)] = -1.0
            rhs += supply_val.get((node, k), 0.0)
            live = np.nonzero(
                pipeline_live & (pipeline_k == k) & (pipeline_head == node) & (pipeline_arrival - week == h - 1)
            )[0]
            rhs += float(pipeline_qty[live].sum()) if live.size else 0.0
            for s, dest in enumerate(topology["slot_dest"]):
                if dest == node and topology["slot_k"][s] == k:
                    h_dep = (h - 1) - int(tau[s])
                    if 0 <= h_dep < H:
                        row[ix(s, h_dep)] += -1.0
            for s in topology["outgoing"].get((node, k), []):
                row[ix(s, h - 1)] += 1.0
            if d_index is not None:
                row[iserved_new(d_index, h - 1)] += 1.0
                if bi is not None:
                    row[iserved_old(bi, h - 1)] += 1.0
            # production: the input is consumed the week a process runs; the output becomes stock tau weeks later
            for j in topology["proc_in"].get((node, k), []):
                row[iproc(j, h - 1)] += 1.0
            for j in topology["proc_out"].get((node, k), []):
                h_start = (h - 1) - procs[j]["tau"]
                if 0 <= h_start < H:
                    row[iproc(j, h_start)] += -1.0
            if i in spill_pos:
                row[ispill(i, h)] += 1.0
            for b in burn_at.get((node, k), []):  # a grid burns its fuel the week it runs
                row[iburn(b, h - 1)] += 1.0
            A_eq_rows.append(row)
            b_eq.append(rhs)

    for d in backlog_demands:
        bi = backlog_pos[d]
        dem_h = demand_h[d]
        for h in range(1, H + 1):
            row = np.zeros(n_vars)
            row[ibacklog(bi, h)] = 1.0
            rhs = float(dem_h[h - 1])
            if h == 1:
                rhs += float(obs["backlog.qty"][d])
            else:
                row[ibacklog(bi, h - 1)] = -1.0
            row[iserved_new(d, h - 1)] += 1.0
            row[iserved_old(bi, h - 1)] += 1.0
            A_eq_rows.append(row)
            b_eq.append(rhs)
        for h in range(1, H):
            row = np.zeros(n_vars)
            row[iserved_old(bi, h)] = 1.0
            row[ibacklog(bi, h)] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)

    # `cumulative_cap` nodes (fab/OSAT output: chip_le_raw, chip_le) are NOT capped by stock.qty here: see
    # agents/mpc_lp/agent.py for why (confirmed harmful empirically).

    # each fab's / OSAT's effective capacity this week (persisted), shared by an OSAT's packages
    # terminal value: credit stock left at h = H, shipments arriving after H, and production finishing after H
    term = float(params.get("terminal_scale", 0.0))
    if term > 0.0:
        for i, (node, k) in enumerate(tracked):
            if node not in topology["grid_set"] or grids:
                c[istock(i, H)] -= term * float(commodities_v[k])
        for s in range(n):
            k = topology["slot_k"][s]
            if topology["slot_dest"][s] in topology["grid_set"] and not grids:
                continue
            for h in range(H):
                if h + int(tau[s]) >= H:
                    c[ix(s, h)] -= term * float(commodities_v[k])
        for j, pr in enumerate(procs):
            for h in range(H):
                if h + pr["tau"] >= H:
                    c[iproc(j, h)] -= term * float(commodities_v[pr["k_out"]])

    if grids:
        G_bar = obs["graph_now.grid.G_bar"]
        y_bar = obs["graph_now.grid.y_bar"]
        tidx = {p: i for i, p in enumerate(tracked)}
        psi = topology["psi"]
        for b, (g, k, share) in enumerate(burns):
            gd = grids[g]
            cap_b = share * max(float(G_bar[gd["row"]]), 0.0)
            i = tidx.get((gd["node"], k))
            for h in range(H):
                ub[iburn(b, h)] = cap_b
                # rationing (15): psi I-bar burn <= share G-bar I^{h-1} for the rationed fuel
                if k == gd["rationed"] and i is not None and psi * gd["ibar"].get(k, 0.0) > 0:
                    thr = psi * gd["ibar"][k]
                    row = np.zeros(n_vars)
                    row[iburn(b, h)] = thr
                    if h == 0:
                        rhs = cap_b * stock0[(gd["node"], k)]
                    else:
                        row[istock(i, h)] = -cap_b
                        rhs = 0.0
                    A_ub_rows.append(row)
                    b_ub.append(rhs)
        for g, gd in enumerate(grids):
            yb = max(float(y_bar[gd["row"]]), 0.0)
            null = gd["share_null"] * max(float(G_bar[gd["row"]]), 0.0)
            members = [b for b, (gg, _k, _sh) in enumerate(burns) if gg == g]
            for h in range(H):
                ub[igserved(g, h)] = yb
                c[igserved(g, h)] -= gd["voll"]  # shed = y_bar - served, at VOLL
                row = np.zeros(n_vars)
                row[igserved(g, h)] = 1.0
                for b in members:
                    row[iburn(b, h)] = -1.0
                A_ub_rows.append(row)
                b_ub.append(null)

    rate_src = {"fab": obs["graph_now.fab.cap_eff"], "osat": obs["graph_now.osat.thr_eff"]}
    for g, (kind, row_idx) in enumerate(topology["cap_groups"]):
        members = [j for j, pr in enumerate(procs) if pr["group"] == g]
        rate = max(float(rate_src[kind][row_idx]), 0.0)
        for h in range(H):
            row = np.zeros(n_vars)
            for j in members:
                row[iproc(j, h)] = 1.0
            A_ub_rows.append(row)
            b_ub.append(rate)

    bounds = list(zip(lb.tolist(), ub.tolist()))  # last: the grid and spill blocks above still set ub
    A_eq = np.array(A_eq_rows) if A_eq_rows else np.zeros((0, n_vars))
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub)

    return {
        "c": c,
        "bounds": bounds,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "n_vars": n_vars,
        "H": H,
        "n_slots": n,
        "cap": cap,
        "index": {"x": ix, "stock": istock, "served_new": iserved_new, "backlog": ibacklog, "served_old": iserved_old,
                  "proc": iproc, "spill": ispill},
    }


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        self.topology = build_topology(static, layout)
        fm = PARAMS.get("fuel_mult", 1.0)
        self._fuel_mult = float(fm.get(static["instance"]["kind"], 1.0) if isinstance(fm, dict) else fm)
        floor = PARAMS.get("upstream_floor", 0.0)
        self._upstream_floor = float(floor.get(static["instance"]["kind"], 0.0) if isinstance(floor, dict) else floor)
        self.commodities_v = np.asarray(static["commodities"]["v"], dtype=float)
        self.rng = np.random.default_rng(config["policy_seed"])
        action = config["spaces"]["action"]
        self._zero_override = np.zeros(action["override_qty"]["shape"])
        self._zero_release = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

        # reroute: per release pair (strait c, tanker commodity k), its override slots with the straits each slot's
        # lane passes after c (the route the released cargo still has ahead)
        lanes_ck = static["lanes"]["chokepoints"]
        ov = static["override_slots"]
        self._pairs = [tuple(p) for p in layout["release_pairs"]]
        pair_index = {p: i for i, p in enumerate(self._pairs)}
        self._pair_slots = [[] for _ in self._pairs]  # per pair: (override slot, straits ahead)
        self._pair_lane_ahead = [[] for _ in self._pairs]  # per pair: straits ahead on every lane through c
        for o in range(len(ov["chokepoint"])):
            c, k, lane = ov["chokepoint"][o], ov["k"][o], ov["lane"][o]
            ahead = []
            if lane is not None:
                chain = list(lanes_ck[lane])
                ahead = chain[chain.index(c) + 1:] if c in chain else []
            pi = pair_index.get((c, k))
            if pi is not None:
                self._pair_slots[pi].append((o, ahead))
        for pi, (c, k) in enumerate(self._pairs):
            for lane, chain in enumerate(lanes_ck):
                chain = list(chain)
                if c in chain:
                    self._pair_lane_ahead[pi].append(chain[chain.index(c) + 1:])
        self._n_override = len(ov["chokepoint"])

        # the fallback rule: agents/heuristic's persist + closure_power derate, self-contained here so a
        # solver failure never depends on the LP machinery that just failed
        u0 = static["edges"]["u0"]
        slots = static["action_slots"]
        self._fallback_cap = np.array([u0[e] for e in slots["edge"]], dtype=float)
        self._fallback_power = 1.0
        position = {node: i for i, node in enumerate(layout["chokepoints"])}
        lane_chokepoints = static["lanes"]["chokepoints"]
        self._fallback_through = [
            [position[c] for c in lane_chokepoints[lane]] if lane is not None else [] for lane in slots["lane"]
        ]

    def act(self, observation):
        lp = None  # None when the LP failed this week (the fallback ran)
        try:
            lp = build_lp(observation, self.topology, PARAMS, self.commodities_v)
            res = linprog(
                lp["c"], A_eq=lp["A_eq"], b_eq=lp["b_eq"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], bounds=lp["bounds"]
            )
            if not res.success:
                raise RuntimeError(getattr(res, "message", "linprog did not succeed"))
            ix = lp["index"]["x"]
            flows = np.array([res.x[ix(s, 0)] for s in range(self.topology["n_slots"])])
            heuristic_flows = self._fallback(observation)

            # EVOLVE-BLOCK-START
            # Decide, per action slot, whether to trust the LP's week-0 flow or the plain heuristic rule
            # (``heuristic_flows``). The LP only ever rewards shipments it can trace to served demand, so it
            # correctly but unhelpfully zeroes out any slot whose lane does not reach the demand sink --
            # confirmed to include chokepoint-crossing slots that feed a grid rather than the sink. The
            # current rule: trust the LP only for slots whose lane destination is a demand node; everything
            # else uses the heuristic. A different criterion (e.g. also trusting the LP where its own
            # forecast shows meaningful risk) may do better -- this block decides, per slot ``s``, whether
            # ``flows[s]`` keeps the LP's value or is overwritten with ``heuristic_flows[s]``.
            # Stage 2 option: with production modeled, the LP also prices wafer/raw-chip shipments, so
            # lp_scope="all_but_grid" trusts it everywhere but the lanes into a grid (whose lng draw is still
            # unmodeled); lp_scope="demand" keeps the Stage 1 rule (only lanes into a demand node).
            lp_cap0 = lp["cap"][:, 0]
            all_but_grid = PARAMS.get("lp_scope", "all_but_grid") == "all_but_grid"
            for s in range(self.topology["n_slots"]):
                dest_k = (self.topology["slot_dest"][s], self.topology["slot_k"][s])
                if all_but_grid and PARAMS.get("model_grids", False):
                    # the last leg into a grid stays with the heuristic (drain the terminal into the grid): the LP's
                    # lagged accounting there underships fuel and sheds more (small: +45..257 B$ shed per episode)
                    use_heuristic = self.topology["slot_dest"][s] in self.topology["grid_set"]
                elif all_but_grid:
                    use_heuristic = (
                        self.topology["slot_dest"][s] in self.topology["grid_set"]
                        or self.topology["slot_k"][s] in self.topology["grid_fuels"]
                    )
                else:
                    use_heuristic = dest_k not in self.topology["demand_set"]
                if not use_heuristic and dest_k not in self.topology["demand_set"]:
                    # an upstream slot the LP decides (Stage 2): never below upstream_floor x the heuristic, so a
                    # horizon too short to see the chain's lead time cannot starve production ahead of a shock
                    floor = self._upstream_floor * min(heuristic_flows[s], lp_cap0[s])
                    flows[s] = max(flows[s], floor)
                if use_heuristic:
                    # heuristic rule only derates on *observed* graph_now.open; the LP's own week-0
                    # cap also folds in pending_prohibitions that take effect this week (banned before
                    # graph_now.prohibited reflects it) and the forecast closure probability. Capping
                    # the heuristic flow by it is a one-sided safety clamp, never a step up.
                    flows[s] = min(heuristic_flows[s], lp_cap0[s])
            # EVOLVE-BLOCK-END
        except Exception:
            flows = self._fallback(observation)
        flows_before_fuel = flows.copy()
        try:
            self._fuel(observation, flows, lp)
        except Exception:
            flows = flows_before_fuel  # a failing fuel rule must not cost the week
        override_qty, release_mode = self._zero_override, self._zero_release
        if PARAMS.get("reroute", False):
            try:
                override_qty, release_mode = self._reroute(observation)
            except Exception:
                override_qty, release_mode = self._zero_override, self._zero_release
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}

    def _fuel(self, observation, flows, lp):
        """Set the grid fuel slots' requests in place (see the block's comment)."""
        # EVOLVE-BLOCK-START
        # Grid fuel (lng, crude, nucfuel on small): ~60% of the cost on small is power shed at grids, and 98% of it
        # falls in weeks a grid is short of one specific fuel (fuels are not substitutes: fuel k gives at most
        # share_k * G_bar). Fuel piles up at sources while grids shed; the environment clips each request to
        # edge capacity and stock, so asking for more than the heuristic's nominal u0 already helped (x10:
        # small 0.558 -> 0.610). This block sets flows[s] for every fuel slot s (topology["grid_fuels"]
        # holds their commodities). Available: observation (stock.qty, graph_now.u, graph_now.open,
        # graph_now.grid.G_bar / y_bar, last_week.shed.qty, last_week.clip.requested / executed, ...),
        # self.topology (slot_tail, slot_dest, slot_k, slot_chokepoints, grids with their fuel shares,
        # stock_index), lp["cap"] (forecast-derated capacity, n_slots x H; lp is None when the LP failed this week). Requests must be finite and >= 0.
        mult = self._fuel_mult
        if mult != 1.0:
            for s in range(self.topology["n_slots"]):
                if self.topology["slot_k"][s] in self.topology["grid_fuels"]:
                    flows[s] = flows[s] * mult
        # EVOLVE-BLOCK-END

    def _reroute(self, observation):
        """Release queued tanker cargo at an open strait onto override slots whose route ahead is open, when some
        lane through that strait runs into a closed one; the simulator clips the quantity to the queue, the out-edge
        capacity and the strait's throughput. Openness: the forecast over the next weeks (minimum), not just now.
        """
        H = int(PARAMS["H"])
        fc = forecast_open(observation, self.topology, H, PARAMS)
        pos = self.topology["chokepoint_pos"]
        thr = float(PARAMS.get("reroute_open", 0.5))
        look = min(H, 4)

        use_fc = bool(PARAMS.get("reroute_forecast", False))

        def is_open(node):
            if node not in pos:
                return True
            if use_fc:
                return float(np.min(fc[node][:look])) >= thr
            return float(observation["graph_now.open"][pos[node]]) >= thr  # observed this week

        mask = observation["override_mask"]
        qty = np.zeros(self._n_override)
        mode = np.zeros(len(self._pairs), dtype=np.int64)
        for pi, (c, _k) in enumerate(self._pairs):
            if c in pos and float(observation["graph_now.open"][pos[c]]) < thr:
                continue  # nothing leaves a closed strait anyway
            blocked = any(not all(is_open(n) for n in ahead) for ahead in self._pair_lane_ahead[pi])
            if not blocked:
                continue
            good = [o for o, ahead in self._pair_slots[pi] if mask[o] == 1 and all(is_open(n) for n in ahead)]
            if not good:
                continue
            mode[pi] = 1
            for o in good:
                qty[o] = 1e12
        return qty, mode

    def _fallback(self, observation):
        flows = self._fallback_cap * observation["action_mask"]
        open_now = observation["graph_now.open"]
        seen = observation["graph_now.open.observed"] == 1
        for s, chokepoints in enumerate(self._fallback_through):
            for c in chokepoints:
                if seen[c]:
                    flows[s] *= max(float(open_now[c]), 0.0) ** self._fallback_power
        return flows
