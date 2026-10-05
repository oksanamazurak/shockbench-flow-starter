# MPC agent with warning/messages closure forecast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new submission agent, `agents/mpc_lp/`, that plans `flows` with a rolling-horizon linear program (`scipy.optimize.linprog`, HiGHS) whose forecast of each chokepoint's future openness comes from `warning.score` and `messages.*`, not just persistence.

**Architecture:** One file, `agents/mpc_lp/agent.py`. A static topology pass at `__init__` (lane/edge chains, tau sources, which stock slots are dynamically tracked vs. cumulative-capped vs. ignored — see "Stock-tracking rule" below). Every `act()`: forecast chokepoint openness, build the LP's `c`/`bounds`/`A_eq`/`b_eq`/`A_ub`/`b_ub` fresh from the current observation, solve, return week 0's flows. A solve failure falls back to the `agents/mine` heuristic rule instead of crashing.

**Tech Stack:** numpy, `scipy.optimize.linprog` (HiGHS, already the default method in scipy 1.18, confirmed installed). Tests: `pytest`, using a real `tiny` episode (`gymnasium` + `shockbench_flow_gym`, training-only, fine in `tests/`) rather than fabricated fixtures, matching this repo's existing test/example style.

## Global Constraints

- `agent.py` may import only Python 3.13 stdlib, numpy, scipy (CPU), torch (CPU) — confirmed: this plan's `agent.py` imports only `json`, `pathlib`, `numpy`, `scipy.optimize`.
- Load files relative to `Path(__file__).parent`; seed every RNG from `config["policy_seed"]`.
- CPU budget: 2s/week on `small`, 4s/week on `full` (no published number for `tiny`, but `sbf check` meters it); a crashed/malformed/over-budget week is played by the naive rule on the real server — this plan's fallback exists so a solver failure degrades to the heuristic instead, not to that.
- Per this session's instruction: **do not run `git add`/`git commit` at any step of this plan.** Steps that would normally end in a commit instead end in "stop here; do not commit" — the user commits manually, if at all.
- Spec: `docs/superpowers/specs/2026-10-04-mpc-warning-forecast-design.md`. This plan implements Stage 1 only (no war-risk/disposal/shed/queue-lot/production modeling).

## Stock-tracking rule (concrete reading of the spec's §2 stock balance, pinned down from `tiny`'s real `static`/`layout`)

Inspecting `tiny`'s real `config["layout"]["stock_slots"]` shows every fab/OSAT node holds **two** stock slots: an input commodity (e.g. `fab_tw/wafer`) whose stock the environment consumes to run production we are not modeling, and an output commodity (e.g. `fab_tw/chip_le_raw`) the environment produces by that same unmodeled process. Only the **output** side has an outgoing action slot we control. This splits every stock slot into exactly one of three roles, decided once per episode in `build_topology`:

1. **`tracked`** — the slot is a `supply_slot` (a source replenished by `graph_now.supply.avail`) or the destination of at least one action slot's lane/edge (something we ship into, including the demand sink). Gets a full per-week `stock[i, h]` LP variable, `h = 1..H`, with a known `h=0`.
2. **`cumulative_cap`** — has an outgoing action slot but no modeled inflow (fab/OSAT output commodities: inflow is the unmodeled production). No per-week variable; one inequality caps total shipments over the whole horizon by what is on hand right now: `sum_h outflow(node, k, h) <= stock.qty[node, k]` (conservative — ignores any future unmodeled production, so the LP never promises more than it can already see).
3. **ignored** — neither (e.g. a grid's `lng` buffer, a fab's input `wafer` stock): no action slot in our controlled graph depends on their level, so no constraint is needed beyond the normal edge-capacity bound on whatever ships into them.

The demand sink is a `tracked` node whose outflow is `served`, not an action slot; see Task 3.

## File Structure

- Create `agents/mpc_lp/agent.py` — everything: topology, forecast, LP assembly, `Agent` class, fallback, `PARAMS`/`params.json`.
- Create `tests/test_mpc_agent.py` — unit tests against a real `tiny` episode and small hand-built scenarios.
- Create `examples/07_mpc_policy_search.py` — calibration script (Task 5), a copy of `06_policy_search.py`'s search loop over the new agent's parameters.

---

### Task 1: Topology preprocessing

**Files:**
- Create: `agents/mpc_lp/agent.py` (this task writes the module docstring, imports, `PARAMS`, and `build_topology`)
- Test: `tests/test_mpc_agent.py`

**Interfaces:**
- Produces: `build_topology(static: dict, layout: dict) -> dict` with keys `n_slots`, `slot_edges` (list[list[int]], edge-index chain per slot), `slot_dest` (list[int], destination node per slot), `slot_tail` (list[int]), `slot_chokepoints` (list[list[int]], chokepoint node indices per slot), `slot_k` (list[int]), `chokepoint_pos` (dict node->row in `graph_now.open`/`graph_now.kappa.*`), `chokepoint_warn_row` (dict node->row in `warning.score`), `stock_index` (dict (node,k)->row in `stock.qty`), `supply_set` (set of (node,k)), `supply_list` (list of (node,k) in `layout["supply_slots"]` order), `demand_set` (set of (node,k)), `demands` (list of dicts `{"node", "k", "pi", "backlog"}` in `layout["demands"]`/`static["sinks"]` order), `demand_index` (dict (node,k)->position in `demands`), `outgoing`/`incoming` (dict (node,k)->list[int] of slot indices), `tracked`/`cumulative_cap` (list of (node,k), per the stock-tracking rule above), `edges_head` (list[int], `static["edges"]["head"]`, kept for the pipeline-arrival lookup in Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mpc_agent.py
"""Unit tests for agents/mpc_lp/agent.py, against a real tiny episode (not fabricated fixtures)."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import shockbench_flow_gym  # noqa: F401 - registers ShockBench/*
from shockbench_flow_agent import agent_config

from sbf_starter import env_id

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "mpc_lp"))
import agent as mpc  # noqa: E402


@pytest.fixture(scope="module")
def tiny_episode():
    env = gym.make(env_id("tiny"))
    obs, info = env.reset(options={"episode": 0})
    config = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    return config, obs


def test_build_topology_tiny_shapes(tiny_episode):
    config, _obs = tiny_episode
    topo = mpc.build_topology(config["static"], config["layout"])
    n_slots = len(config["static"]["action_slots"]["edge"])
    assert topo["n_slots"] == n_slots == 20
    assert len(topo["slot_edges"]) == n_slots
    assert len(topo["slot_dest"]) == n_slots
    # slot 0 is lane L0 (edges [0, 1]): src_gulf(0) -> chk(3) -> grid_tw(4)
    assert topo["slot_edges"][0] == [0, 1]
    assert topo["slot_tail"][0] == 0
    assert topo["slot_dest"][0] == 4
    assert topo["slot_chokepoints"][0] == [3]
    # slot 19 (E24, no lane): osat_sea(10) -> sink_us(11) directly
    assert topo["slot_edges"][19] == [24]
    assert topo["slot_chokepoints"][19] == []


def test_build_topology_chokepoint_rows(tiny_episode):
    config, _obs = tiny_episode
    topo = mpc.build_topology(config["static"], config["layout"])
    # tiny's one chokepoint is node 3, row 0 of graph_now.open, row 15 of warning.score
    assert topo["chokepoint_pos"] == {3: 0}
    assert topo["chokepoint_warn_row"] == {3: 15}


def test_build_topology_stock_roles(tiny_episode):
    config, _obs = tiny_episode
    topo = mpc.build_topology(config["static"], config["layout"])
    # source: src_gulf/lng (node 0, k 0) is a supply_slot with an outgoing slot -> tracked
    assert (0, 0) in topo["tracked"]
    # fab_tw/chip_le_raw (node 7, k 2): unmodeled inflow, has outgoing slots -> cumulative_cap
    assert (7, 2) in topo["cumulative_cap"]
    # fab_tw/wafer (node 7, k 1): unmodeled inflow, no outgoing slot (only incoming) -> ignored
    assert (7, 1) not in topo["cumulative_cap"]
    assert (7, 1) not in topo["tracked"]
    # the demand sink (sink_us/chip_le, node 11, k 3) is tracked and listed in demands
    assert (11, 3) in topo["tracked"]
    assert topo["demands"][topo["demand_index"][(11, 3)]]["pi"] == pytest.approx(50360.0)
    assert topo["demands"][topo["demand_index"][(11, 3)]]["backlog"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_mpc_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent'` (the module does not exist yet).

- [ ] **Step 3: Write `agents/mpc_lp/agent.py`'s topology code**

```python
# agents/mpc_lp/agent.py
"""Rolling-horizon LP agent (routing + timing only) with a chokepoint-closure forecast built from
``warning.score`` and ``messages.*``. Design: docs/superpowers/specs/2026-10-04-mpc-warning-forecast-design.md.

Stage 1 only: no war-risk surcharge, disposal/shed cost, queue-lot/release-mode mechanics, or
fab/OSAT/grid production modeling. ``override_qty``/``release_mode`` stay at their default.
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
    "holding_rate": 1.0,  # USD per unit per week held in a tracked stock slot (not published; tunable)
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())

CLOSURE_CHANNELS = (3, 4, 5)  # sanction_legal, ties_threat, mid_threat (tariff channels excluded: Stage 1)
MESSAGE_TARGET_CHOKEPOINT = 0


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
        {"node": static["sinks"]["node"][i], "k": static["sinks"]["k"][i], "pi": static["sinks"]["pi"][i],
         "backlog": static["sinks"]["backlog"][i]}
        for i in range(len(static["sinks"]["node"]))
    ]
    demand_index = {(d["node"], d["k"]): i for i, d in enumerate(demands)}
    demand_set = set(demand_index)

    outgoing: dict = {}
    incoming: dict = {}
    for s in range(n_slots):
        outgoing.setdefault((slot_tail[s], slot_k[s]), []).append(s)
        incoming.setdefault((slot_dest[s], slot_k[s]), []).append(s)

    tracked, cumulative_cap = [], []
    for p in stock_slots:
        if p in supply_set or p in incoming:
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
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_mpc_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Stop here — do not commit** (per this session's instruction).

---

### Task 2: Forecast module

**Files:**
- Modify: `agents/mpc_lp/agent.py` (add `forecast_open`)
- Test: `tests/test_mpc_agent.py` (add)

**Interfaces:**
- Consumes: `build_topology`'s output (`chokepoint_pos`, `chokepoint_warn_row`).
- Produces: `forecast_open(obs: dict, topology: dict, H: int, params: dict) -> dict[int, np.ndarray]` — chokepoint node index -> length-`H` array of forecast open fraction, index 0 equal to the observed `graph_now.open` exactly.

- [ ] **Step 1: Write the failing test**

```python
def test_forecast_open_persists_without_signals(tiny_episode):
    config, obs = tiny_episode
    topo = mpc.build_topology(config["static"], config["layout"])
    forecast = mpc.forecast_open(obs, topo, H=4, params=mpc.PARAMS)
    assert set(forecast) == {3}
    arr = forecast[3]
    assert arr.shape == (4,)
    assert arr[0] == pytest.approx(float(obs["graph_now.open"][0]))
    # warning.score at tiny's reset is not necessarily 0: just check the forecast never exceeds open_now
    assert np.all(arr <= float(obs["graph_now.open"][0]) + 1e-9)
    assert np.all(arr >= 0.0)


def test_forecast_open_message_with_effective_week():
    obs = {
        "week": np.array([5]),
        "graph_now.open": np.array([1.0]),
        "warning.score": np.zeros(16),
        "messages.msg_id.observed": np.array([1, 0]),
        "messages.channel": np.array([3, 0]),  # sanction_legal
        "messages.kind": np.array([2, 0]),  # threat
        "messages.target_kind": np.array([0, 0]),  # chokepoint
        "messages.target": np.array([3, 0]),
        "messages.stated_effective_week": np.array([7, 0]),
        "messages.stated_effective_week.observed": np.array([1, 0]),
    }
    topo = {"chokepoint_pos": {3: 0}, "chokepoint_warn_row": {3: 15}}
    params = dict(mpc.PARAMS)
    params["msg_weight"] = {"2": 0.4}
    forecast = mpc.forecast_open(obs, topo, H=5, params=params)
    arr = forecast[3]
    # week 5 + h = 7 -> h = 2; weight 0.4 -> open_forecast = 1.0 * (1 - 0.4) = 0.6
    assert arr[2] == pytest.approx(0.6)
    assert arr[0] == pytest.approx(1.0)  # h=0 is the observed week, untouched
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_mpc_agent.py -v -k forecast_open`
Expected: FAIL — `AttributeError: module 'agent' has no attribute 'forecast_open'`.

- [ ] **Step 3: Write `forecast_open`**

```python
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
        p_warn = 0.0
        row = topology["chokepoint_warn_row"].get(node)
        if row is not None:
            score = float(obs["warning.score"][row])
            p_warn = 1.0 / (1.0 + np.exp(-(a * score - b)))

        p_msg = np.zeros(H, dtype=float)
        for i in np.nonzero(msg_live & (msg_target == node))[0]:
            weight = msg_weight.get(int(msg_kind[i]), 0.0)
            if weight <= 0.0:
                continue
            if msg_eff_observed[i]:
                h = int(msg_eff_week[i]) - week
                if 0 <= h < H:
                    p_msg[h] = min(1.0, p_msg[h] + weight)
            else:
                for h in range(min(bump_weeks, H)):
                    p_msg[h] = min(1.0, p_msg[h] + bump * weight)

        p_close = np.clip(p_warn + p_msg, 0.0, 1.0)
        forecast = open_now * (1.0 - p_close)
        forecast[0] = open_now
        out[node] = forecast
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_mpc_agent.py -v -k forecast_open`
Expected: PASS (2 tests).

- [ ] **Step 5: Stop here — do not commit.**

---

### Task 3: LP assembly

**Files:**
- Modify: `agents/mpc_lp/agent.py` (add `slot_capacity`, `slot_tau_total`, `build_lp`)
- Test: `tests/test_mpc_agent.py` (add)

**Interfaces:**
- Consumes: `build_topology`'s output, `forecast_open`'s output.
- Produces:
  - `slot_tau_total(obs: dict, topology: dict) -> np.ndarray` — shape `(n_slots,)`, integer lead time per slot (summed over its chain), this week's persisted.
  - `slot_capacity(obs, topology, H, params, open_forecast) -> np.ndarray` — shape `(n_slots, H)`, the LP's upper bound on each `x[s, h]`.
  - `build_lp(obs, topology, params, commodities_v) -> dict` with keys `c`, `bounds` (list of `(lo, hi)`), `A_eq`, `b_eq`, `A_ub`, `b_ub`, `n_vars`, `H`, `n_slots`, and `index` — a dict of the four index functions below, for `Agent.act` and tests to read the solution back:
    - `index["x"](s, h)`, `index["stock"](i, h)` (`h=1..H`), `index["served_new"](d, h)`, `index["backlog"](bi, h)` (`h=1..H`), `index["served_old"](bi, h)` — `bi` is the position within `[d for d in demands if demands[d]["backlog"]]`.

- [ ] **Step 1: Write the failing test**

```python
def test_slot_capacity_zero_when_prohibited(tiny_episode):
    config, obs = tiny_episode
    topo = mpc.build_topology(config["static"], config["layout"])
    forecast = mpc.forecast_open(obs, topo, H=3, params=mpc.PARAMS)
    cap = mpc.slot_capacity(obs, topo, H=3, params=mpc.PARAMS, open_forecast=forecast)
    assert cap.shape == (20, 3)
    action_mask = obs["action_mask"]
    for s in range(20):
        if action_mask[s] == 0:
            assert cap[s, 0] == 0.0


def test_slot_capacity_pending_prohibition_blocks_future_week():
    topo = {
        "n_slots": 1,
        "slot_edges": [[0]],
        "slot_k": [0],
        "slot_chokepoints": [[]],
    }
    obs = {
        "week": np.array([10]),
        "graph_now.u": np.array([50.0]),
        "graph_now.prohibited": np.zeros((1, 1)),
        "pending_prohibitions.edge": np.array([0, 0]),
        "pending_prohibitions.effective_week": np.array([12, 0]),
        "pending_prohibitions.edge.observed": np.array([1, 0]),
    }
    cap = mpc.slot_capacity(obs, topo, H=4, params=mpc.PARAMS, open_forecast={})
    # week 10: h=0 -> week 10 (ok), h=1 -> week 11 (ok), h=2 -> week 12 (banned), h=3 -> week 13 (banned)
    assert list(cap[0]) == [50.0, 50.0, 0.0, 0.0]


def test_build_lp_single_source_two_weeks():
    """A hand-built 1-source/1-slot/no-demand scenario: the LP should just respect the capacity bound."""
    topo = {
        "n_slots": 1, "slot_edges": [[0]], "slot_dest": [1], "slot_tail": [0], "slot_chokepoints": [[]],
        "slot_k": [0], "chokepoint_pos": {}, "chokepoint_warn_row": {}, "stock_index": {(0, 0): 0, (1, 0): 1},
        "supply_list": [(0, 0)], "supply_set": {(0, 0)}, "demands": [], "demand_index": {}, "demand_set": set(),
        "outgoing": {(0, 0): [0]}, "incoming": {(1, 0): [0]}, "tracked": [(0, 0), (1, 0)], "cumulative_cap": [],
        "edges_head": [1],
    }
    obs = {
        "week": np.array([1]), "graph_now.u": np.array([10.0]), "graph_now.tau": np.array([0]),
        "graph_now.c": np.array([2.0]), "graph_now.tariff": np.array([[0.0]]),
        "graph_now.prohibited": np.zeros((1, 1)), "graph_now.open": np.array([]),
        "action_mask": np.array([1]),
        "pending_prohibitions.edge": np.array([0]), "pending_prohibitions.effective_week": np.array([0]),
        "pending_prohibitions.edge.observed": np.array([0]),
        "stock.qty": np.array([100.0, 0.0]),
        "graph_now.supply.avail": np.array([0.0]),
        "pipeline.edge": np.array([0]), "pipeline.k": np.array([0]), "pipeline.qty": np.array([0.0]),
        "pipeline.arrival_week": np.array([0]), "pipeline.qty.observed": np.array([0]),
        "messages.msg_id.observed": np.zeros(1), "messages.channel": np.zeros(1), "messages.kind": np.zeros(1),
        "messages.target_kind": np.zeros(1), "messages.target": np.zeros(1),
        "messages.stated_effective_week": np.zeros(1), "messages.stated_effective_week.observed": np.zeros(1),
        "demand_forecast.qty": np.zeros((0, 8)), "backlog.qty": np.zeros(0),
    }
    params = dict(mpc.PARAMS, H=2)
    lp = mpc.build_lp(obs, topo, params, commodities_v=np.array([1.0]))
    assert lp["n_vars"] > 0
    from scipy.optimize import linprog
    res = linprog(lp["c"], A_eq=lp["A_eq"], b_eq=lp["b_eq"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], bounds=lp["bounds"])
    assert res.success
    x = res.x
    ix = lp["index"]["x"]
    # cheapest valid plan ships nothing (no demand to serve, holding/freight cost only penalizes shipping)
    assert x[ix(0, 0)] == pytest.approx(0.0, abs=1e-6)
    assert x[ix(0, 1)] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_mpc_agent.py -v -k "slot_capacity or build_lp"`
Expected: FAIL — `AttributeError: module 'agent' has no attribute 'slot_capacity'`.

- [ ] **Step 3: Write `slot_tau_total`, `slot_capacity`, `build_lp`**

```python
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

    off_x = 0
    off_stock = off_x + n_x
    off_served_new = off_stock + n_stock
    off_backlog = off_served_new + n_served_new
    off_served_old = off_backlog + n_backlog
    n_vars = off_served_old + n_served_old

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
    tariff = obs["graph_now.tariff"]
    for s in range(n):
        e0 = topology["slot_edges"][s][0]
        k = topology["slot_k"][s]
        unit_cost = float(freight[e0]) + float(tariff[e0, k]) * float(commodities_v[k])
        for h in range(H):
            c[ix(s, h)] += unit_cost
            ub[ix(s, h)] = cap[s, h]

    for i in range(len(tracked)):
        for h in range(1, H + 1):
            c[istock(i, h)] += float(params["holding_rate"])

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

    bounds = list(zip(lb.tolist(), ub.tolist()))

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
            live = np.nonzero(pipeline_live & (pipeline_k == k) & (pipeline_head == node)
                               & (pipeline_arrival - week == h - 1))[0]
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

    for node, k in topology["cumulative_cap"]:
        row = np.zeros(n_vars)
        for s in topology["outgoing"].get((node, k), []):
            for h in range(H):
                row[ix(s, h)] += 1.0
        A_ub_rows.append(row)
        b_ub.append(float(obs["stock.qty"][topology["stock_index"][(node, k)]]))

    A_eq = np.array(A_eq_rows) if A_eq_rows else np.zeros((0, n_vars))
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub)

    return {
        "c": c, "bounds": bounds, "A_eq": A_eq, "b_eq": b_eq, "A_ub": A_ub, "b_ub": b_ub,
        "n_vars": n_vars, "H": H, "n_slots": n,
        "index": {"x": ix, "stock": istock, "served_new": iserved_new, "backlog": ibacklog,
                   "served_old": iserved_old},
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_mpc_agent.py -v -k "slot_capacity or build_lp"`
Expected: PASS (3 tests). If `test_build_lp_single_source_two_weeks` fails on a shape or index error, print `lp["A_eq"].shape`, `lp["n_vars"]` and re-check the offsets above — this is the task's main risk area, flagged for exactly this reason.

- [ ] **Step 5: Stop here — do not commit.**

---

### Task 4: `Agent` class, fallback, `params.json`

**Files:**
- Modify: `agents/mpc_lp/agent.py` (add `Agent`)
- Test: `tests/test_mpc_agent.py` (add)

**Interfaces:**
- Consumes: `build_topology`, `build_lp`, `PARAMS`.
- Produces: `class Agent` with `__init__(self, config=None)` and `act(self, observation) -> dict` (`{"flows": ...}`), matching every other agent in this repo.

- [ ] **Step 1: Write the failing test**

```python
def test_agent_act_returns_valid_flows(tiny_episode):
    config, obs = tiny_episode
    agent = mpc.Agent(config)
    action = agent.act(obs)
    flows = action["flows"]
    assert flows.shape == (20,)
    assert np.all(flows >= -1e-9)
    assert np.all(flows[obs["action_mask"] == 0] <= 1e-9)
    u = obs["graph_now.u"]
    for s in range(20):
        e0 = agent.topology["slot_edges"][s][0]
        assert flows[s] <= float(u[e0]) + 1e-6


def test_agent_falls_back_on_solver_failure(tiny_episode, monkeypatch):
    config, obs = tiny_episode
    agent = mpc.Agent(config)

    class _FailedResult:
        success = False

    monkeypatch.setattr(mpc, "linprog", lambda *a, **k: _FailedResult())
    action = agent.act(obs)
    assert action["flows"].shape == (20,)
    assert np.all(np.isfinite(action["flows"]))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_mpc_agent.py -v -k test_agent`
Expected: FAIL — `AttributeError: module 'agent' has no attribute 'Agent'`.

- [ ] **Step 3: Write `Agent`**

```python
class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        self.topology = build_topology(static, layout)
        self.commodities_v = np.asarray(static["commodities"]["v"], dtype=float)
        self.rng = np.random.default_rng(config["policy_seed"])
        action = config["spaces"]["action"]
        self._zero_override = np.zeros(action["override_qty"]["shape"])
        self._zero_release = np.zeros(action["release_mode"]["shape"], dtype=np.int64)

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
        try:
            lp = build_lp(observation, self.topology, PARAMS, self.commodities_v)
            res = linprog(lp["c"], A_eq=lp["A_eq"], b_eq=lp["b_eq"], A_ub=lp["A_ub"], b_ub=lp["b_ub"],
                           bounds=lp["bounds"])
            if not res.success:
                raise RuntimeError(res.message if hasattr(res, "message") else "linprog did not succeed")
            ix = lp["index"]["x"]
            flows = np.array([res.x[ix(s, 0)] for s in range(self.topology["n_slots"])])
        except Exception:
            flows = self._fallback(observation)
        return {"flows": flows, "override_qty": self._zero_override, "release_mode": self._zero_release}

    def _fallback(self, observation):
        flows = self._fallback_cap * observation["action_mask"]
        open_now = observation["graph_now.open"]
        seen = observation["graph_now.open.observed"] == 1
        for s, chokepoints in enumerate(self._fallback_through):
            for c in chokepoints:
                if seen[c]:
                    flows[s] *= max(float(open_now[c]), 0.0) ** self._fallback_power
        return flows
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_mpc_agent.py -v -k test_agent`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full test file**

Run: `uv run pytest tests/test_mpc_agent.py -v`
Expected: PASS, all tests from Tasks 1-4.

- [ ] **Step 6: Smoke-test through the real CLI**

Run: `uv run sbf evaluate mpc --quick --task=tiny`
Expected: a score printed, no traceback. (`mpc` resolves because `agents/mpc_lp/agent.py` now exists — `sbf_starter.agents.resolve` takes any folder under `agents/`.)

- [ ] **Step 7: Stop here — do not commit.**

---

### Task 5: Calibration script

**Files:**
- Create: `examples/07_mpc_policy_search.py`

**Interfaces:**
- Consumes: `agents/mpc_lp/agent.py` (via `sbf_starter.scoring`, the same way `06_policy_search.py` consumes `agents/heuristic`).
- Produces: a runnable script, no new importable interface (mirrors `06_policy_search.py`'s shape exactly, so no new design needed here — copy its structure, retarget at `mpc`'s parameters).

- [ ] **Step 1: Write `examples/07_mpc_policy_search.py`**

```python
"""An evolutionary search over agents/mpc's forecast and LP parameters (same method as 06_policy_search.py).

    uv run python examples/07_mpc_policy_search.py
    uv run python examples/07_mpc_policy_search.py --task=small --generations=10 --population=12

A candidate is the mpc agent's agent.py plus a params.json overriding warn_a, warn_b, msg_weight (one number
per message kind 0-4), msg_bump, msg_bump_weeks, holding_rate, closure_power and H. Fitness is its score on your
own root, under the CPU budget. The best is kept only if it beats the start (this session's default PARAMS) on
the held-out dev episodes.
"""

import json
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import fire
import numpy as np

from sbf_starter import scoring
from sbf_starter.agents import resolve


MPC = resolve("mpc") / "agent.py"
# order: warn_a, warn_b, msg_weight[0..4], msg_bump, msg_bump_weeks, holding_rate, closure_power, H
LOWER = np.array([0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
UPPER = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 6.0, 20.0, 3.0, 10.0])


def _load_defaults() -> dict:
    """``agents/mpc_lp/agent.py``'s PARAMS, loaded by path (``agents`` is a loose folder, not an installed
    package, so a dotted ``import agents.mpc.agent`` is not reliable from a script run as ``examples/07_...py``).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("mpc_agent_defaults", MPC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PARAMS


def start_params() -> np.ndarray:
    DEFAULT = _load_defaults()
    weight = DEFAULT["msg_weight"]
    return np.array([
        DEFAULT["warn_a"], DEFAULT["warn_b"],
        weight.get("0", 0.0), weight.get("1", 0.0), weight.get("2", 0.0), weight.get("3", 0.0), weight.get("4", 0.0),
        DEFAULT["msg_bump"], DEFAULT["msg_bump_weeks"], DEFAULT["holding_rate"], DEFAULT["closure_power"],
        DEFAULT["H"],
    ])


def write_candidate(params: np.ndarray, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(MPC, folder / "agent.py")
    clipped = np.clip(params, LOWER, UPPER)
    numbers = {
        "warn_a": round(float(clipped[0]), 4), "warn_b": round(float(clipped[1]), 4),
        "msg_weight": {str(i): round(float(clipped[2 + i]), 4) for i in range(5)},
        "msg_bump": round(float(clipped[7]), 4), "msg_bump_weeks": int(round(clipped[8])),
        "holding_rate": round(float(clipped[9]), 4), "closure_power": round(float(clipped[10]), 4),
        "H": int(round(clipped[11])),
    }
    (folder / "params.json").write_text(json.dumps(numbers) + "\n")
    return folder


def mutate(parents: list, rng: np.random.Generator, n: int, sigma: float = 0.15) -> list:
    out = []
    span = UPPER - LOWER
    for _ in range(n):
        p = parents[rng.integers(len(parents))]
        child = np.clip(p + rng.normal(0.0, sigma, p.shape) * span, LOWER, UPPER)
        out.append(child)
    return out


def main(
    task: str = "tiny",
    entropy: int = 20261004,
    train_episodes: int = 16,
    holdout: str = "dev",
    generations: int = 10,
    population: int = 12,
    elite: int = 3,
    sigma: float = 0.15,
    quick: bool = False,
    n_jobs: int = -1,
    seed: int = 0,
    out: str | None = None,
) -> None:
    if entropy == 0:
        raise ValueError("train on a root of your own (--entropy=...): root 0 holds the dev episodes of the check")
    out = Path(out or f"outputs/07_mpc_policy_search/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    rng = np.random.default_rng(seed)
    train = scoring.episode_set(task, train_episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
    held_out = scoring.episode_set(task, holdout, quick=quick, n_jobs=n_jobs)
    with tempfile.TemporaryDirectory(prefix="sbf-mpc-search-") as tmp:
        work = Path(tmp)

        def fitness(params: np.ndarray, name: str) -> float:
            score = train.score(str(write_candidate(params, work / name)), cpu_budget=True)
            return -float("inf") if score.rss is None else score.rss

        start = start_params()
        archive = [(fitness(start, "g0_start"), start)]
        print(f"{task}: training on {len(train.episodes)} episodes of root {entropy}")
        print(f"generation 0: current defaults, training {scoring.SCALE} {archive[0][0]:.4f}")
        for g in range(1, generations + 1):
            parents = [p for _s, p in sorted(archive, key=lambda x: -x[0])[:elite]]
            children = mutate(parents, rng, population, sigma=sigma)
            scored = [(fitness(c, f"g{g}_{i}"), c) for i, c in enumerate(children)]
            archive += scored
            best_score = max(s for s, _ in archive)
            print(f"generation {g}: best of {population} {max(s for s, _ in scored):.4f}; best so far {best_score:.4f}")
        best_score, best = max(archive, key=lambda x: x[0])
        print(f"the best candidate: training {scoring.SCALE} {best_score:.4f}")
        best_dir = write_candidate(best, work / "best")
        cmp = held_out.compare(str(best_dir), str(write_candidate(start, work / "start")), cpu_budget=True)
        cmp = replace(cmp, a=replace(cmp.a, agent="the best candidate"), b=replace(cmp.b, agent="current defaults"))
        print(f"held out, on {len(held_out.episodes)} dev episodes:\n{cmp}")
        if cmp.diff is not None and cmp.diff > 0:
            shutil.copytree(best_dir, out / "best", dirs_exist_ok=True)
            print(f"written {out / 'best'}: next, uv run sbf check {out / 'best'} --task={task}")
        else:
            print("not written: the best candidate does not beat the current defaults held out")


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 2: Run it**

Run: `uv run python examples/07_mpc_policy_search.py --task=tiny --generations=10 --population=12`
Expected: generation-by-generation output like `06_policy_search.py`'s, ending in either "written outputs/..." or "not written: ...". Either is a valid outcome — this step is a smoke test of the script, not a pass/fail gate on the search's result.

- [ ] **Step 3: Stop here — do not commit.**

---

### Task 6: Validation against `agents/mine` and `agents/heuristic`

**Files:** none (CLI verification only, matching this repo's standard workflow — `AGENTS.md`, "Evaluating reliably").

- [ ] **Step 1: Compliance check**

Run: `uv run sbf check mpc --task=tiny`
Expected: `all checks passed` (imports: only `json`, `pathlib`, `numpy`, `scipy.optimize` — all allowed; CPU budget: the LP for `tiny` is `n_slots * H` ≈ 120 variables plus a similar count of stock/demand variables, trivial for HiGHS).

- [ ] **Step 2: Held-out comparison**

Run:
```
uv run sbf compare mpc agents/mine --task=tiny
uv run sbf compare mpc agents/heuristic --task=tiny
```
Expected: read the printed paired interval exactly as done earlier this session for `06_policy_search.py`'s output — report the numbers, and treat "the interval holds 0" as "not yet distinguishable," not as a pass. If `mpc`'s uncalibrated `PARAMS` defaults lose to `agents/mine`, that is an expected outcome before Task 5's search has been pointed at a real training run (Task 5's smoke test above used few generations; a real calibration run needs more, same as `06_policy_search.py` did earlier this session) — do not treat a loss here as a bug in Tasks 1-4's code unless `sbf check` (Step 1) also fails or the comparison errors out.

- [ ] **Step 3: Report results to the user; stop here — do not commit.**

## Self-review

- **Spec coverage:** §1 forecast (Task 2), §2 LP incl. the stock-tracking refinement pinned down above (Task 3), §3 rolling horizon/fallback/files/calibration (Tasks 4, 5), Testing plan items 1-4 (Tasks 4 Step 6, 5 Step 2, 6 Steps 1-2) — all covered.
- **Placeholders:** none; every step has complete code or an exact command with its expected output.
- **Type/signature consistency:** `build_topology` → `forecast_open` → `slot_capacity` → `build_lp` → `Agent.act` all pass the same `topology` dict and the same key names throughout; checked against each function's "Consumes"/"Produces" line.
- **Known risk, flagged explicitly:** `build_lp`'s index arithmetic (Task 3) is the single most error-prone part of this plan; Task 3 Step 4 already tells the implementer what to inspect if the hand-built scenario test fails, rather than leaving them to guess.
