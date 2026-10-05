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
    # fab_tw/chip_le_raw (node 7, k 2): fab_tw's production output -> tracked, fed by a production variable
    assert (7, 2) in topo["tracked"]
    assert (7, 2) not in topo["cumulative_cap"]
    fab_tw = topo["procs"][topo["proc_out"][(7, 2)][0]]
    assert (fab_tw["k_in"], fab_tw["k_out"], fab_tw["tau"]) == (1, 2, 8)
    # osat_sea: one package chip_le_raw -> chip_le, lead time 2, its own capacity group
    osat = topo["procs"][topo["proc_out"][(10, 3)][0]]
    assert (osat["k_in"], osat["k_out"], osat["tau"]) == (2, 3, 2)
    assert topo["cap_groups"][osat["group"]] == ("osat", 0)
    # real costs from static["instance"]: holding per (node, k), disposal per k (0.1 v_k on tiny)
    assert topo["holding_cost"][(7, 2)] == pytest.approx(16.192307692307693)
    assert topo["disposal_cost"][3] == pytest.approx(2000.0)
    # fab_tw/wafer (node 7, k 1): we ship wafer INTO fab_tw (slots 7-12) but never out -> tracked, no outflow
    assert (7, 1) in topo["tracked"]
    assert topo["outgoing"].get((7, 1), []) == []
    # the demand sink (sink_us/chip_le, node 11, k 3) is tracked and listed in demands
    assert (11, 3) in topo["tracked"]
    assert topo["demands"][topo["demand_index"][(11, 3)]]["pi"] == pytest.approx(50360.0)
    assert topo["demands"][topo["demand_index"][(11, 3)]]["backlog"] is False


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
    params["warn_a"], params["warn_b"] = 0.0, 50.0  # isolate the message signal: p_warn ~= sigmoid(-50) ~= 0
    forecast = mpc.forecast_open(obs, topo, H=5, params=params)
    arr = forecast[3]
    # week 5 + h = 7 -> h = 2; weight 0.4 -> open_forecast = 1.0 * (1 - 0.4) = 0.6
    assert arr[2] == pytest.approx(0.6)
    assert arr[0] == pytest.approx(1.0)  # h=0 is the observed week, untouched


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
        "n_slots": 1,
        "slot_edges": [[0]],
        "slot_dest": [1],
        "slot_tail": [0],
        "slot_chokepoints": [[]],
        "slot_k": [0],
        "chokepoint_pos": {},
        "chokepoint_warn_row": {},
        "stock_index": {(0, 0): 0, (1, 0): 1},
        "supply_list": [(0, 0)],
        "supply_set": {(0, 0)},
        "demands": [],
        "demand_index": {},
        "demand_set": set(),
        "outgoing": {(0, 0): [0]},
        "incoming": {(1, 0): [0]},
        "tracked": [(0, 0), (1, 0)],
        "cumulative_cap": [],
        "edges_head": [1],
        "holding_cost": {},
        "storage": {},
        "fabs": [],
        "osats": [],
        "fab_input_at": {},
        "fab_output_at": {},
        "osat_input_at": {},
        "osat_output_at": {},
        "grid_set": set(),
        "grid_fuels": set(),
        "grids": [],
        "psi": 0.0,
        "procs": [],
        "cap_groups": [],
        "proc_in": {},
        "proc_out": {},
        "disposal_cost": [0.0],
        "supply_nodes": {0},
        "chokepoint_nodes": set(),
    }
    obs = {
        "week": np.array([1]),
        "graph_now.u": np.array([10.0]),
        "graph_now.tau": np.array([0]),
        "graph_now.c": np.array([2.0]),
        "graph_now.tariff": np.array([[0.0]]),
        "graph_now.prohibited": np.zeros((1, 1)),
        "graph_now.open": np.array([]),
        "action_mask": np.array([1]),
        "pending_prohibitions.edge": np.array([0]),
        "pending_prohibitions.effective_week": np.array([0]),
        "pending_prohibitions.edge.observed": np.array([0]),
        "stock.qty": np.array([100.0, 0.0]),
        "graph_now.supply.avail": np.array([0.0]),
        "graph_now.fab.cap_eff": np.zeros(0),
        "graph_now.osat.thr_eff": np.zeros(0),
        "pipeline.edge": np.array([0]),
        "pipeline.k": np.array([0]),
        "pipeline.qty": np.array([0.0]),
        "pipeline.arrival_week": np.array([0]),
        "pipeline.qty.observed": np.array([0]),
        "messages.msg_id.observed": np.zeros(1),
        "messages.channel": np.zeros(1),
        "messages.kind": np.zeros(1),
        "messages.target_kind": np.zeros(1),
        "messages.target": np.zeros(1),
        "messages.stated_effective_week": np.zeros(1),
        "messages.stated_effective_week.observed": np.zeros(1),
        "demand_forecast.qty": np.zeros((0, 8)),
        "backlog.qty": np.zeros(0),
    }
    params = dict(mpc.PARAMS, H=2, tariff_bump=0.0)
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


def test_agent_act_returns_valid_flows(tiny_episode):
    config, obs = tiny_episode
    agent = mpc.Agent(config)
    action = agent.act(obs)
    flows = action["flows"]
    assert flows.shape == (20,)
    assert np.all(flows >= -1e-9)
    assert np.all(flows[obs["action_mask"] == 0] <= 1e-9)
    # slots whose lane does not reach the demand sink fall back to the heuristic rule (nominal static u0,
    # not the live graph_now.u -- same as agents/heuristic; over-ordering is safe, the server clips it, per
    # AGENTS.md). Only demand-reaching slots are LP-bounded by the live, forecast-derated capacity.
    u0 = config["static"]["edges"]["u0"]
    demand_set = agent.topology["demand_set"]
    for s in range(20):
        e0 = agent.topology["slot_edges"][s][0]
        dest_k = (agent.topology["slot_dest"][s], agent.topology["slot_k"][s])
        if dest_k in demand_set:
            assert flows[s] <= float(obs["graph_now.u"][e0]) + 1e-6
        else:
            assert flows[s] <= float(u0[e0]) + 1e-6


def test_agent_falls_back_on_solver_failure(tiny_episode, monkeypatch):
    config, obs = tiny_episode
    agent = mpc.Agent(config)

    class _FailedResult:
        success = False

    monkeypatch.setattr(mpc, "linprog", lambda *a, **k: _FailedResult())
    action = agent.act(obs)
    assert action["flows"].shape == (20,)
    assert np.all(np.isfinite(action["flows"]))


def test_tariff_forecast_bumps_named_edge_from_effective_week():
    obs = {
        "week": np.array([4]),
        "graph_now.tariff": np.full((2, 1), 0.1),
        "messages.msg_id.observed": np.array([1, 0]),
        "messages.channel": np.array([2, 0]),  # tariff_final
        "messages.target_kind": np.array([1, 0]),  # edge
        "messages.target": np.array([1, 0]),
        "messages.k": np.array([0, 0]),
        "messages.k.observed": np.array([1, 0]),
        "messages.stated_effective_week": np.array([6, 0]),
        "messages.stated_effective_week.observed": np.array([1, 0]),
    }
    topo = {"edges_head": [1, 2], "edges_tail": [0, 1], "node_region": [0, 0, 0]}
    params = dict(mpc.PARAMS, tariff_bump=0.2, tariff_weight={"2": 1.0})
    tf = mpc.tariff_forecast(obs, topo, H=4, params=params)
    assert tf.shape == (2, 1, 4)
    assert list(tf[0, 0]) == pytest.approx([0.1] * 4)  # edge 0 untouched
    assert list(tf[1, 0]) == pytest.approx([0.1, 0.1, 0.3, 0.3])  # week 6 = h 2 onward
    # bump 0 (the default): the persisted rate, unchanged
    assert np.allclose(mpc.tariff_forecast(obs, topo, H=4, params=dict(mpc.PARAMS, tariff_bump=0.0)), 0.1)
