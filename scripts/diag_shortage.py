"""Diagnostic: where mpc_det's shortage exceeds the clairvoyant LP's (per demand, fab starts, energy). Training only."""

import fire
import numpy as np
from shockbench_flow.dynamics.env import rollout
from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, solve_oracle
from shockbench_flow.policies.registry import PolicyContext, make_policy
from shockbench_flow_agent import EpisodeSet
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _policy_seed, _world


def main(task="small", n=0, entropy=0, policy="mpc_det"):
    es = EpisodeSet.build(task, "dev", entropy=entropy)
    spec = es._spec
    inst, omega, marks, fallback = _world(task, entropy, n, spec[3], spec[4])
    model = build_lp(inst, marks)
    x = solve_oracle(model, method=ORACLE_METHOD).x
    traj = rollout(inst, make_policy(policy, PolicyContext()), omega, "standard",
                   _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    T = inst.T
    idx = model.index

    def col(key):
        j = idx.get(key)
        return 0.0 if j is None else float(x[j])

    print("demands:")
    for d, dem in enumerate(inst.demands):
        node = inst.nodes[dem.node]
        o_short = sum(col(("U", t, d)) + col(("B", t, d)) for t in range(1, T + 1))
        m_short = sum(float(r.lost[d] + r.backlog[d]) for r in traj.records)
        o_served = sum(col(("D", t, d)) for t in range(1, T + 1))
        m_served = sum(float(r.served[d]) for r in traj.records)
        dem_tot = sum(float(r.demand[d]) for r in traj.records)
        print(f"  d{d} {node.id} k={inst.commodities[dem.k].id} pi={dem.pi:.0f} demand {dem_tot:.0f} "
              f"served {o_served:.0f}/{m_served:.0f} short-units {o_short:.0f}/{m_short:.0f} "
              f"cost {dem.pi * o_short / 1e9:.0f}/{dem.pi * m_short / 1e9:.0f}bn")
    print("fab lots started per week (oracle | mpc):")
    for fi, f in enumerate(inst.fabs):
        o = [round(col(("p", t, fi))) for t in range(1, T + 1)]
        m = [round(float(r.lots_started[fi])) for r in traj.records]
        print(f"  {inst.nodes[f].id}: total {sum(o)} | {sum(m)}")
        print(f"     o {o}")
        print(f"     m {m}")
    for gi, g in enumerate(inst.grids):
        grid = inst.nodes[g].grid
        fabs = [inst.nodes[inst.fabs[fi]].id for fi in inst.grid_fabs[gi]]
        print(f"grid {inst.nodes[g].id}: priority {grid.priority}, fuels {[inst.commodities[k].id for k in grid.fuels]}"
              f", rationed {grid.rationed}, fabs {fabs}")
    print("fab input stock per week (oracle | mpc):")
    for fi, f in enumerate(inst.fabs):
        fab = inst.nodes[f].fab
        s = inst.slot_index[(f, fab.input)]
        o = [round(col(("I", t, s))) for t in range(1, T + 1)]
        m = [round(float(r.stock[s])) for r in traj.records]
        print(f"  {inst.nodes[f].id} cap0 {fab.cap0} e {fab.e} tau {fab.tau}: o {o[:40]}")
        print(f"  {' ' * len(inst.nodes[f].id)}   m {m[:40]}")
    print("energy to fabs per week (oracle | mpc):")
    for fi, f in enumerate(inst.fabs):
        o = [round(col(("E", t, fi))) for t in range(1, T + 1)]
        m = [round(float(r.energy[fi])) for r in traj.records]
        print(f"  {inst.nodes[f].id}: total {sum(o)} | {sum(m)}")
    print("OSAT starts per week (oracle | mpc):")
    for oi, o_node in enumerate(inst.osats):
        for pk in inst.nodes[o_node].osat.packages.values():
            o = sum(col(("xi", t, oi, pk)) for t in range(1, T + 1))
            m = sum(float(r.packaged.get((oi, pk), 0.0)) for r in traj.records)
            print(f"  {inst.nodes[o_node].id} k={inst.commodities[pk].id}: total {o:.0f} | {m:.0f}")


if __name__ == "__main__":
    fire.Fire(main)
