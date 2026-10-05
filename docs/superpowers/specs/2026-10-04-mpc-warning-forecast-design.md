# MPC agent with closure forecast from warning.*/messages.* — design

Status: approved (sections 1-3), Stage 1 scope only.

## Goal

A new submission agent (`agents/mpc_lp/`) that plans `flows` over a rolling
horizon with a linear program (`scipy.optimize.linprog`, HiGHS), instead of
the single-week heuristic rule. The LP's forecast of each chokepoint's future
openness is built from `warning.score` and `messages.*` (not just persistence
of this week's `graph_now.open`), which is the part the naive rule and the
plain heuristic cannot do.

Stage 1 (this spec) deliberately excludes: war-risk surcharge, disposal/shed
costs, queue-lot/release-mode mechanics, and fab/OSAT/grid production
modeling — all out of scope because the real formulas live in the training-
only `oracle/lp.py` (922 lines) and are not published. Stage 2 (future, not
this spec) can add them incrementally, each change validated with
`sbf compare` before being kept.

## Scope / non-goals

- Target network for development: `tiny`. Move to `small` only after a Stage
  1 agent beats the current `agents/mine` heuristic on `tiny`'s held-out dev
  episodes.
- The LP controls `flows` only. `override_qty`/`release_mode` stay at their
  default (0), same as every other agent in this repo so far.
- No modeling of fab/OSAT/grid production transformation. Chokepoints and
  intermediate nodes are pass-through within a lane; a lane's total lead time
  is the sum of `graph_now.tau` over its edges.
- Tariff and freight cost are persisted from `graph_now.*` over the horizon
  (not forecast); only chokepoint openness is forecast from warning/messages.

## 1. Forecast module

For each chokepoint `c` (from `config["layout"]["chokepoints"]`) and each
horizon step `h = 1 .. H-1` (h=0 is the current, already-observed week):

- Locate `c`'s row in `warning.score` by name-matching
  `config["layout"]["warning_units"]` for the entry `"chokepoint " + name`
  (never a hardcoded index — this differs by network).
- `p_warn[c] = sigmoid(a * warning.score[idx] - b)`, `a`/`b` tunable.
- Scan `messages.*` for `target_kind == chokepoint`, `target == c`, channel in
  `{sanction_legal, ties_threat, mid_threat}` (closure-relevant; tariff
  channels are excluded — they would affect tariffs, out of scope here).
  - If `stated_effective_week` is observed and falls in `[t, t+H-1]`: add
    `weight[kind]` (one tunable weight per message `kind`) to that week's
    closure probability.
  - If not observed: add a flat small bump to the next few weeks (one more
    tunable constant).
- `p_close[c, h] = clip(p_warn[c] + p_msg[c, h], 0, 1)`.
- `open_forecast[c, h] = open_now[c] * (1 - p_close[c, h])`, `open_forecast[c,
  0] = graph_now.open[c]` (observed, no forecasting needed).

## 2. LP formulation

Decision variables `x[s, h] >= 0` for each action slot `s`, `h = 0 .. H-1`.
Only `x[:, 0]` is returned as this week's `flows`; the rest is discarded and
rebuilt next week (no warm start in Stage 1).

- Capacity: `x[s, h] <= graph_now.u[e] * open_forecast[c, h] ** closure_power`
  for each chokepoint `c` on `s`'s lane (product over multiple chokepoints on
  the same lane); `0` wherever `action_mask`/`pending_prohibitions` forbids
  `(e, k)` at week `t + h`.
- Lead time: a unit sent on slot `s` at step `h` arrives at the lane's
  destination at step `h + tau_total[s]` (`tau_total` = sum of
  `graph_now.tau` over the lane's edges, persisted). Already-in-transit
  `pipeline.*` entries are added as known arrivals at their `arrival_week`.
- Stock balance per `stock_slot` node/commodity: `stock[n, k, h+1] =
  stock[n, k, h] + arrivals(n, k, h) - departures(n, k, h)`.
- Demand: `served[h] <= demand_forecast[h] + backlog[h]` (demand_forecast
  beyond its published 8 weeks persists the last value); `backlog[h+1] =
  backlog[h] + demand[h] - served[h]` if `sinks.backlog` else no carry.
- Objective (minimize, summed over `h`): freight (`graph_now.c[e] * x`) +
  tariff (`graph_now.tariff[e,k] * commodities.v[k] * x`) + holding
  (`holding_rate * stock[n,k,h]`, `holding_rate` a new tunable constant — the
  real rate is not published) + shortage (`sinks.pi * unmet[h]`, `unmet[h] =
  demand[h] - served_from_this_week[h] >= 0` a slack variable charged once,
  the week demand first goes unserved; a carried `backlog` that stays
  unserved in later weeks is not charged again — an approximation, since the
  real per-week-until-served formula is not published).
- Solve with `scipy.optimize.linprog(method="highs")`; take `x[:, 0]` as the
  action.

## 3. Rolling horizon integration, fallback, files, calibration

- `agents/mpc_lp/agent.py`: a new, self-contained submission folder (does not
  touch `agents/mine`). Forecast + LP + fallback logic all in this one file
  (multi-file imports inside a submission zip are untested in this repo; not
  worth the risk for Stage 1).
- Every week: rebuild the LP fresh from the current observation (no
  cross-week LP state). `H` is a tunable constant in `params.json` (default
  ~6 for tiny).
- Fallback: if `linprog` does not report success (infeasible, numerical
  failure, anything but optimal), do not raise — fall back to the current
  `agents/mine` heuristic's rule (persist + `closure_power` derate) for that
  week, so a bad solve degrades to a known-OK policy instead of the server's
  naive-rule crash fallback.
- Calibration: `examples/07_mpc_policy_search.py`, a copy of
  `06_policy_search.py`'s search loop (Gaussian mutation + elite, trained on
  a root of our own, kept only if it beats the start on held-out dev episodes)
  over the new parameters: `a`, `b`, per-`kind` message weights, the flat
  message bump, `holding_rate`, `closure_power`, `H`.
- Validation: `sbf compare mpc agents/mine` and `sbf compare mpc
  agents/heuristic` on `tiny`'s dev episodes before considering Stage 1 done;
  accepted only if the paired interval does not hold 0 in `mpc`'s favor (or,
  at minimum, is never worse, consistent with how `agents/mine`'s params were
  judged earlier this session).

## Testing plan

1. `uv run sbf evaluate mpc --quick --task=tiny` — smoke test (loads, runs,
   no crash).
2. `uv run python examples/07_mpc_policy_search.py --task=tiny` — calibrate.
3. `uv run sbf compare mpc agents/mine --task=tiny` and `... agents/heuristic
   --task=tiny` — held-out validation (the real bar for "done").
4. `uv run sbf check mpc --task=tiny` — CPU budget and import compliance.
