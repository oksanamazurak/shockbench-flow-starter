# mpc

Rolling-horizon linear program behind the flat Dict interface.

Each week `Decoder` turns the observation back into the protocol's lists (stock, pipeline, queues, WIP, backlog, the demand forecast, pending sanctions, this week's graph). `SignalMpc`, a subclass of the package's `mpc_det`, solves one window LP with SciPy's HiGHS. The forecast is persistence: every impairment observed this week (a closed strait, a tariff, a lost factory, an energy shock) is held for the whole window. Week 1 of that plan becomes `flows`, `override_qty` and `release_mode`.

A failed solve is played by the naive rule inside the planner.

The planner ships beside `agent.py` as `sbfv`, so the zip is self-contained. On the server the allowed imports are the standard library, numpy, SciPy and PyTorch (CPU). HiGHS comes with SciPy.

## Window

The horizon is the package's canonical lead-time window `L`, plus `H_extra` weeks. `H_extra` 0 is the package's `mpc_det`.

Persistence is then adjusted by the early signals whose gains are non-zero:

| Key | Default | Effect |
| --- | --- | --- |
| `H_extra` | `0` | Weeks added to `L`. |
| `demand_scale` | `1.0` | Plan for this multiple of the forecast. |
| `throughput_scale` | `1.0` | Plan for this multiple of the observed strait throughput. |
| `closure_end` | `false` | From an announced end week, reopen that strait and restore its nominal throughput. |
| `warn_gain` | `0.0` | From window week `warn_lag`, multiply a strait's planned open fraction by `1 - warn_gain * warning score`. |
| `warn_lag` | `1` | First window week the warning cut applies to. |
| `threat_gain` | `0.0` | Each live military threat that names a strait multiplies its open fraction by `1 - threat_gain`. |
| `tariff_rate` | `0.0` | From an announced tariff's stated week, the planned rate on that edge is at least this. `0` leaves announcements unused. |

The shipped defaults leave the gains at 0 and `closure_end` off, so the window is plain persistence. A `params.json` beside `agent.py` is merged over these defaults at import time. `Agent(..., params={...})` merges onto the same defaults and ignores the file.

## Action

`flows[slot]` is the planned quantity on that route. `release_mode` is 0 for the simulator's default queue release, 1 when the matching `override_qty` should be released, and 2 when the (strait, good) pair is held. The planner's `overrides` set mode 1; its `hold` list sets mode 2.

## Layout

- `agent.py` — `Decoder`, `SignalMpc`, `Agent`.
- `sbfv/policies/mpc_det.py` — the rolling planner this agent subclasses.
- `sbfv/policies/lp_common.py` — persistence arrays, the window LP, week-1 readout.
- `sbfv/oracle/lp.py` — the LP builder (HiGHS).
- `sbfv/policies/naive.py` — the fallback when every solve rung fails.

The rest of `sbfv/` is the instance, dynamics and scoring code those modules import.

## Run

```bash
uv run sbf evaluate mpc
uv run sbf check mpc --task=small
```

The public board allows 2 seconds per week on Small and 4 on Full. `Agent(config)` counts toward week 1. A week over budget is played by the naive rule.
