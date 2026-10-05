# mpc2

Rolling-horizon linear program that lets observed impairments end.

Each week the flat observation is decoded into the protocol's lists, and the planner solves window LPs with SciPy's HiGHS. Week 1 of the plan is returned as `flows`, `override_qty` and `release_mode`. Flows on sanctioned routes are zeroed by `action_mask`. A failed solve falls back to the naive rule.

`agents/mpc` holds every impairment seen this week for the whole window. This agent ends each one after a quantile of its residual duration under the public generator's duration law, conditioned on how long it has already lasted. Several quantiles are a two-stage sample-average approximation: the scenarios differ after week 1 and share the week-1 action. Announced closure ends, warnings, threats and tariff notices are applied on top of every window.

The planner ships beside `agent.py` as `sbfv`, so the zip is self-contained. On the server the allowed imports are the standard library, numpy, SciPy and PyTorch (CPU).

## Residual windows

For each quantile `q` in `quantiles`:

1. Start from the nominal future. If `future_draws` is greater than 0, start from a public-generator draw of future onsets instead.
2. Every impairment present now is kept at its observed severity for a residual life `residual_duration(..., q, age)`. That covers a closure, a capacity loss, a prohibition, a tariff, a war-risk class, a fab or OSAT rate, a grid, a supply cap and a freight surcharge.
3. Age is how many weeks the element has already been impaired. When the start was seen, the residual is the conditional survival `S(age + r) / S(age)`. When it was not (present at week 1, or first seen after a masked week), the residual is the length-biased life.

An empty `quantiles` list keeps impairments for the whole window, which is the persistence forecast used by `mpc`.

`future_draws` 0, the default, adds no future onsets. Pending prohibitions still switch on at their effective week.

## Early signals

Applied after the residual layer, on each window:

| Key | Default in `params.json` | Effect |
| --- | --- | --- |
| `closure_end` | `true` | A strait with a known end week reopens from that week, and its throughput is restored. |
| `closure_hold` | `true` | Until that week, keep the open fraction observed now. |
| `warn_gain` | `0.0` | Cut a strait's open fraction and throughput by `warn_gain * warning score`. |
| `warn_lag` | `1` | First window week a warning cut applies. |
| `warn_weeks` | `8` | How long the warning cut lasts. `0` means the rest of the window. |
| `threat_gain` | `0.0` | The same cut, once per live military threat that names the strait. |
| `threat_weeks` | `12` | How long a threat cut lasts. `0` means the rest of the window. |
| `demand_scale` | `1.0` | Plan for this multiple of the forecast. |
| `throughput_scale` | `1.0` | Plan for this multiple of the observed strait throughput. |
| `tariff_rate` | `0.0` | From an announced tariff's stated week, the planned rate is at least this. `0` leaves notices unused. |
| `tariff_proposal_scale` | `0.4` | Notices that are not a final tariff, and messages whose kind is a proposal, use this fraction of `tariff_rate`. A non-final proposal is scaled twice. |

`warn_gain`, `threat_gain` and `tariff_rate` are 0 in `params.json` because those signals are noisy on Small. Residual duration carries the forecast there.

## Base load

A `base_first` grid in the simulator never sheds base load in order to power a fab. The relaxed LP can plan that. Two guards stop it:

- `fab_energy_cap`: in every window week, fabs on a `base_first` grid get at most `max(0, G_bar - y_bar)`.
- `base_first_fix`: after the solve, any week that both sheds base load and powers a fab is re-solved. The first pass prices that shed, so the fabs must run on energy beyond the base load. If the shed remains, those fabs are switched off for that week. At most `bf_passes` re-solves. Week 1 is read from the last successful solution.

## CPU

`cpu_limit` is this agent's own guard. The board still meters its own budget: 2 seconds per week on Small, 4 on Full. A week over the board's budget is played by the naive rule. `Agent(config)` counts toward week 1.

On Small and Full the first week has no previous timing, so it is treated as tight: extra quantiles collapse to the one nearest 0.5, and `bf_passes` drops to 1. Later weeks do the same when the previous week used more than `0.55 * cpu_limit` seconds. Tiny (`T <= 30`) always keeps the full quantile set and the full pass count, because its LP is cheap.

## Parameters

`params.json` is merged over the defaults in `agent.py` at import. On Tiny (`config["T"] <= 30`) `TINY_OVERRIDES` then replace the Small-tuned numbers. Passing `params=` into `Agent` merges onto the defaults only: the file and the Tiny overrides are both skipped.

Shipped `params.json`, used on Small and Full:

| Key | Value | Meaning |
| --- | --- | --- |
| `H_extra` | `8` | Weeks added to the lead-time window `L`. |
| `quantiles` | `[0.55]` | Residual-duration quantile. One scenario. |
| `future_draws` | `0` | Public-generator future onsets. |
| `demand_scale` | `1.0` | Multiple of forecast demand. |
| `throughput_scale` | `1.0` | Multiple of observed strait throughput. |
| `closure_end` | `true` | Reopen at the announced end week. |
| `closure_hold` | `true` | Keep the observed open fraction until that week. |
| `warn_gain` | `0.0` | Warning cut. `0` trusts residual duration. |
| `warn_lag` | `1` | First window week a warning cut applies. |
| `warn_weeks` | `8` | How long a warning cut lasts. |
| `threat_gain` | `0.0` | Military-threat cut. |
| `threat_weeks` | `12` | How long a threat cut lasts. |
| `tariff_rate` | `0.0` | Announced-tariff rate. |
| `tariff_proposal_scale` | `0.4` | Fraction of `tariff_rate` for proposals and non-final notices. |
| `fab_energy_cap` | `true` | Cap fab energy by grid surplus every window week. |
| `base_first_fix` | `true` | Re-solve weeks that shed base load to power a fab. |
| `bf_passes` | `1` | Re-solves per week. |
| `cpu_limit` | `1.2` | Seconds. Above `0.55` of this, drop extra scenarios and passes. |

Tiny overrides, applied when `T <= 30`: `quantiles` `[0.4, 0.7]`, `demand_scale` `1.05`, `warn_gain` `0.35`, `warn_weeks` `8`, `threat_gain` `0.5`, `threat_weeks` `16`, `tariff_rate` `0.15`, `tariff_proposal_scale` `0.4`, `closure_hold` `true`, `base_first_fix` `true`, `bf_passes` `2`, `cpu_limit` `1.55`.

## Layout

- `agent.py` — `Decoder`, residual windows, early signals, the base-first re-solve, and the flat action.
- `params.json` — the Small and Full numbers above.
- `sbfv/policies/scenarios.py` — residual life, impairment ages, public-generator draws.
- `sbfv/policies/mpc_det.py`, `sbfv/policies/lp_common.py`, `sbfv/oracle/lp.py` — the window LP and the HiGHS session.
- `sbfv/disruption/` — duration laws the residual life is taken from (`laws.py`, `events.py`, `profiles.py`).
- `sbfv/marks.py` — how overlapping events become open fractions and the other marks.
- `sbfv/omega/` — the public generator's event container.

## Run

```bash
uv run sbf evaluate mpc2
uv run sbf compare mpc2 mpc --task=small
uv run sbf check mpc2 --task=small
```
