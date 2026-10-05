# OpenEvolve (`agents/evo`)

Evolved shipping rule. `Signals` turns each week's observation into arrays lined up with the flow slots, so `act` can use closures, sanctions, tariffs, warnings and demand without parsing the padded lists.

OpenEvolve may rewrite only the block between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` in `evolve/initial_agent.py`. Imports inside the submission are the standard library and numpy.

The search lives in `evolve/` (`initial_agent.py`, `evaluator.py`, `config.yaml`) and is started by `examples/08_openevolve_agent.py`. This folder is the submission: score it with `sbf evaluate evo`. Copy a champion from `outputs/08_openevolve_agent/` here when it should be the one that is packed.

## What `act` does

1. Read this week's signals.
2. A route with a sanction or a tariff inside `HORIZON` weeks is urgent. `RUSH` scales that urgency.
3. Room is this week's capacity, times `action_mask`, times the lowest open fraction on the route's straits.
4. `RISK` shrinks room when warnings and threats are high. Urgent routes keep more of their room.
5. `PREBUILD` lets an urgent route carry extra weeks of its sink's need. The extra shrinks as the episode runs out of weeks.
6. Routes that deliver to the same sink share one need (`demand_of_slot`). Routes that are about to close are filled first, earliest disruption first. The remaining budget is split across the calmer routes in proportion to `remaining room * calm`.

The return value is `{"flows": ...}`. Non-finite entries are replaced with 0.

## Signals

`self.sig.read(observation)` returns a dict. Each array is one value per action slot, unless the name says otherwise.

| Key | Meaning |
| --- | --- |
| `week`, `weeks_left` | Current week and weeks still to play, including this one. |
| `mask` | 1 on routes allowed this week. |
| `cap_now` | Capacity this week: observed `graph_now.u`, otherwise the nominal capacity. |
| `open_strait` | Open fraction of each strait. Unobserved straits count as open. |
| `open` | Lowest open fraction among the straits on the route. |
| `threat_strait`, `warning_strait` | Military-threat count and early-warning score, per strait. |
| `threat` | Worst strait threat on the route, plus threats aimed at the route's edges. |
| `warning` | Highest early-warning score on the route's straits. |
| `ban_in` | Weeks until an announced sanction hits the route. `inf` when none is pending. |
| `tariff` | Highest current tariff on the route's edges for this good. |
| `tariff_in` | Weeks until an announced tariff hits the route. `inf` when none is announced. |
| `need` | This week's forecast plus backlog for the sink this route delivers to. `inf` when the route does not end at a demand. |
| `stock`, `backlog`, `forecast` | Raw arrays. Their lengths differ from `flows`. |
| `clip_requested`, `clip_executed`, `costs` | Last week's clipped quantities and cost components. |

`self.sig.demand_of_slot[slot]` is the sink index, or `-1` when the route does not end at a demand. Slots with the same index share one `need`.

`forecast` is `(demands, horizon)`, `backlog` is `(demands,)`, and `stock` and `costs` have their own lengths. Combine them with `flows` only after indexing down to a per-slot array such as `need`. A shape mismatch raises, the naive rule plays the week, and the score for that week is 0.

## Knobs

Declared at the top of the evolve block.

| Name | Default | Meaning |
| --- | --- | --- |
| `RUSH` | `1.0` | How far an imminent ban or tariff protects the route from the risk cut, and how large the prebuild is. |
| `RISK` | `0.45` | How fast warnings and threats shrink the flow. |
| `HORIZON` | `5` | A sanction or tariff inside this many weeks counts as urgent. |
| `PREBUILD` | `3.0` | Extra weeks of need an urgent route may ship. |

## Run

```bash
uv run sbf evaluate evo --task=small
uv sync --extra evolve
uv run python examples/08_openevolve_agent.py
```

The search reads `OPENAI_API_KEY`, rewrites `evolve/initial_agent.py`, and writes the best program under `outputs/08_openevolve_agent/`. Copy that champion here when it should be the submission.
