# Архітектура `agents/mpc_lp`

Агент — **MPC (model predictive control) з rolling horizon**: щотижня будує прогноз на H тижнів,
розв'язує лінійну задачу (LP) на цьому горизонті, виконує лише рішення тижня 0, наступного
тижня — усе заново. Там, де LP програє простому правилу (паливо для електростанцій), рішення
приймає просте правило.

Весь код — в одному файлі `agents/mpc_lp/agent.py` (сервер вимагає `agent.py` у корені zip; модулі
поруч не перевірялись). Залежності: stdlib, numpy, `scipy.optimize.linprog` (HiGHS).

## 1. Ініціалізація: `Agent.__init__(config)`

Один раз на епізод.

- `build_topology(static, layout)` — статична форма мережі (див. §2).
- `commodities_v` — ціна товару `v_k` (для тарифу ad valorem і terminal value).
- Значення, залежні від мережі, вибираються за `static["instance"]["kind"]` (`"tiny"`, `"small"`,
  `"full"`): `self._upstream_floor`, `self._fuel_mult`.
- Структури для простого правила (`_fallback`) і для `_reroute` (пари протока–товар, їхні
  override-слоти й протоки попереду на кожному лейні).

`PARAMS` — словник за замовчуванням у коді; `params.json` поруч із `agent.py` перекриває його
(`PARAMS |= json.load(...)` при імпорті).

## 2. Топологія: `build_topology`

Обчислюється один раз. Ключові поля результату:

| Поле | Зміст |
| --- | --- |
| `slot_edges[s]` | ланцюг ребер слота: весь лейн (`lanes.edges`) або одне ребро |
| `slot_tail[s]`, `slot_dest[s]`, `slot_k[s]` | вузол відправлення (хвіст першого ребра), вузол призначення (голова останнього), товар |
| `slot_chokepoints[s]` | протоки на лейні слота |
| `chokepoint_pos[node]` | рядок протоки в `graph_now.open` |
| `chokepoint_warn_row[node]` | рядок протоки в `warning.score` (пошук за `layout["warning_units"]`, де записи — `[kind, index]`, а не хардкод) |
| `stock_index[(node, k)]` | рядок у `stock.qty` |
| `supply_list` | `(node, k)` джерел (`layout["supply_slots"]`) |
| `demands`, `demand_index`, `demand_set` | sink-и: вузол, товар, `pi` (штраф за нестачу), `backlog` |
| `outgoing[(node, k)]`, `incoming[(node, k)]` | слоти, що виходять з вузла / приходять у вузол з цим товаром |
| `holding_cost[(node, k)]`, `storage[(node, k)]` | з `static["instance"]["nodes"][...]["stock"]` |
| `disposal_cost[k]` | з `static["instance"]["commodities"]` (на tiny 0.1 · `v_k`) |
| `procs` | процеси виробництва: fab (1 на fab) і OSAT (1 на package); `node`, `k_in`, `k_out`, `tau`, `group` |
| `cap_groups` | групи спільної потужності: `("fab", row)` / `("osat", row)` — рядок у `graph_now.fab.cap_eff` / `graph_now.osat.thr_eff`; packages одного OSAT ділять одну групу |
| `proc_in`, `proc_out` | `(node, k)` → процеси, що споживають / виробляють цей товар у цьому вузлі |
| `supply_nodes`, `chokepoint_nodes`, `grid_set` | множини вузлів |
| `grid_fuels` | товари, які grid тримають на складі (tiny: lng; small: lng, crude, nucfuel) |
| `grids`, `psi` | для вимкненої моделі grid: частки палив, частка `unmodelled`, VOLL, раціоноване паливо, `ibar`; `psi` з `instance.params` |
| `tracked` | `(node, k)`, для яких LP веде змінну запасу: джерело, призначення слота, вхід або вихід процесу |
| `cumulative_cap` | вихід без моделі притоку (на tiny і small порожній); **не обмежується** — обмеження сумарного вивозу поточним запасом виявилось шкідливим |

## 3. Тиждень: `Agent.act(observation)`

Сім кроків (дублюються в docstring модуля й маркерами в коді).

### Крок 1. Прогноз

**Закриття проток — `forecast_open`.** Для кожної протоки c і тижня h = 0..H−1:

- `p_warn = sigmoid(warn_a · score_c − warn_b)`, `score_c` — рядок протоки в `warning.score`;
- `p_msg[h]` — від живих повідомлень (`messages.msg_id.observed`) з `target_kind == 0` (протока),
  `target == c` і каналом із `CLOSURE_CHANNELS = (3, 4, 5)` (sanction_legal, ties_threat,
  mid_threat):
  - з `stated_effective_week`: `+ msg_weight[kind]` у тиждень набрання чинності;
  - без нього: `+ msg_bump · msg_weight[kind]` на перші `msg_bump_weeks` тижнів;
  - обмеження зверху 1;
- `p_close = 1 − (1 − p_warn)(1 − p_msg)` (noisy-OR, знайдено OpenEvolve);
- `open[c, h] = open_now[c] · (1 − p_close[h])`, а `open[c, 0] = open_now[c]` (спостережене).

**Тарифи — `tariff_forecast`.** Матриця `(edge, k, h)`: поточний `graph_now.tariff`, persisted. Для
кожного живого повідомлення з каналом `TARIFF_CHANNELS = (0, 1, 2)` і відомим
`stated_effective_week` додається `tariff_bump · tariff_weight[channel]` від тижня набрання чинності
на ребра цілі: ребро; ребра, що входять у вузол або виходять з нього; або імпорт у регіон (голова ребра в
регіоні, хвіст — поза ним). Товар — з `messages.k`, якщо він спостережений, інакше всі. На практиці всі
tariff-повідомлення адресовані регіону.

### Крок 2. Capacity слотів — `slot_capacity`

`cap[s, h]` = `min(graph_now.u[e] for e in chain)` (persisted) × `∏ open[c, h] ** closure_power` по
протоках лейна; 0, якщо будь-яке ребро ланцюга має `graph_now.prohibited[e, k]`, або від тижня, коли
набирає чинності `pending_prohibitions` на ребрі ланцюга.

### Крок 3. LP — `build_lp`

**Змінні** (усі ≥ 0; h — індекс тижня горизонту):

| Змінна | Індекси | Зміст | Верхня межа |
| --- | --- | --- | --- |
| `x[s, h]` | слот, h = 0..H−1 | відправка слотом | `cap[s, h]` |
| `stock[i, h]` | `tracked`, h = 1..H | запас на кінець тижня h−1 | `storage`, якщо є `spill` |
| `served_new[d, h]` | sink, h = 0..H−1 | обслугований попит тижня | `demand_forecast` (після 8 тижнів — останнє значення) |
| `backlog[d, h]`, `served_old[d, h]` | лише sink з backlog | перенесений попит і його обслуговування | — |
| `proc[j, h]` | процес, h = 0..H−1 | запуск виробництва | через групові обмеження |
| `spill[i, h]` | `tracked` зі storage, не протока, не grid | надлишок понад storage | — |
| `burn`, `gserved` | лише при `model_grids` | спалене паливо, обслужене навантаження | `share · G_bar`, `y_bar` |

**Баланс запасу** для кожного `(node, k)` з `tracked` і h = 1..H (рівність):

```
stock[h] = stock[h-1]                       (stock[0] = stock.qty, спостережене)
         + supply_avail(node, k)             (graph_now.supply.avail, persisted)
         + pipeline, що приходить у node на тиждень h-1
         + Σ x[s, h-1-tau_s]   для слотів s з призначенням (node, k)
         + Σ proc[j, h-1-tau_j] для процесів j з виходом (node, k)
         − Σ x[s, h-1]         для слотів s з відправленням (node, k)
         − served_new[d, h-1] (− served_old)  якщо (node, k) — sink
         − Σ proc[j, h-1]      для процесів j зі входом (node, k)
         − spill[h]            якщо є storage-обмеження
         − Σ burn[h-1]         лише при model_grids
```

`tau_s` — сума `graph_now.tau` по ребрах ланцюга слота (persisted), `tau_j` — з
`static["instance"]["nodes"]` (fab 6–8, OSAT 2). Наближення: pipeline-вантаж зараховується в голову
свого **поточного** ребра (довгі лейни через протоки не простежуються далі).

**Інші обмеження:**

- група потужності, кожен h: `Σ_{j ∈ група} proc[j, h] ≤ cap_eff` (fab) або `thr_eff` (OSAT),
  persisted;
- backlog-sink: `backlog[h] = backlog[h-1] + demand[h-1] − served_new − served_old`,
  `served_old[h] ≤ backlog[h]`;
- `model_grids`: `gserved[g, h] − Σ burn[g, ·, h] ≤ share_unmodelled · G_bar`; раціонування lng
  `ψ · ibar · burn ≤ share · G_bar · stock_prev`.

**Ціль (мінімізація):**

| Член | Коефіцієнт |
| --- | --- |
| `x[s, h]` | `graph_now.c[e0] + tariff_forecast[e0, k, h] · v_k` (лише перше ребро ланцюга) |
| `stock[i, h]` | `holding_scale · holding_cost[(node, k)]` |
| `served_new[d, h]` | `−pi_d` (штраф за нестачу; константа `pi · demand` відкинута) |
| `spill[i, h]` | 0 на джерелах (як у симуляторі), `disposal_cost[k]` деінде |
| terminal value | `−terminal_scale · v_k` на `stock[i, H]`; на `x[s, h]` з `h + tau_s ≥ H` (за товаром слота); на `proc[j, h]` з `h + tau_j ≥ H` (за `k_out`); grid виключено, якщо `model_grids` вимкнено |
| `gserved[g, h]` | `−VOLL` (лише `model_grids`) |

Розмір LP: tiny ~530 змінних, small ~2340 (H = 16); розв'язок 0.01 / 0.07 с; на full `act` ≈ 0.33 с
медіанно при бюджеті 4 с.

### Крок 4. Розв'язок

`linprog(c, A_eq, b_eq, A_ub, b_ub, bounds)`. Якщо `res.success` хибне — виняток і fallback (крок
"Fallback" нижче). Береться лише `x[:, 0]`.

### Крок 5. Hybrid: хто вирішує кожен слот

`heuristic_flows = _fallback(observation)` = `u0 · action_mask · ∏ open_now^1` по протоках лейна
(правило `agents/heuristic`, використовує статичний `u0`; середовище обрізає надлишок).

- `lp_scope = "all_but_grid"` (за замовчуванням): **просте правило** для слотів із товаром у
  `grid_fuels` і лейнів, що закінчуються в grid; **LP** — для решти. Критерій — за товаром, бо на
  small паливо йде в grid через вузол `terminal`.
- Слот, що бере просте правило: `min(heuristic, cap[s, 0])` — LP-capacity додає майбутні санкції й
  прогноз, яких просте правило не бачить.
- LP-слот, що не веде в sink: `max(lp, upstream_floor · min(heuristic, cap[s, 0]))`.
- `lp_scope = "demand"`: правило Stage 1 — LP лише для лейнів у sink.

Цей блок обгорнутий маркерами `EVOLVE-BLOCK` для OpenEvolve.

### Крок 6. Паливо

Для слотів із товаром у `grid_fuels`: `flows[s] *= fuel_mult` (tiny 1, small 10, full 10).

### Крок 7. Випуск із проток

`override_qty = 0`, `release_mode = 0` (дефолтний випуск середовища). Якщо `reroute` увімкнено,
`_reroute` випускає танкерний вантаж у обхід закритої протоки попереду (вимкнено: шкодить).

### Fallback

Будь-який виняток у кроках 1–5, зокрема нерозв'язне LP, — `flows = _fallback(observation)` для
всіх слотів; кроки 6–7 виконуються далі. Тиждень ніколи не віддається серверному naive.

## 4. Що свідомо не змодельовано

- Споживання палива електростанціями (є, але вимкнене: `model_grids`).
- Черги на протоках (`queue_lots`), throughput `kappa`, war-risk, `queue_holding`.
- Енергоспоживання fab; ефективна потужність fab береться зі спостереження (`cap_eff`) і persisted.
- Майбутні зміни capacity, freight, supply — persisted на весь горизонт.
- Freight і тариф лейна рахуються лише за першим ребром.

## 5. Обмеження сервера, яких дотримується код

- Імпорти: stdlib, numpy, scipy.
- Немає форм, захардкоджених під tiny: усі розміри — з `config`.
- RNG `np.random.default_rng(config["policy_seed"])` (агент детермінований).
- Файли читаються відносно `Path(__file__).parent`.
