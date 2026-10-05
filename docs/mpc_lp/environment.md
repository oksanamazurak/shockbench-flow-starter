# Що відомо про середовище ShockBench-Flow

Факти про симулятор, з'ясовані під час роботи над агентом. Джерела: публічні поля `config` і
observation, `docs/GUIDE.md`, `docs/fields/*.md` та код пакета `shockbench_flow` у `.venv`. Цей
пакет можна **читати** для розуміння, але `agent.py` не може його **імпортувати**, бо на сервері
його нема.

## Мережі

| | tiny | small | full |
| --- | --- | --- | --- |
| Тижнів в епізоді | 26 | 52 | 104 |
| Action slots | 20 | 108 | 395 |
| Протоки | 1 (`chk`) | 7 (Hormuz, Malacca, Suez, Cape, Taiwan, Panama, Turkish) | більше |
| Товари | lng, wafer, chip_le_raw, chip_le | + crude, nucfuel, chip_mat_raw, chip_mat | — |
| Вузли | source, chokepoint, grid, material, fab, osat, sink | + 4 `terminal` (паливо йде source → terminal → grid), 6 fab, 3 OSAT, 4 grid, 4 sink | — |
| Override-слоти / release pairs | 4 / 1 | 36 / 14 | — |
| CPU-бюджет на тиждень | — | 2 с | 4 с |

`static["instance"]["kind"]` повертає `"tiny"` / `"small"` / `"full"`.

**Форми не можна хардкодити**: параметри, прив'язані до кількості слотів (як `fraction` у
`06_policy_search`), підібрані на tiny, ламають агента на small (`ValueError: shapes (108,) (20,)`,
усі тижні грає naive).

## Публічні дані в `config["static"]["instance"]`

- `nodes[i]["stock"][commodity]`: `holding_cost`, `storage`, `salvage`, `supply_rate` (у джерел).
- `nodes[i]["fab"]`: `input`, `product`, `tau` (lead time, тижнів), `cap0`, `grid`, `e`, `w_scr`.
- `nodes[i]["osat"]`: `packages` (`{raw: packaged}`, кілька на один OSAT), `tau`, `thr`.
- `nodes[i]["grid"]`: `shares` (за паливом, плюс `unmodelled`), `rationed`, `priority`
  (`base_first` на всіх grid tiny і small), `voll`, `base_load`, `deliverable`, `ibar`, `days_cover`.
- `nodes[i]["chokepoint"]`: `queue_holding`, `war_risk_cost`, `mu`, `k_c`.
- `nodes[i]["sink"]["demand"][k]`: `dbar`, `phi`, `sigma`, `pi`, `backlog`.
- `commodities[k]`: `id`, `v` (ціна), `disposal_cost`.
- `params`: `psi` (поріг раціонування), `top_tariff` тощо.

`layout["warning_units"]` складається з пар `[kind, index]` (наприклад, `["chokepoint", 3]`), а не з
рядків. На tiny рядок протоки — 15, але його треба шукати, а не хардкодити.

## Вартість

Компоненти `last_week.cost_components`: freight, war_risk, tariff, holding, queue_holding,
shortage, disposal, shed.

Розподіл вартості агента `mpc` на small (4 train episodes): **shed 60%, shortage 39%**, holding 0.4%,
disposal 0.2%, tariff 0.2%, freight 0.2%, queue_holding і war_risk ≈ 0.

## Електростанції (grid)

З коду симулятора (`dynamics/sim.py`, `dynamics/production.py`):

- паливо k дає до `share_k · G_bar` енергії, але не більше, ніж є на складі grid; частка
  `unmodelled` (`share_None · G_bar`) доступна завжди;
- раціонування: для `rationed`-палива (lng) доступне множиться на `min(1, I_prev / (psi · ibar))`;
- `base_first`: базове навантаження `y_bar` обслуговується першим, решта енергії йде fab;
  непокрите базове навантаження — **shed за VOLL** (≈ 4.13 млн $ за GWh);
- спалене паливо = доступне × коефіцієнт завантаження; списується зі складу grid;
- одиниці: паливо в GWh, енергія в GWh, конверсія 1:1.

Спостерігаються `graph_now.grid.G_bar` і `graph_now.grid.y_bar`.

**Виміряно** (агент `mpc`, small, 4 train episodes):
- 98% shed — у тижні, коли grid бракує конкретного палива; 2% — коли `G_bar < y_bar`;
- палива на джерелах у середньому ~162 тис. GWh/тиждень (з `fuel_mult` ~145 тис.) при shed
  ~6.5–6.9 тис. GWh/тиждень;
- на джерелах **виконано ≈ запитано**: обмежує запит простого правила, а не capacity, тому
  `fuel_mult` допомагає;
- палива різних видів не взаємозамінні: у grid, що відключається, часто лежить надлишок іншого
  палива.

## Виробництво

- Fab: споживає `input`, через `tau` тижнів дає `product`; потужність тижня —
  `graph_now.fab.cap_eff` (уже враховує енергію й відновлення).
- OSAT: packages ділять throughput `graph_now.osat.thr_eff`; пакування пропорційне сирим запасам.
- Одиниці wafer, chip_*_raw, chip_* однакові ("wafer-eq 300 mm"), тому конверсія 1:1.

## Storage і disposal

З коду симулятора: надлишок понад storage **утилізується за `disposal_cost` лише на вузлах, що не є
протоками й не є джерелами**. На джерелах поповнення просто обрізається до storage, безкоштовно. На
протоках ліміту нема.

Жорсткий storage-ліміт у LP на всіх вузлах (перша спроба Stage 2) робив LP нерозв'язним щотижня.

## Протоки, черги, випуск

- Закриття: `graph_now.open` (1 відкрита … 0 закрита). Вантаж, що прийшов до протоки, стає лотом у
  черзі (`queue_lots.*`; на small видно лише `qty`).
- **Дефолтний випуск** (`dynamics/chokepoint.py`): кожен лот іде своїм лейном далі, FIFO по когортах,
  пропорційно capacity наступного ребра й throughput `kappa`; лот, чиє наступне ребро заборонене,
  стоїть.
- **Override** (`release_mode = 1`, `override_qty`) — **лише танкерні товари** (`commodities.override`:
  tiny — lng; small — lng, crude). Забирає товар k із черги протоки c FIFO і шле виходом обраного
  override-слота; обрізається capacity ребра, вмістом черги й throughput. Будь-який override або
  hold вимикає дефолтний випуск цієї пари (c, k) на тиждень. `release_mode = 2` — hold.
- Черги на small великі: 40–300 тис. одиниць щотижня. Перенаправлення в обхід закритої протоки
  попереду програло дефолтному випуску (див. `experiments.md`).

## Повідомлення й попередження

- `warning.score` — оцінка ризику для регіонів, пар регіонів і проток, із лагом 1 тиждень.
- `messages.*` — живі анонси (оголошені, ще не чинні, не відкликані). Канали: 0 tariff_formal,
  1 tariff_informal, 2 tariff_final, 3 sanction_legal, 4 ties_threat, 5 mid_threat. Kind: 0 proposal,
  1 final_notice, 2 threat, 3 publication, 4 withdrawal. Частина — хибні тривоги.
- `pending_prohibitions.*` — оголошені санкції з точним тижнем набрання чинності.
- **Тарифні повідомлення** (24 episodes tiny): усі адресовані регіону (`target_kind = 3`) з
  конкретним товаром; тариф на імпорт виростає в ~45% випадків, у середньому на ~0.2.

## Нестача чипів

На small нестача є майже щотижня на всіх 8 sink (106–155 зі 156 тижнів). Запити на останньому
відрізку обрізаються браком товару, а не capacity, тобто чипів бракує вже на виробництві, а воно
залежить від електрики.

## Інше

- Middle-of-lane: `pipeline.*` показує поточне ребро вантажу й тиждень прибуття в його голову.
- `action_mask` = 1, якщо на маршруті слота немає санкції; закриття й capacity треба читати з
  `graph_now.open` і `graph_now.u`.
- Запит понад capacity безпечний: середовище обрізає (`last_week.clip.requested` /
  `executed`).
- Тиждень, у якому агент упав, перевищив CPU або повернув неправильну дію, грає naive rule
  (`fallback_count` на Codabench).
