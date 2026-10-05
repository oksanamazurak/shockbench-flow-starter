# Робочі процеси

Усі команди виконуються з кореня репозиторію через `uv run`. Агента можна вказати назвою (`mpc` —
папка в `agents/`) або шляхом до папки, zip чи `agent.py`.

## Оцінка й порівняння

```bash
uv run sbf evaluate mpc_lp --task=small                  # оцінка на 20 dev episodes
uv run sbf evaluate mpc_lp --task=small --quick          # швидкий smoke-тест (не цифри борду)
uv run sbf compare NEW OLD --task=small               # парне порівняння; виграш, якщо інтервал не містить 0
uv run sbf compare NEW OLD --task=full --episodes=6   # full дорогий: еталони рахуються довго
```

Щоб порівняти з попередньою версією, витягни її з git у тимчасову папку:

```bash
mkdir -p /tmp/mpc_old && git show 715f9ed:agents/mpc_lp/agent.py > /tmp/mpc_old/agent.py \
  && git show 715f9ed:agents/mpc_lp/params.json > /tmp/mpc_old/params.json
uv run sbf compare mpc_lp /tmp/mpc_old --task=small
```

Оцінка на тренувальному root у Python (для перебору налаштувань):

```python
from sbf_starter import scoring
es = scoring.episode_set("small", 8, entropy=20261004, verbose=False)
r = es.score("path/to/folder_with_agent_py", cpu_budget=True)   # r.rss, r.rows[i]["fallback_weeks"]
```

Шаблон перебору: тимчасова папка на кожен варіант (копія `agent.py` + `params.json` =
`dict(base, **override)`), оцінка через `es.score`, вивід `rss` і суми `fallback_weeks`.

## Валідаційний бенчмарк: `sbf bench`

Вибір між кандидатами робиться на бенчмарку, а не на dev (dev лишається для фінального `sbf compare`).
Набори в `benchmarks/suites.yaml`: `*-val` (інтенсивність борду, рівні за вартістю naive) і `*-stress`
(`gamma` 0.79/0.95/0.97). Оцінка: частка економії відносно naive `(J_naive − J) / J_naive` на епізод, без
clairvoyant плану, тобто це не RSS борду, а порядок кандидатів. Тижні, які зіграв naive (CPU-бюджет, падіння),
показані окремо.

```bash
uv run sbf bench NEW mpc_lp --baseline=mpc_lp                      # усі набори
uv run sbf bench NEW mpc_lp --baseline=mpc_lp --suites=small-val   # лише один
```

Результати кешуються по епізоду в `outputs/bench-cache/` (ключ: вміст папки агента), повторний запуск грає лише
змінених агентів. Звіт: `outputs/bench/<дата>/report.md` і `report.json`.

## Перевірка, що зміна — лише рефакторинг

1. До зміни: `scoring.evaluate("agents/mpc_lp", task="small", episodes=4, entropy=20261004).rss`.
2. Після зміни — те саме; результат має збігатися до останнього знака (зараз **0.6316986377034457**).
3. Для `PARAMS`: порівняти словник до і після, виконавши код до рядка `if (HERE / "params.json")`.

## Діагностика

- **Чи розв'язується LP:** у циклі по тижнях викликати `build_lp(...)` і
  `linprog(...)`, рахувати `not res.success`. Має бути 0. Інакше агент тихо грає просте правило.
- **Розклад вартості:** сумувати `observation["last_week.cost_components"]` після кожного
  `env.step`; компоненти — freight, war_risk, tariff, holding, queue_holding, shortage, disposal, shed.
  Порівнювати два варіанти покомпонентно і по episodes: середнє ховає катастрофи в окремих episodes.
- **Що шле LP проти простого правила:** `a.act(o)["flows"]` проти `a._fallback(o)` по слотах.
- Середовище для діагностики:

```python
import gymnasium as gym, shockbench_flow_gym
from sbf_starter import env_id
from shockbench_flow_agent import agent_config
env = gym.make(env_id("small"), entropy=20261004)
o, info = env.reset(options={"episode": 0})
cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, o)
a = Agent(cfg)
o, r, done, trunc, info = env.step(a.act(o))
```

## Тюнінг: `examples/07_mpc_policy_search.py`

Еволюційний пошук 16 параметрів (прогноз закриттів, ваги повідомлень, `holding_scale`,
`closure_power`, H, тарифи). Навчання на своєму root; кандидат пишеться в
`outputs/07_mpc_policy_search/<дата>/best`, лише якщо він кращий за поточні значення на валідаційному наборі
`<task>-val` (`sbf bench`); `--holdout=dev` повертає старе порівняння на dev.

```bash
uv run python examples/07_mpc_policy_search.py --task=small --generations=8 --population=10 --train_episodes=8
```

- Small повільний (52 тижні × LP): 8×10 — десятки хвилин. Запускати у фоні.
- Поки йде пошук, **не змінювати `agents/mpc_lp/agent.py`**: пошук копіює його для кожного кандидата.
- Переможця застосувати: `cp outputs/07_mpc_policy_search/<дата>/best/params.json agents/mpc_lp/` — потім
  обов'язково `sbf compare` на dev і `sbf check`.

## OpenEvolve над агентом

Файли в `openevolve/`:
- `mpc_initial_program.py` — копія агента з трьома EVOLVE-BLOCK: формула прогнозу, hybrid,
  `Agent._fuel`;
- `mpc_evaluator.py` — small, 6 episodes, entropy 20261004; підкладає `params.json` поруч із кандидатом;
- `mpc_config.yaml` — sonnet, бюджет $1, timeout 180 с, 20 iterations, `max_code_length: 60000`;
- `params.json` — копія `agents/mpc_lp/params.json`.

```bash
cp agents/mpc_lp/params.json openevolve/params.json   # після кожної зміни параметрів агента
uv run openevolve-run openevolve/mpc_initial_program.py openevolve/mpc_evaluator.py \
  --config openevolve/mpc_config.yaml --output outputs/mpc_openevolve_small --iterations 20 --log-level INFO
```

- Перед запуском перевір, що seed дає ту саму оцінку, що й агент (`scoring.evaluate` обох на тих
  самих episodes).
- Після змін в агенті seed треба **перегенерувати**: скопіювати `agent.py` і знову вставити маркери
  блоків.
- Кожна iteration викликає `claude -p` і витрачає ліміт користувача. Iteration на small займає
  ~30 с + час LLM.
- Результат: `outputs/.../best/best_program.py`; diffs — у `checkpoints/checkpoint_N/programs/*.json`
  (`code`, `metrics`). Переможця перевіряти на dev через `sbf compare`.

## Тести й перевірка сервером

```bash
uv run pytest tests/test_mpc_agent.py -q        # 11 тестів агента
uv run pytest -n 3                              # усі тести репо (3 пропущені потребують Docker)
uv run sbf check mpc_lp --task=small               # імпорти, zip, CPU на тиждень; також --task=full
```

## Пакування й заливання (лише на прохання користувача)

```bash
uv run sbf pack mpc_lp                             # -> outputs/mpc_lp.zip (агент + params.json)
uv run sbf upload outputs/mpc_lp.zip --dry_run     # перевірка без заливання
uv run sbf upload outputs/mpc_lp.zip               # справжнє заливання: 1 з 5 спроб на день
uv run sbf status 962063 --wait                 # дочекатися оцінки
```

- Потрібен `.env` з `CODABENCH_COMPETITION` (URL змагання; competition 18290) і `CODABENCH_TOKEN`.
- `uv run sbf token` питає логін і пароль інтерактивно: запускати у звичайному терміналі, а не
  через `!` у Claude Code (там нема stdin, буде `EOFError`).
- Заливати zip, зібраний із закоміченого стану, і звіряти: `unzip -p outputs/mpc_lp.zip agent.py | diff - agents/mpc_lp/agent.py`.
- `__pycache__` у zip не потрапляє.

## Коміти

- Комітити лише за проханням користувача; гілка `oleksii`.
- Комітити явно за шляхами (`git commit -F - <paths>`): `openevolve/config.yaml` був у staging ще до
  нас, і його не можна захопити випадково.
- Наприкінці повідомлення коміту — рядок `Co-Authored-By`.
