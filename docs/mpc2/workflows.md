# Як працювати з `agents/mpc2`

## Оцінка

```bash
uv run sbf check mpc2 --task=small            # час ходу, локально
uv run sbf check mpc2 --task=small --docker   # у контейнері бенчмарку
uv run sbf check mpc2 --task=full
uv run sbf bench mpc2 <копія> --baseline=mpc2 --suites=small-val --n_jobs=16   # кандидат проти поточного
uv run sbf bench mpc2 <копія> --baseline=mpc2 --suites=full-val
uv run sbf compare <копія> mpc2 --task=small --cpu_budget                    # dev, RSS як на борді
```

- Кандидат — копія папки агента (`agent.py`, `params.json`, `sbfv/`, `terminal_*.npz`) зі зміненим параметром.
  Параметр у `params.json` перекривається `SMALL_OVERRIDES` / `FULL_OVERRIDES`, тож для параметрів, які
  задаються там, міняйте сам `agent.py` копії (`sed` у блоці `*_OVERRIDES`).
- `sbf bench` кешує епізоди за вмістом папки (`outputs/bench-cache/`): будь-яка зміна `agents/mpc2/agent.py`
  скидає кеш `mpc2`, і базовий агент перераховується.
- `--n_jobs=16` ближче до сервера (15 епізодів на 8 ядрах з SMT); 8 процесів занижують ризик перевищення.
- `small-val`: 80 епізодів, ≈30–60 хв на агента; `full-val`: 40 епізодів, ≈1 год.

## Цінність запасу на кінці вікна

```bash
uv run python scripts/terminal_values.py --task=small --root=9301 --episodes=60
uv run python scripts/terminal_values.py --task=full --root=9401 --episodes=48 [--out=outputs/x.npz]
```

Пише `agents/mpc2/terminal_<task>.npz` (або `--out`). Small — ≈15 с, full — ≈4 хв на 8 процесах. Корені
9301/9401 окремі від бенч-наборів (7xxx), dev (0) і тюнінгу (12345). Агент читає всі `terminal_*.npz` біля себе
під час імпорту й вибирає таблицю за `T`.

## Перевірка, що прискорення не міняє рішень

```bash
uv run python -c "from sbf_starter import bench; print(bench.play_one('agents/mpc2','small',None,7101,3,False)['J_cents'])"
```

Порівняти J до і після зміни на кількох епізодах small і одному full: має збігатися до цента.

## Профіль часу

```python
import cProfile, pstats, shockbench_flow_gym as g
from shockbench_flow_agent.shim import load_agent_class
env = g.make_env("small"); obs, info = env.reset(seed=7)
a = load_agent_class("agents/mpc2", "p")(g.agent_config_from_reset(env, obs, info)); act = a.act(obs)
pr = cProfile.Profile(); pr.enable()
for _ in range(40):
    obs, *_ = env.step(act); act = a.act(obs)
pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(15)
```

CPU-час усіх потоків проти реального: `time.process_time()` проти `time.perf_counter()` навколо `a.act`.

## OpenEvolve через Claude Code

`evolve/config.yaml` і `evolve/mpc2/config.yaml` використовують `provider: claude_code`, модель `sonnet`
(`claude -p` з входом Claude Code, API-ключ не потрібен). `max_budget_usd` треба задавати в налаштуваннях
моделі: значення верхнього рівня до моделей не доходить, а CLI відкидає порожнє.

```bash
uv run --extra evolve python examples/08_openevolve_agent.py --setup=mpc2 --iterations=40
```

`evolve/mpc2/`: `initial_agent.py` — обгортка над `mpc2` з функцією `adjust(flows, naive, info, week, mem)`;
`evaluator.py` — середня економія відносно naive на 24 епізодах `small-val` (7101/7102 n 0..11), у
окремому процесі Python (joblib не стартує свої процеси всередині робочого процесу OpenEvolve). Переможець
пишеться в `outputs/08_openevolve_agent/<час>/champion/` як готова папка.

## Нейромережа й PPO (дослідження)

```bash
uv run python scripts/nn_collect.py --task=small --root=9101 --episodes=200   # дані з mpc2
uv run python scripts/nn_train.py --task=small                                # agents/nn_mpc/weights_small.pt
uv run python scripts/ppo_slots.py --task=small --iterations=300              # agents/ppo_slots/policy_small.pt
```

Обидва агенти — мережа зі спільними вагами на слот (будь-яка кількість слотів), окремі ваги для small і full.
Результати — у [research.md](research.md); на борд не завантажувались.
