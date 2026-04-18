# Terrain Testing Suite — MIT Mini Cheetah

A self-contained folder for building, testing, and benchmarking terrain environments
for the Mini Cheetah RL locomotion project.

Your friend who works on this reads papers → picks terrain ideas → implements them here
→ runs benchmark → reports results. Nothing in `src/env/terrain_env.py` needs to change.

---

## Folder Layout

```
terrain_testing/
├── README.md                    ← this file
├── envs/
│   ├── __init__.py
│   ├── base_terrain_wrapper.py  ← thin wrapper around AdvancedTerrainEnv
│   ├── paper_terrains.py        ← terrains from research papers (RMA, Isaac Gym, etc.)
│   └── custom_terrains.py       ← your friend's own invented terrains
├── maps/
│   ├── __init__.py
│   ├── map_generator.py         ← NumPy heightfield builders (no MuJoCo needed)
│   └── map_registry.py          ← central dict: name → generator function
├── scripts/
│   ├── run_single_terrain.py    ← spin up one terrain, run random/trained policy
│   ├── benchmark_all.py         ← loop over all registered terrains, collect metrics
│   └── visualize_heightfields.py← matplotlib preview of every map before running
├── tests/
│   ├── test_maps.py             ← unit tests: shape, range, no NaN
│   ├── test_envs.py             ← gym API compliance per terrain
│   └── test_benchmark.py        ← smoke test: 100 steps per terrain, no crash
├── configs/
│   ├── default.yaml             ← default episode/randomization settings
│   └── paper_configs.yaml       ← per-paper recommended params
├── results/                     ← auto-created; benchmark CSVs go here
└── docs/
    └── terrain_sources.md       ← which terrain came from which paper
```

---

## Quick Start

```bash
# 1. Preview all heightfields (no MuJoCo needed)
python terrain_testing/scripts/visualize_heightfields.py

# 2. Run a single terrain interactively
python terrain_testing/scripts/run_single_terrain.py --terrain rubble --steps 1000

# 3. Smoke-test every terrain (fast, ~5s each)
python terrain_testing/scripts/benchmark_all.py --steps 200 --policy random

# 4. Full benchmark with a trained checkpoint
python terrain_testing/scripts/benchmark_all.py \
    --steps 2000 \
    --policy checkpoint \
    --checkpoint checkpoints/best_model.zip \
    --output results/full_benchmark.csv

# 5. Unit tests
python -m pytest terrain_testing/tests/ -v
```

---

## Do I need to edit `src/env/terrain_env.py`?

**No.** The existing file is not touched.

`AdvancedTerrainEnv` already exposes:
- `set_terrain(terrain_type, difficulty)` — swap terrain at reset
- `set_skill(skill)` — change skill mode
- `set_command(vx, vy, wz, mode)` — change velocity command
- `randomize_terrain`, `randomize_domain`, `randomize_skill` flags
- `terrain_gen` — a `TerrainGenerator` instance you can replace

`BaseTe rrainWrapper` in `envs/base_terrain_wrapper.py` monkey-patches the
`terrain_gen` with a custom `TerrainGenerator`-compatible object so any map
your friend builds just slots in — no changes to the original file needed.

---

## How to Add a New Terrain

### Option A — Paper terrain
Add a function to `maps/map_generator.py` following the signature:

```python
def my_terrain(resolution: int, difficulty: float, rng: np.random.RandomState) -> np.ndarray:
    heights = np.zeros((resolution, resolution), dtype=np.float32)
    # ... fill heights ...
    return heights
```

Then register it in `maps/map_registry.py`:

```python
from maps.map_generator import my_terrain
REGISTRY["my_terrain"] = TerrainEntry(
    generator=my_terrain,
    source="AuthorName et al. YEAR",
    paper="Short description",
    difficulty_range=(0.0, 1.0),
    recommended_skill=["trot"],
)
```

### Option B — Custom terrain
Put the function in `maps/map_generator.py` under the `# === CUSTOM TERRAINS ===`
section and register with `source="custom"`.

---

## Metrics Collected by Benchmark

| Metric | Description |
|--------|-------------|
| `mean_ep_reward` | Mean total reward per episode |
| `mean_ep_length` | Mean steps before termination or truncation |
| `survival_rate` | Fraction of episodes not terminated (not fallen) |
| `mean_forward_vel` | Mean body-frame forward velocity |
| `fall_rate` | 1 - survival_rate |
| `mean_reward_components` | Per-term breakdown (if info has reward_components) |

Results are saved as CSV to `results/` and printed as a table.

---

## Paper Sources

See `docs/terrain_sources.md` for full citations and notes on each terrain.
