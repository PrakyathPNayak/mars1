"""
Benchmark smoke tests — 100 steps on every terrain, no exceptions allowed.

This is the fastest integration gate: if it passes, the full benchmark
can be run safely.

Run:
    python -m pytest terrain_testing/tests/test_benchmark.py -v
    python -m pytest terrain_testing/tests/test_benchmark.py -v --tb=short
"""

import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_TESTING = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TERRAIN_TESTING)
for _p in [_TERRAIN_TESTING, _REPO_ROOT, os.path.join(_REPO_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from maps.map_registry import list_terrains
from envs.base_terrain_wrapper import BaseTerrainWrapper

ALL_TERRAINS = list_terrains()
SMOKE_STEPS = 100


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_smoke_100_steps(name):
    """Every terrain must survive 100 random-action steps without crashing."""
    env = BaseTerrainWrapper(
        terrain_name=name,
        difficulty=0.5,
        fixed_difficulty=True,
        fixed_skill="trot",
        render_mode="none",
        randomize_domain=False,
    )
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == (57,), f"Bad obs shape: {obs.shape}"
        assert info["terrain_name"] == name

        rewards = []
        for step in range(SMOKE_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Core invariants every step
            assert obs.shape == (57,), f"step {step}: bad obs shape"
            assert np.isfinite(reward), f"step {step}: non-finite reward {reward}"
            assert not np.any(np.isnan(obs)), f"step {step}: NaN in obs"
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            rewards.append(reward)

            if terminated or truncated:
                obs, info = env.reset(seed=step)
                assert not np.any(np.isnan(obs))

        # Sanity: at least some reward signal
        assert len(rewards) > 0
        assert any(np.isfinite(r) for r in rewards)

    finally:
        env.close()


@pytest.mark.parametrize("name", ALL_TERRAINS)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_smoke_difficulty_levels(name, difficulty):
    """Env must not crash at any difficulty level."""
    env = BaseTerrainWrapper(
        terrain_name=name,
        difficulty=difficulty,
        fixed_difficulty=True,
        fixed_skill="trot",
        render_mode="none",
        randomize_domain=False,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert not np.any(np.isnan(obs))
        for _ in range(20):
            obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(r)
            if terminated or truncated:
                break
    finally:
        env.close()


@pytest.mark.parametrize("skill", ["walk", "trot", "run", "jump", "crouch", "stand"])
def test_smoke_all_skills_on_flat(skill):
    """All 6 skill modes must work on flat terrain."""
    env = BaseTerrainWrapper(
        terrain_name="flat",
        difficulty=0.0,
        fixed_difficulty=True,
        fixed_skill=skill,
        render_mode="none",
        randomize_domain=False,
    )
    try:
        obs, info = env.reset(seed=0)
        assert info["skill_mode"] == skill
        for _ in range(SMOKE_STEPS):
            obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(r)
            if terminated or truncated:
                env.reset(seed=1)
    finally:
        env.close()


def test_benchmark_all_terrains_complete():
    """All terrains complete 10 steps without any exception — used as gate."""
    failed = []
    for name in ALL_TERRAINS:
        try:
            env = BaseTerrainWrapper(
                terrain_name=name,
                difficulty=0.3,
                fixed_difficulty=True,
                fixed_skill="trot",
                render_mode="none",
                randomize_domain=False,
            )
            env.reset(seed=0)
            for _ in range(10):
                obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            env.close()
        except Exception as e:
            failed.append(f"{name}: {e}")

    if failed:
        pytest.fail("These terrains crashed:\n" + "\n".join(failed))
