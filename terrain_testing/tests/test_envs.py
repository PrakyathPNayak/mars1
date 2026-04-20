"""
Gym API compliance tests for every terrain environment.

Requires MuJoCo + the full project src/ to be installed.

Run:
    python -m pytest terrain_testing/tests/test_envs.py -v
    python -m pytest terrain_testing/tests/test_envs.py -v -k "flat or rma_rough"
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

# Skip entire module if MuJoCo is not installed
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from maps.map_registry import list_terrains
from envs.base_terrain_wrapper import BaseTerrainWrapper

ALL_TERRAINS = list_terrains()
FAST_TERRAINS = ["flat", "rma_rough", "pyramid_stairs", "rubble_field",
                 "parkour_gap", "frozen_lake"]

N_STEPS_QUICK = 50    # fast checks
N_STEPS_FULL = 500    # thorough checks


def make_env(name, difficulty=0.5, skill="trot"):
    return BaseTerrainWrapper(
        terrain_name=name,
        difficulty=difficulty,
        fixed_difficulty=True,
        fixed_skill=skill,
        render_mode="none",
        randomize_domain=False,   # deterministic for testing
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observation / action space checks (fast — all terrains)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_obs_space_shape(name):
    env = make_env(name)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (
        f"{name}: obs shape {obs.shape} != space shape {env.observation_space.shape}"
    )
    env.close()


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_obs_in_space(name):
    env = make_env(name)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs.astype(np.float32)), (
        f"{name}: reset obs not in observation_space"
    )
    env.close()


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_action_space_shape(name):
    env = make_env(name)
    assert env.action_space.shape == (12,), (
        f"{name}: expected action shape (12,), got {env.action_space.shape}"
    )
    env.close()


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_obs_no_nan_on_reset(name):
    env = make_env(name)
    obs, _ = env.reset(seed=0)
    assert not np.any(np.isnan(obs)), f"{name}: reset obs contains NaN"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Step API (quick — selected terrains)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FAST_TERRAINS)
def test_step_returns_5_tuple(name):
    env = make_env(name)
    env.reset(seed=0)
    result = env.step(env.action_space.sample())
    assert len(result) == 5, f"{name}: step should return 5-tuple"
    env.close()


@pytest.mark.parametrize("name", FAST_TERRAINS)
def test_step_obs_no_nan(name):
    env = make_env(name)
    env.reset(seed=0)
    for _ in range(N_STEPS_QUICK):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert not np.any(np.isnan(obs)), f"{name}: step obs contains NaN"
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("name", FAST_TERRAINS)
def test_step_reward_is_finite(name):
    env = make_env(name)
    env.reset(seed=0)
    for _ in range(N_STEPS_QUICK):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward), f"{name}: reward is not finite: {reward}"
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("name", FAST_TERRAINS)
def test_step_bools(name):
    env = make_env(name)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert isinstance(terminated, bool), f"{name}: terminated should be bool"
    assert isinstance(truncated, bool), f"{name}: truncated should be bool"
    env.close()


@pytest.mark.parametrize("name", FAST_TERRAINS)
def test_info_has_required_keys(name):
    env = make_env(name)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    required = {"step", "base_height", "skill_mode", "terrain_type",
                "terrain_name", "foot_contacts"}
    missing = required - set(info.keys())
    assert not missing, f"{name}: info missing keys: {missing}"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-episode reset (quick — selected terrains)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FAST_TERRAINS)
def test_multiple_resets(name):
    env = make_env(name)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))
        assert info["terrain_name"] == name
    env.close()


@pytest.mark.parametrize("name", FAST_TERRAINS)
def test_reset_after_termination(name):
    """Env should reset cleanly after a fall (termination)."""
    env = BaseTerrainWrapper(
        terrain_name=name,
        difficulty=1.0,        # hard → more likely to fall
        fixed_difficulty=True,
        fixed_skill="trot",
        render_mode="none",
        randomize_domain=False,
        episode_length=200,
    )
    for _ in range(2):
        obs, _ = env.reset(seed=0)
        for _ in range(200):
            obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        # Should be able to reset cleanly
        obs, _ = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Observation dimensionality (57 dims for terrain env)
# ─────────────────────────────────────────────────────────────────────────────

def test_obs_dim_is_57():
    env = make_env("flat")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (57,), f"Expected obs (57,), got {obs.shape}"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper passthrough
# ─────────────────────────────────────────────────────────────────────────────

def test_wrapper_set_command():
    env = make_env("flat")
    env.reset(seed=0)
    env.set_command(1.0, 0.0, 0.0, mode="trot")
    np.testing.assert_allclose(env.unwrapped.command, [1.0, 0.0, 0.0])
    env.close()


def test_wrapper_set_skill():
    env = BaseTerrainWrapper(
        terrain_name="flat",
        difficulty=0.0,
        fixed_difficulty=True,
        render_mode="none",
        randomize_skill=False,
    )
    env.reset(seed=0)
    env.set_skill("walk")
    assert env.unwrapped.skill_mode == "walk"
    env.close()


def test_wrapper_set_difficulty():
    env = make_env("flat")
    env.set_difficulty(0.8)
    assert abs(env._difficulty - 0.8) < 1e-6
    env.close()
