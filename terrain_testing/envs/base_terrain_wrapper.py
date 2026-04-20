"""
BaseTe rrainWrapper — thin wrapper around AdvancedTerrainEnv that injects
a custom heightfield generator without touching the original source file.

How it works:
  AdvancedTerrainEnv uses self.terrain_gen (a TerrainGenerator instance).
  This wrapper replaces terrain_gen with a CompatibleTerrainGenerator that
  delegates to our map_registry functions.

  The wrapper also supports special per-terrain domain randomization overrides
  (e.g. very low friction for frozen_lake).

Usage:
    from envs.base_terrain_wrapper import BaseTerrainWrapper

    env = BaseTerrainWrapper(
        terrain_name="rubble_field",
        difficulty=0.7,
        skill_mode="trot",
        render_mode="human",
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
"""

import sys
import os
import numpy as np
from typing import Optional

# Make project src importable when running from repo root or terrain_testing/
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _path in [_REPO_ROOT, os.path.join(_REPO_ROOT, "src")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from env.terrain_env import AdvancedTerrainEnv, TerrainGenerator
from terrain_testing.maps.map_registry import REGISTRY, get_generator, list_terrains


class CompatibleTerrainGenerator:
    """Drop-in replacement for TerrainGenerator that delegates to map_registry.

    AdvancedTerrainEnv calls:
      self.terrain_gen.generate(terrain_type, difficulty) → ndarray
      self.terrain_gen.get_terrain_encoding(x, y) → ndarray (8,)
      self.terrain_gen.resolution
      self.terrain_gen.size
      self.terrain_gen._heightfield

    We implement all of these.
    """

    def __init__(self, terrain_name: str, size: float = 10.0,
                 resolution: int = 200, seed: int = None):
        if terrain_name not in REGISTRY:
            raise ValueError(
                f"Unknown terrain '{terrain_name}'. "
                f"Available: {list_terrains()}"
            )
        self.terrain_name = terrain_name
        self.size = size
        self.resolution = resolution
        self.rng = np.random.RandomState(seed)
        self._heightfield: Optional[np.ndarray] = None
        # Use underlying TerrainGenerator only for get_terrain_encoding
        self._inner = TerrainGenerator(size=size, resolution=resolution, seed=seed)

    def generate(self, terrain_type: str = None, difficulty: float = 0.5
                 ) -> np.ndarray:
        """Generate heightfield using our custom generator (ignores terrain_type arg)."""
        gen_fn = get_generator(self.terrain_name)
        heights = gen_fn(self.resolution, difficulty, self.rng)
        self._heightfield = heights
        self._inner._heightfield = heights
        return heights

    def get_terrain_encoding(self, x: float, y: float,
                            radius: float = 0.5) -> np.ndarray:
        """Delegate to inner TerrainGenerator's spatial encoding."""
        return self._inner.get_terrain_encoding(x, y, radius)


# Per-terrain domain randomization overrides (friction, etc.)
# These are applied after the standard domain randomization in reset().
_TERRAIN_OVERRIDES = {
    "frozen_lake": {
        "floor_friction": (0.05, 0.15),   # very low friction
        "foot_friction": (0.05, 0.20),
    },
    "rma_stepping_stones": {
        "floor_friction": (0.4, 0.8),
    },
    "parkour_wall": {
        "floor_friction": (0.8, 1.5),
        "foot_friction": (1.2, 2.0),
    },
}


class BaseTerrainWrapper:
    """Wraps AdvancedTerrainEnv with a custom terrain from the registry.

    All gym.Env methods are forwarded directly. The wrapper only intercepts
    reset() to inject the custom terrain generator before the parent resets.

    Args:
        terrain_name: name in map_registry.REGISTRY
        difficulty: 0.0–1.0 (overrides randomization if fixed_difficulty=True)
        fixed_difficulty: if True, use exactly this difficulty each reset
        fixed_skill: if not None, lock skill_mode to this value
        render_mode: "human" | "rgb_array" | "none"
        **env_kwargs: passed directly to AdvancedTerrainEnv
    """

    def __init__(
        self,
        terrain_name: str = "flat",
        difficulty: float = 0.5,
        fixed_difficulty: bool = False,
        fixed_skill: Optional[str] = None,
        render_mode: str = "none",
        resolution: int = 200,
        size: float = 10.0,
        **env_kwargs,
    ):
        self.terrain_name = terrain_name
        self._difficulty = difficulty
        self.fixed_difficulty = fixed_difficulty
        self.fixed_skill = fixed_skill
        self.resolution = resolution
        self.size = size

        # Always pass randomize_terrain=False: we control terrain ourselves
        env_kwargs["randomize_terrain"] = False
        env_kwargs["render_mode"] = render_mode
        if fixed_skill is not None:
            env_kwargs["randomize_skill"] = False
            env_kwargs["skill_mode"] = fixed_skill

        self._env = AdvancedTerrainEnv(**env_kwargs)

        # Replace terrain generator immediately
        self._inject_generator(seed=0)

        # Forward gym spaces
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.metadata = self._env.metadata

    def _inject_generator(self, seed: int = None):
        gen = CompatibleTerrainGenerator(
            terrain_name=self.terrain_name,
            size=self.size,
            resolution=self.resolution,
            seed=seed,
        )
        self._env.terrain_gen = gen

    def _apply_terrain_overrides(self):
        """Apply per-terrain friction overrides after standard domain randomization."""
        if self.terrain_name not in _TERRAIN_OVERRIDES:
            return
        overrides = _TERRAIN_OVERRIDES[self.terrain_name]
        model = self._env.model
        mj = self._env._mj
        rng = self._env.np_random

        floor_friction = overrides.get("floor_friction", None)
        foot_friction = overrides.get("foot_friction", None)

        for i in range(model.ngeom):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            if name and "floor" in name and floor_friction is not None:
                model.geom_friction[i][0] = float(rng.uniform(*floor_friction))
            elif name and "foot_collision" in name and foot_friction is not None:
                model.geom_friction[i][0] = float(rng.uniform(*foot_friction))

    def reset(self, seed=None, options=None):
        # Inject a fresh generator with a new seed each episode
        ep_seed = seed if seed is not None else int(np.random.randint(0, 2**31))
        self._inject_generator(seed=ep_seed)

        # Override difficulty if fixed
        if self.fixed_difficulty:
            self._env.difficulty = self._difficulty
        else:
            # Sample difficulty in [0, 1] — curriculum can call env.set_difficulty()
            pass

        obs, info = self._env.reset(seed=seed, options=options)
        self._apply_terrain_overrides()
        obs = obs[:45]
        info["terrain_name"] = self.terrain_name
        info["terrain_source"] = REGISTRY[self.terrain_name].source
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # 🔥 FIX: trim observation to match trained model (45)
        obs = obs[:45]

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def set_command(self, vx, vy, wz, mode="trot"):
        self._env.set_command(vx, vy, wz, mode)

    def set_skill(self, skill):
        self._env.set_skill(skill)

    def set_difficulty(self, d: float):
        self._difficulty = float(np.clip(d, 0.0, 1.0))
        self._env.difficulty = self._difficulty

    @property
    def unwrapped(self):
        return self._env

    def __getattr__(self, name):
        # Forward anything not explicitly defined to the inner env
        return getattr(self._env, name)
