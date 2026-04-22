"""
BaseTerrainWrapper — wraps MiniCheetahEnv (196-dim obs) with custom terrain
heightfields from the map_registry, replacing the old AdvancedTerrainEnv
(57-dim obs) wrapper that was incompatible with the trained checkpoint.

How it works
------------
MiniCheetahEnv._generate_terrain(cfg, rng) creates a TerrainGenerator and
calls its .generate(terrain_type, difficulty) → heights ndarray, then writes
that ndarray to MuJoCo hfield_data.

MiniCheetahRegistryEnv subclasses MiniCheetahEnv and overrides
_generate_terrain() to:
  1. Create the standard TerrainGenerator (for get_height_at / sample_heightmap).
  2. Call the map_registry generator instead of the built-in one.
  3. Inject the heights into _terrain_gen._heightfield so all downstream
     observation/reward code works identically.

This preserves the full 196-dim observation and all reward structure.

Usage
-----
    from envs.base_terrain_wrapper import BaseTerrainWrapper

    env = BaseTerrainWrapper(
        terrain_name="pyramid_stairs",
        difficulty=0.5,
        render_mode="none",
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
"""

import sys
import os
import numpy as np
from typing import Optional

# Make project root and src importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_TESTING = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TERRAIN_TESTING)
for _path in [_REPO_ROOT, os.path.join(_REPO_ROOT, "src"),
              os.path.join(_REPO_ROOT, "terrain_testing")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.env.cheetah_env import (
    MiniCheetahEnv, TerrainGenerator,
    HFIELD_NROW, HFIELD_NCOL, HFIELD_SIZE,
)
from terrain_testing.maps.map_registry import REGISTRY, get_generator, list_terrains


# ── Per-terrain domain-randomization overrides (friction etc.) ───────────────
# Applied inside reset() after standard domain randomization.
_TERRAIN_OVERRIDES: dict = {
    "frozen_lake": {
        "floor_friction": (0.05, 0.15),
        "foot_friction":  (0.05, 0.20),
    },
    "rma_stepping_stones": {
        "floor_friction": (0.4, 0.8),
    },
    "parkour_wall": {
        "floor_friction": (0.8, 1.5),
        "foot_friction":  (1.2, 2.0),
    },
}


# ── MiniCheetahEnv subclass that uses a registry terrain generator ────────────

class MiniCheetahRegistryEnv(MiniCheetahEnv):
    """MiniCheetahEnv with terrain generated from terrain_testing/maps/map_registry.

    The only change from the parent is _generate_terrain(): instead of calling
    the built-in TerrainGenerator.generate(terrain_type, difficulty), we look
    up the registry generator for `terrain_registry_name` and inject its output.
    Everything else (obs, rewards, 196-dim observation) is identical.

    Args:
        terrain_registry_name: key in map_registry.REGISTRY (e.g. "pyramid_stairs")
        All other args/kwargs: forwarded unchanged to MiniCheetahEnv.__init__
    """

    def __init__(self, terrain_registry_name: str, **kwargs):
        if terrain_registry_name not in REGISTRY:
            raise ValueError(
                f"Unknown terrain '{terrain_registry_name}'. "
                f"Available: {list_terrains()}"
            )
        self._registry_name = terrain_registry_name
        # Placeholder so parent __init__ doesn't raise on unknown terrain_type;
        # the actual heightfield comes from the registry in _generate_terrain().
        kwargs.setdefault("terrain_type", "flat")
        kwargs.setdefault("use_terrain", True)
        super().__init__(**kwargs)

    def _generate_terrain(self, cfg: dict, rng):
        """Override: use registry generator instead of built-in terrain types.

        Writes the registry-generated heights into both the MuJoCo hfield and
        self._terrain_gen._heightfield so observation / height-query code works.
        """
        seed = (int(rng.integers(0, 2**31))
                if hasattr(rng, "integers")
                else int(rng.randint(0, 2**31)))

        # Build the standard TerrainGenerator shell (carries get_height_at /
        # sample_heightmap which use _heightfield internally).
        self._terrain_gen = TerrainGenerator(
            size=HFIELD_SIZE * 2,
            resolution=HFIELD_NROW,
            seed=seed,
        )

        # Generate heights via registry, then inject into the generator shell.
        reg_rng = np.random.RandomState(seed)
        gen_fn = get_generator(self._registry_name)
        heights = gen_fn(HFIELD_NROW, cfg.get("difficulty", 0.5), reg_rng)
        self._terrain_gen._heightfield = heights  # used by get_height_at / sample_heightmap

        # Write to MuJoCo hfield (same normalization as parent).
        if self._has_hfield:
            h_min = float(heights.min())
            h_max = float(heights.max())
            h_range = h_max - h_min
            if h_range > 1e-6:
                normalized = (heights - h_min) / h_range
            else:
                normalized = np.full_like(heights, 0.2)
            self.model.hfield_data[:HFIELD_NROW * HFIELD_NCOL] = (
                normalized.flatten().astype(np.float32)
            )
            self.model.hfield_size[self._hfield_id][2] = max(h_range, 0.001)
            self.model.hfield_size[self._hfield_id][3] = 0.01
            if self._floor_geom_id >= 0:
                self.model.geom_pos[self._floor_geom_id][2] = h_min


# ── Public wrapper (preserves original API) ───────────────────────────────────

class BaseTerrainWrapper:
    """Thin wrapper around MiniCheetahRegistryEnv adding per-terrain overrides.

    Exposes the same gym.Env interface.  observation_space is (196,) —
    identical to the base MiniCheetahEnv, so checkpoints are directly
    compatible.

    Args:
        terrain_name: key in map_registry.REGISTRY
        difficulty: terrain difficulty [0.0, 1.0]
        fixed_difficulty: if True, always use exactly this difficulty
        fixed_skill: if not None, force this skill mode every episode
        render_mode: "none" | "human" | "rgb_array"
        **env_kwargs: forwarded to MiniCheetahRegistryEnv / MiniCheetahEnv
    """

    def __init__(
        self,
        terrain_name: str = "flat",
        difficulty: float = 0.5,
        fixed_difficulty: bool = False,
        fixed_skill: Optional[str] = None,
        render_mode: str = "none",
        **env_kwargs,
    ):
        if terrain_name not in REGISTRY:
            raise ValueError(
                f"Unknown terrain '{terrain_name}'. "
                f"Available: {list_terrains()}"
            )
        self.terrain_name = terrain_name
        self._difficulty = difficulty
        self.fixed_difficulty = fixed_difficulty
        self.fixed_skill = fixed_skill

        # Resolve fixed skill: MiniCheetahEnv accepts forced_mode=
        if fixed_skill is not None:
            env_kwargs["forced_mode"] = fixed_skill

        self._env = MiniCheetahRegistryEnv(
            terrain_registry_name=terrain_name,
            render_mode=render_mode,
            terrain_difficulty=difficulty,
            use_terrain=True,
            **env_kwargs,
        )

        # Forward gym spaces — 196-dim obs, 12-dim action
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self.metadata          = getattr(self._env, "metadata", {})

    # ── terrain friction overrides ────────────────────────────────────────────

    def _apply_terrain_overrides(self):
        """Apply per-terrain friction overrides after standard domain randomization."""
        if self.terrain_name not in _TERRAIN_OVERRIDES:
            return
        overrides = _TERRAIN_OVERRIDES[self.terrain_name]
        model     = self._env.model
        rng       = self._env.np_random

        floor_friction = overrides.get("floor_friction")
        foot_friction  = overrides.get("foot_friction")

        import mujoco
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "floor" in name and floor_friction is not None:
                model.geom_friction[i][0] = float(rng.uniform(*floor_friction))
            elif name and "foot" in name and foot_friction is not None:
                model.geom_friction[i][0] = float(rng.uniform(*foot_friction))

    # ── gym.Env interface ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if self.fixed_difficulty:
            options = options or {}
            options["terrain_difficulty"] = self._difficulty

        obs, info = self._env.reset(seed=seed, options=options)
        self._apply_terrain_overrides()
        info["terrain_name"]   = self.terrain_name
        info["terrain_source"] = REGISTRY[self.terrain_name].source
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        info["terrain_name"] = self.terrain_name
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ── convenience helpers ───────────────────────────────────────────────────

    def set_difficulty(self, d: float):
        self._difficulty = float(np.clip(d, 0.0, 1.0))
        self._env.terrain_difficulty = self._difficulty

    @property
    def unwrapped(self):
        return self._env

    def __getattr__(self, name):
        return getattr(self._env, name)
