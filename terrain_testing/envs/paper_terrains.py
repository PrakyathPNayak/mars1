"""
Convenience constructors — one function per paper terrain.

Each returns a BaseTerrainWrapper pre-configured with the settings
recommended in the corresponding paper.

Usage:
    from envs.paper_terrains import make_rma_rough, make_pyramid_stairs

    env = make_rma_rough(difficulty=0.5, render_mode="human")
    obs, info = env.reset()
"""

from envs.base_terrain_wrapper import BaseTerrainWrapper


def _make(terrain_name, difficulty, skill_mode, fixed_difficulty,
          render_mode, **kw) -> BaseTerrainWrapper:
    return BaseTerrainWrapper(
        terrain_name=terrain_name,
        difficulty=difficulty,
        fixed_difficulty=fixed_difficulty,
        fixed_skill=skill_mode if skill_mode else None,
        render_mode=render_mode,
        **kw,
    )


# ── Isaac Gym / legged_gym ────────────────────────────────────────────────────

def make_flat(difficulty=0.0, skill_mode=None,
              fixed_difficulty=True, render_mode="none", **kw):
    """Flat ground baseline."""
    return _make("flat", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_pyramid_stairs(difficulty=0.5, skill_mode="trot",
                         fixed_difficulty=False, render_mode="none", **kw):
    """Rudin 2022 pyramid stair terrain."""
    return _make("pyramid_stairs", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_discrete_obstacles(difficulty=0.5, skill_mode="trot",
                              fixed_difficulty=False, render_mode="none", **kw):
    """Rudin 2022 discrete obstacle terrain."""
    return _make("discrete_obstacles", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_wave_terrain(difficulty=0.5, skill_mode="trot",
                       fixed_difficulty=False, render_mode="none", **kw):
    """Rudin 2022 wave undulation terrain."""
    return _make("wave_terrain", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_sloped_terrain(difficulty=0.5, skill_mode="trot",
                         fixed_difficulty=False, render_mode="none", **kw):
    """Rudin 2022 sloped terrain (direction randomised each reset)."""
    return _make("sloped_terrain", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


# ── RMA ──────────────────────────────────────────────────────────────────────

def make_rma_rough(difficulty=0.5, skill_mode="trot",
                    fixed_difficulty=False, render_mode="none", **kw):
    """Kumar 2021 RMA rough terrain."""
    return _make("rma_rough", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_rma_stepping_stones(difficulty=0.5, skill_mode="walk",
                               fixed_difficulty=False, render_mode="none", **kw):
    """Kumar 2021 RMA stepping stones."""
    return _make("rma_stepping_stones", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


# ── DreamWaQ ──────────────────────────────────────────────────────────────────

def make_dreamwaq_rough(difficulty=0.5, skill_mode="trot",
                         fixed_difficulty=False, render_mode="none", **kw):
    """Nahrendra 2023 DreamWaQ rough terrain."""
    return _make("dreamwaq_rough", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_dreamwaq_mixed(difficulty=0.5, skill_mode="trot",
                         fixed_difficulty=False, render_mode="none", **kw):
    """Nahrendra 2023 DreamWaQ mixed terrain."""
    return _make("dreamwaq_mixed", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


# ── Walk These Ways ───────────────────────────────────────────────────────────

def make_walk_these_ways_rough(difficulty=0.5, skill_mode=None,
                                fixed_difficulty=False, render_mode="none", **kw):
    """Margolis 2023 Walk These Ways rough terrain (all gaits)."""
    return _make("walk_these_ways_rough", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_walk_these_ways_stairs(difficulty=0.5, skill_mode="trot",
                                 fixed_difficulty=False, render_mode="none", **kw):
    """Margolis 2023 Walk These Ways stair terrain."""
    return _make("walk_these_ways_stairs", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


# ── ANYmal / Hwangbo 2019 ────────────────────────────────────────────────────

def make_anymal_rough(difficulty=0.5, skill_mode="trot",
                       fixed_difficulty=False, render_mode="none", **kw):
    """Hwangbo 2019 ANYmal rough terrain."""
    return _make("anymal_rough", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_anymal_steps(difficulty=0.5, skill_mode="walk",
                       fixed_difficulty=False, render_mode="none", **kw):
    """Hwangbo 2019 ANYmal step obstacle curriculum."""
    return _make("anymal_steps", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


# ── Parkour ───────────────────────────────────────────────────────────────────

def make_parkour_hurdle(difficulty=0.5, skill_mode="run",
                         fixed_difficulty=False, render_mode="none", **kw):
    """Zhuang 2024 Robot Parkour hurdle terrain."""
    return _make("parkour_hurdle", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_parkour_gap(difficulty=0.5, skill_mode="jump",
                      fixed_difficulty=False, render_mode="none", **kw):
    """Zhuang 2024 Robot Parkour gap terrain."""
    return _make("parkour_gap", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_parkour_wall(difficulty=0.5, skill_mode="jump",
                       fixed_difficulty=False, render_mode="none", **kw):
    """Zhuang 2024 Robot Parkour wall terrain."""
    return _make("parkour_wall", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


# ── PGTT ──────────────────────────────────────────────────────────────────────

def make_pgtt_progressive(difficulty=0.5, skill_mode="trot",
                            fixed_difficulty=False, render_mode="none", **kw):
    """Yang 2023 PGTT progressive curriculum terrain."""
    return _make("pgtt_progressive", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


# ── CHRL ──────────────────────────────────────────────────────────────────────

def make_chrl_mixed(difficulty=0.5, skill_mode="trot",
                     fixed_difficulty=False, render_mode="none", **kw):
    """CHRL 2024 mixed challenge terrain."""
    return _make("chrl_mixed", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_chrl_challenge(difficulty=0.8, skill_mode="trot",
                         fixed_difficulty=True, render_mode="none", **kw):
    """CHRL 2024 maximum challenge terrain."""
    return _make("chrl_challenge", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)
