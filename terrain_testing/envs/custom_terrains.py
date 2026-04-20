"""
Convenience constructors for the 7 custom (non-paper) terrain environments.

Usage:
    from envs.custom_terrains import make_rubble_field, make_frozen_lake

    env = make_rubble_field(difficulty=0.7, render_mode="human")
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


def make_crater_field(difficulty=0.5, skill_mode="trot",
                       fixed_difficulty=False, render_mode="none", **kw):
    """Custom C1: Gaussian crater / mound field.
    Tests lateral stability. 5–25 craters, depth 2–15 cm.
    """
    return _make("crater_field", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_tunnel_exit(difficulty=0.5, skill_mode="walk",
                      fixed_difficulty=False, render_mode="none", **kw):
    """Custom C2: Flat → elevated corridor → drop.
    Tests height adaptation. Corridor height 5–20 cm.
    """
    return _make("tunnel_exit", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_sand_dunes(difficulty=0.5, skill_mode="trot",
                     fixed_difficulty=False, render_mode="none", **kw):
    """Custom C3: Overlapping sinusoidal dunes.
    Tests energy efficiency. Amplitude 2–12 cm.
    """
    return _make("sand_dunes", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_rubble_field(difficulty=0.5, skill_mode="walk",
                       fixed_difficulty=False, render_mode="none", **kw):
    """Custom C4: Dense random block rubble.
    Densest obstacle environment, 50–200 small blocks.
    """
    return _make("rubble_field", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_asymmetric_slope(difficulty=0.5, skill_mode="trot",
                            fixed_difficulty=False, render_mode="none", **kw):
    """Custom C5: Left side up, right side down — lateral V-groove.
    Tests roll compensation. Angle 3–15 degrees per side.
    """
    return _make("asymmetric_slope", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_frozen_lake(difficulty=0.5, skill_mode="walk",
                      fixed_difficulty=True, render_mode="none", **kw):
    """Custom C6: Very low friction flat terrain (friction 0.05–0.15).
    Wrapper automatically overrides friction after domain randomization.
    """
    return _make("frozen_lake", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)


def make_trench_crossing(difficulty=0.5, skill_mode="trot",
                           fixed_difficulty=False, render_mode="none", **kw):
    """Custom C7: Parallel perpendicular trenches.
    Tests stride length adaptation. Depth 5–20 cm, width 5–25 cm.
    """
    return _make("trench_crossing", difficulty, skill_mode, fixed_difficulty, render_mode, **kw)
