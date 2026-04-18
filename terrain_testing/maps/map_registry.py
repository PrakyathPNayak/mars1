"""
Map Registry — single source of truth for all terrain names, generators, and metadata.

Usage:
    from maps.map_registry import REGISTRY, list_terrains, get_generator

    # All terrain names
    names = list_terrains()

    # Generator for a terrain
    gen = get_generator("pyramid_stairs")
    heights = gen(resolution=200, difficulty=0.5, rng=np.random.RandomState(42))

    # Full metadata
    entry = REGISTRY["pyramid_stairs"]
    print(entry.source, entry.recommended_skill)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np

from .map_generator import (
    # Paper terrains — Isaac Gym / legged_gym
    flat,
    pyramid_stairs,
    discrete_obstacles,
    wave_terrain,
    sloped_terrain,
    # Paper terrains — RMA
    rma_rough,
    rma_stepping_stones,
    # Paper terrains — DreamWaQ
    dreamwaq_rough,
    dreamwaq_mixed,
    # Paper terrains — Walk These Ways
    walk_these_ways_rough,
    walk_these_ways_stairs,
    # Paper terrains — Hwangbo / ANYmal
    anymal_rough,
    anymal_steps,
    # Paper terrains — Parkour
    parkour_hurdle,
    parkour_gap,
    parkour_wall,
    # Paper terrains — PGTT
    pgtt_progressive,
    # Paper terrains — CHRL
    chrl_mixed,
    chrl_challenge,
    # Custom terrains
    crater_field,
    sand_dunes,
    rubble_field,
    asymmetric_slope,
    frozen_lake,
    trench_crossing,
    tunnel_exit,
)


@dataclass
class TerrainEntry:
    """Metadata for one registered terrain."""
    generator: Callable              # fn(resolution, difficulty, rng) → ndarray
    source: str                      # "paper" | "custom"
    paper: str                       # Short citation or description
    difficulty_range: Tuple[float, float] = (0.0, 1.0)
    recommended_skill: List[str] = field(default_factory=lambda: ["trot"])
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: dict[str, TerrainEntry] = {

    # ── Isaac Gym / legged_gym (Rudin et al. 2022) ──────────────────────────
    "flat": TerrainEntry(
        generator=flat,
        source="paper",
        paper="Baseline flat ground",
        recommended_skill=["walk", "trot", "run", "stand"],
        notes="Use as curriculum starting point.",
    ),
    "pyramid_stairs": TerrainEntry(
        generator=pyramid_stairs,
        source="paper",
        paper="Rudin et al. 2022 — legged_gym pyramid stairs",
        difficulty_range=(0.0, 1.0),
        recommended_skill=["walk", "trot"],
        notes="Stairs radiate from centre. Step 3–22 cm.",
    ),
    "discrete_obstacles": TerrainEntry(
        generator=discrete_obstacles,
        source="paper",
        paper="Rudin et al. 2022 — legged_gym discrete obstacles",
        recommended_skill=["trot", "walk"],
        notes="Random rectangular blocks 2–20 cm.",
    ),
    "wave_terrain": TerrainEntry(
        generator=wave_terrain,
        source="paper",
        paper="Rudin et al. 2022 — legged_gym wave terrain",
        recommended_skill=["trot", "run"],
        notes="Sinusoidal undulations, amplitude 2–15 cm.",
    ),
    "sloped_terrain": TerrainEntry(
        generator=sloped_terrain,
        source="paper",
        paper="Rudin et al. 2022 — legged_gym sloped terrain",
        recommended_skill=["walk", "trot", "run"],
        notes="Uniform slope 3–25 degrees. Direction randomised per episode.",
    ),

    # ── RMA (Kumar et al. 2021) ──────────────────────────────────────────────
    "rma_rough": TerrainEntry(
        generator=rma_rough,
        source="paper",
        paper="Kumar et al. 2021 — RMA rough terrain",
        recommended_skill=["trot", "walk", "run"],
        notes="Two-scale Gaussian noise σ 0.5–2.5 cm. A1 hardware validated.",
    ),
    "rma_stepping_stones": TerrainEntry(
        generator=rma_stepping_stones,
        source="paper",
        paper="Kumar et al. 2021 — RMA stepping stones",
        recommended_skill=["walk", "trot"],
        notes="15–30 cm stones with 10–30 cm gaps. Sub-surface water level.",
    ),

    # ── DreamWaQ (Nahrendra et al. 2023) ────────────────────────────────────
    "dreamwaq_rough": TerrainEntry(
        generator=dreamwaq_rough,
        source="paper",
        paper="Nahrendra et al. 2023 — DreamWaQ rough terrain",
        recommended_skill=["trot", "walk"],
        notes="Coarse + fine dual-scale noise up to 5 cm RMS.",
    ),
    "dreamwaq_mixed": TerrainEntry(
        generator=dreamwaq_mixed,
        source="paper",
        paper="Nahrendra et al. 2023 — DreamWaQ mixed",
        recommended_skill=["trot"],
        notes="Rough + isolated steps + shallow trenches.",
    ),

    # ── Walk These Ways (Margolis & Agrawal 2023) ────────────────────────────
    "walk_these_ways_rough": TerrainEntry(
        generator=walk_these_ways_rough,
        source="paper",
        paper="Margolis & Agrawal 2023 — Walk These Ways rough",
        recommended_skill=["walk", "trot", "run"],
        notes="Uniform noise ±1–8 cm, smoothed. Multi-gait terrain.",
    ),
    "walk_these_ways_stairs": TerrainEntry(
        generator=walk_these_ways_stairs,
        source="paper",
        paper="Margolis & Agrawal 2023 — Walk These Ways stairs",
        recommended_skill=["walk", "trot"],
        notes="10–20 cm steps, 20–40 cm wide. Slight surface roughness.",
    ),

    # ── Hwangbo 2019 / ANYmal ────────────────────────────────────────────────
    "anymal_rough": TerrainEntry(
        generator=anymal_rough,
        source="paper",
        paper="Hwangbo et al. 2019 — ANYmal rough terrain",
        recommended_skill=["trot", "walk"],
        notes="2 cm RMS roughness. Original ANYmal training terrain.",
    ),
    "anymal_steps": TerrainEntry(
        generator=anymal_steps,
        source="paper",
        paper="Hwangbo et al. 2019 — ANYmal step curriculum",
        recommended_skill=["walk", "trot"],
        notes="Random isolated platforms 5–15 cm. Step obstacle curriculum.",
    ),

    # ── Parkour / Zhuang 2024 ────────────────────────────────────────────────
    "parkour_hurdle": TerrainEntry(
        generator=parkour_hurdle,
        source="paper",
        paper="Zhuang et al. 2024 — Robot Parkour Learning hurdle",
        recommended_skill=["run", "jump"],
        notes="Thin hurdles 10–40 cm high, 50–100 cm spacing.",
    ),
    "parkour_gap": TerrainEntry(
        generator=parkour_gap,
        source="paper",
        paper="Zhuang et al. 2024 — Robot Parkour Learning gap",
        recommended_skill=["jump", "run"],
        notes="Deep pits 20–60 cm wide, 20–50 cm deep. Jump required.",
    ),
    "parkour_wall": TerrainEntry(
        generator=parkour_wall,
        source="paper",
        paper="Zhuang et al. 2024 — Robot Parkour Learning wall",
        recommended_skill=["jump"],
        notes="Vertical walls 20–60 cm. Requires full jump behaviour.",
    ),

    # ── PGTT (Yang et al. 2023) ──────────────────────────────────────────────
    "pgtt_progressive": TerrainEntry(
        generator=pgtt_progressive,
        source="paper",
        paper="Yang et al. 2023 — PGTT progressive curriculum",
        recommended_skill=["trot", "walk"],
        notes="Difficulty gates 5 terrain zones in sequence.",
    ),

    # ── CHRL (Nature Sci. Reports 2024) ─────────────────────────────────────
    "chrl_mixed": TerrainEntry(
        generator=chrl_mixed,
        source="paper",
        paper="CHRL 2024 — mixed challenge terrain",
        recommended_skill=["trot", "walk"],
        notes="Rough + platforms combination.",
    ),
    "chrl_challenge": TerrainEntry(
        generator=chrl_challenge,
        source="paper",
        paper="CHRL 2024 — maximum challenge",
        recommended_skill=["trot"],
        notes="Stairs + embedded gaps + noise. Hardest paper terrain.",
    ),

    # ── Custom terrains ──────────────────────────────────────────────────────
    "crater_field": TerrainEntry(
        generator=crater_field,
        source="custom",
        paper="Custom C1: Gaussian crater/mound field",
        recommended_skill=["trot", "walk"],
        notes="Tests lateral stability. 5–25 craters, depth 2–15 cm.",
    ),
    "sand_dunes": TerrainEntry(
        generator=sand_dunes,
        source="custom",
        paper="Custom C3: Overlapping sinusoidal dunes",
        recommended_skill=["trot", "run"],
        notes="Tests energy efficiency. 2–4 overlapping sine waves.",
    ),
    "rubble_field": TerrainEntry(
        generator=rubble_field,
        source="custom",
        paper="Custom C4: Dense random block rubble",
        recommended_skill=["walk", "trot"],
        notes="50–200 small blocks. Densest obstacle environment.",
    ),
    "asymmetric_slope": TerrainEntry(
        generator=asymmetric_slope,
        source="custom",
        paper="Custom C5: Lateral V-groove slope",
        recommended_skill=["trot", "walk"],
        notes="Left side up, right side down. Tests roll compensation.",
    ),
    "frozen_lake": TerrainEntry(
        generator=frozen_lake,
        source="custom",
        paper="Custom C6: Low-friction flat terrain",
        recommended_skill=["walk", "stand"],
        notes="Near-flat. Friction 0.05–0.15 set by wrapper override.",
    ),
    "trench_crossing": TerrainEntry(
        generator=trench_crossing,
        source="custom",
        paper="Custom C7: Parallel perpendicular trenches",
        recommended_skill=["trot", "walk"],
        notes="Tests stride length. Depth 5–20 cm, width 5–25 cm.",
    ),
    "tunnel_exit": TerrainEntry(
        generator=tunnel_exit,
        source="custom",
        paper="Custom C2: Approach → elevated corridor → drop",
        recommended_skill=["walk", "crouch"],
        notes="Tests height adaptation. Corridor height 5–20 cm.",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def list_terrains(source: str = None) -> list[str]:
    """Return all registered terrain names, optionally filtered by source."""
    if source is None:
        return list(REGISTRY.keys())
    return [k for k, v in REGISTRY.items() if v.source == source]


def list_paper_terrains() -> list[str]:
    return list_terrains(source="paper")


def list_custom_terrains() -> list[str]:
    return list_terrains(source="custom")


def get_entry(name: str) -> TerrainEntry:
    if name not in REGISTRY:
        raise KeyError(f"Unknown terrain '{name}'. Available: {list_terrains()}")
    return REGISTRY[name]


def get_generator(name: str) -> Callable:
    return get_entry(name).generator


def generate(name: str, resolution: int = 200, difficulty: float = 0.5,
             seed: int = None) -> np.ndarray:
    """Convenience: generate a named terrain heightfield."""
    rng = np.random.RandomState(seed)
    return get_generator(name)(resolution, difficulty, rng)
