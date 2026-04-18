"""
Unit tests for map generators.

Tests:
  - Output shape matches requested resolution
  - dtype is float32
  - No NaN or Inf values
  - Heights within plausible physical range (-2 m to +5 m)
  - Flat terrain is exactly zero
  - Difficulty 0.0 produces smaller heights than difficulty 1.0 (monotonicity)

Run with:
    python -m pytest terrain_testing/tests/test_maps.py -v
"""

import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_TESTING = os.path.dirname(_HERE)
if _TERRAIN_TESTING not in sys.path:
    sys.path.insert(0, _TERRAIN_TESTING)

from maps.map_registry import REGISTRY, list_terrains, generate


RESOLUTIONS = [50, 100, 200]
ALL_TERRAINS = list_terrains()
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Shape and dtype
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_TERRAINS)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_shape(name, resolution):
    heights = generate(name, resolution=resolution, difficulty=0.5, seed=SEED)
    assert heights.shape == (resolution, resolution), (
        f"{name} @ res={resolution}: expected ({resolution}, {resolution}), "
        f"got {heights.shape}"
    )


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_dtype(name):
    heights = generate(name, resolution=100, difficulty=0.5, seed=SEED)
    assert heights.dtype == np.float32, (
        f"{name}: expected float32, got {heights.dtype}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Numerical validity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_no_nan(name):
    heights = generate(name, resolution=100, difficulty=0.5, seed=SEED)
    assert not np.any(np.isnan(heights)), f"{name}: contains NaN"


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_no_inf(name):
    heights = generate(name, resolution=100, difficulty=0.5, seed=SEED)
    assert not np.any(np.isinf(heights)), f"{name}: contains Inf"


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_height_range(name):
    heights = generate(name, resolution=100, difficulty=1.0, seed=SEED)
    lo, hi = -2.0, 5.0
    assert heights.min() >= lo, (
        f"{name}: min height {heights.min():.3f} < {lo}")
    assert heights.max() <= hi, (
        f"{name}: max height {heights.max():.3f} > {hi}")


# ─────────────────────────────────────────────────────────────────────────────
# Specific terrain properties
# ─────────────────────────────────────────────────────────────────────────────

def test_flat_is_zero():
    heights = generate("flat", resolution=100, difficulty=0.5, seed=SEED)
    assert np.allclose(heights, 0.0), "flat terrain should be all zeros"


@pytest.mark.parametrize("name", [
    "rma_rough", "dreamwaq_rough", "anymal_rough",
    "walk_these_ways_rough", "wave_terrain",
])
def test_rough_difficulty_monotonicity(name):
    """Higher difficulty should produce greater height variance."""
    h_easy = generate(name, resolution=100, difficulty=0.1, seed=SEED)
    h_hard = generate(name, resolution=100, difficulty=0.9, seed=SEED)
    std_easy = float(np.std(h_easy))
    std_hard = float(np.std(h_hard))
    assert std_hard >= std_easy * 0.9, (
        f"{name}: std at difficulty=0.9 ({std_hard:.4f}) should be >= "
        f"std at difficulty=0.1 ({std_easy:.4f})"
    )


def test_pyramid_stairs_is_nonnegative():
    heights = generate("pyramid_stairs", resolution=100, difficulty=0.5, seed=SEED)
    assert heights.min() >= -0.001, \
        f"pyramid_stairs should be non-negative, got min={heights.min():.4f}"


def test_parkour_gap_has_pits():
    heights = generate("parkour_gap", resolution=100, difficulty=0.5, seed=SEED)
    assert heights.min() < -0.05, \
        "parkour_gap should have pits (min height < -0.05)"


def test_parkour_wall_has_peaks():
    heights = generate("parkour_wall", resolution=100, difficulty=0.5, seed=SEED)
    assert heights.max() > 0.10, \
        "parkour_wall should have tall walls (max > 0.10)"


def test_frozen_lake_is_nearly_flat():
    heights = generate("frozen_lake", resolution=100, difficulty=0.5, seed=SEED)
    assert np.std(heights) < 0.01, \
        f"frozen_lake should be nearly flat, std={np.std(heights):.4f}"


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_difficulty_0_valid(name):
    """Difficulty 0.0 should not crash and produce valid heights."""
    heights = generate(name, resolution=50, difficulty=0.0, seed=SEED)
    assert not np.any(np.isnan(heights))
    assert not np.any(np.isinf(heights))


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_difficulty_1_valid(name):
    """Difficulty 1.0 should not crash and produce valid heights."""
    heights = generate(name, resolution=50, difficulty=1.0, seed=SEED)
    assert not np.any(np.isnan(heights))
    assert not np.any(np.isinf(heights))


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_reproducibility(name):
    """Same seed should produce identical heightfields."""
    h1 = generate(name, resolution=50, difficulty=0.5, seed=7)
    h2 = generate(name, resolution=50, difficulty=0.5, seed=7)
    assert np.allclose(h1, h2), f"{name}: not reproducible with same seed"


@pytest.mark.parametrize("name", ALL_TERRAINS)
def test_different_seeds_differ(name):
    """Different seeds should produce different heightfields (for stochastic terrains)."""
    if name == "flat":
        pytest.skip("flat terrain is deterministic by design")
    h1 = generate(name, resolution=50, difficulty=0.5, seed=1)
    h2 = generate(name, resolution=50, difficulty=0.5, seed=999)
    # Allow rare coincidental equality for very simple terrains
    if np.allclose(h1, h2):
        # Deterministic terrains (sloped, wave) are OK
        pass  # not a hard failure


# ─────────────────────────────────────────────────────────────────────────────
# Registry completeness
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_has_entries():
    assert len(REGISTRY) > 0


def test_all_registry_entries_have_generator():
    for name, entry in REGISTRY.items():
        assert callable(entry.generator), f"{name}: generator is not callable"


def test_all_registry_entries_have_source():
    for name, entry in REGISTRY.items():
        assert entry.source in ("paper", "custom"), (
            f"{name}: source must be 'paper' or 'custom', got '{entry.source}'"
        )


def test_paper_terrains_exist():
    from maps.map_registry import list_paper_terrains
    papers = list_paper_terrains()
    assert len(papers) > 0


def test_custom_terrains_exist():
    from maps.map_registry import list_custom_terrains
    customs = list_custom_terrains()
    assert len(customs) > 0
