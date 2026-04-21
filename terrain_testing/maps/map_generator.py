"""
Map generators — NumPy heightfield builders.

Each function signature:
    fn(resolution: int, difficulty: float, rng: np.random.RandomState) -> np.ndarray

Returns a (resolution, resolution) float32 array of heights in metres.
No MuJoCo import needed here — pure NumPy / SciPy.

Paper sources: see docs/terrain_sources.md
"""

import math
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    def gaussian_filter(arr, sigma):  # type: ignore
        """Minimal box-blur fallback if scipy not installed."""
        k = max(1, int(sigma * 2) | 1)
        pad = k // 2
        out = arr.copy()
        for _ in range(2):
            padded = np.pad(out, pad, mode="edge")
            out = np.zeros_like(arr)
            for i in range(k):
                out += padded[i:i + arr.shape[0], pad:pad + arr.shape[1]]
            out /= k
        return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(heights: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return gaussian_filter(heights.astype(np.float32), sigma=sigma).astype(np.float32)


def _clip_range(heights: np.ndarray, lo: float = -2.0, hi: float = 2.0) -> np.ndarray:
    return np.clip(heights, lo, hi).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (Isaac Gym / legged_gym — Rudin et al. 2022)
# ─────────────────────────────────────────────────────────────────────────────

def flat(resolution: int, difficulty: float,
         rng: np.random.RandomState) -> np.ndarray:
    """Perfectly flat ground. Baseline."""
    return np.zeros((resolution, resolution), dtype=np.float32)


def pyramid_stairs(resolution: int, difficulty: float,
                   rng: np.random.RandomState) -> np.ndarray:
    """Pyramid-shaped stair terrain from legged_gym (Rudin 2022).

    Stairs rise from the edges toward a central plateau.
    step_height: 0.03–0.22 m based on difficulty.
    step_width: 30–80 cm (finer at high difficulty).
    """
    n = resolution
    step_h = 0.03 + 0.19 * difficulty          # 3–22 cm
    step_w = max(4, int(n * (0.3 - 0.2 * difficulty)))  # finer steps at high d
    heights = np.zeros((n, n), dtype=np.float32)
    cx, cy = n // 2, n // 2
    max_height = 0.50
    for i in range(n):
        for j in range(n):
            dist = max(abs(i - cx), abs(j - cy))
            step_idx = dist // step_w
            heights[i, j] = min(step_idx * step_h, max_height)
    return heights


def discrete_obstacles(resolution: int, difficulty: float,
                        rng: np.random.RandomState) -> np.ndarray:
    """Scattered rectangular blocks (Isaac Gym discrete obstacles).

    Block height: 0.02–0.20 m. Density increases with difficulty.
    Ref: legged_gym terrain_utils.py
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    n_blocks = int(20 + 80 * difficulty)
    max_h = 0.02 + 0.18 * difficulty
    for _ in range(n_blocks):
        bx = rng.randint(0, n - 5)
        by = rng.randint(0, n - 5)
        bw = rng.randint(3, max(4, int(n * 0.12)))
        bd = rng.randint(3, max(4, int(n * 0.12)))
        bh = rng.uniform(0.0, max_h)
        heights[bx:bx + bw, by:by + bd] = bh
    return heights


def wave_terrain(resolution: int, difficulty: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Sinusoidal wave terrain (legged_gym wave curriculum).

    Amplitude: 0.0–0.15 m. Frequency increases with difficulty.
    Ref: legged_gym terrain_utils.py wave_terrain()
    """
    n = resolution
    amplitude = 0.02 + 0.13 * difficulty
    freq_x = 1.0 + 3.0 * difficulty
    freq_y = 0.5 + 2.0 * difficulty
    x = np.linspace(0, 2 * math.pi * freq_x, n)
    y = np.linspace(0, 2 * math.pi * freq_y, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    heights = (amplitude * np.sin(xx) * np.cos(yy)).astype(np.float32)
    return heights


def sloped_terrain(resolution: int, difficulty: float,
                   rng: np.random.RandomState) -> np.ndarray:
    """Uniform slope (legged_gym sloped_terrain).

    Slope angle: 0–25 degrees.
    Direction randomly chosen per episode (controlled by rng).
    """
    n = resolution
    angle_deg = 3.0 + 22.0 * difficulty
    slope = math.tan(math.radians(angle_deg))
    direction = rng.randint(0, 4)   # 0=+x, 1=-x, 2=+y, 3=-y
    heights = np.zeros((n, n), dtype=np.float32)
    idx = np.linspace(0, 1, n)
    if direction == 0:
        heights = np.outer(idx, np.ones(n)).astype(np.float32) * slope
    elif direction == 1:
        heights = np.outer(1 - idx, np.ones(n)).astype(np.float32) * slope
    elif direction == 2:
        heights = np.outer(np.ones(n), idx).astype(np.float32) * slope
    else:
        heights = np.outer(np.ones(n), 1 - idx).astype(np.float32) * slope
    return heights


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (RMA — Kumar et al. 2021)
# ─────────────────────────────────────────────────────────────────────────────

def rma_rough(resolution: int, difficulty: float,
              rng: np.random.RandomState) -> np.ndarray:
    """Random rough terrain as used in RMA (Kumar 2021).

    Gaussian noise σ = 0.005–0.025 m, smoothed.
    A1 robot experiments used σ ≈ 0.02 m (2 cm RMS roughness).
    """
    n = resolution
    sigma = 0.005 + 0.020 * difficulty
    heights = rng.normal(0.0, sigma, (n, n)).astype(np.float32)
    heights = _smooth(heights, sigma=2.0)
    return heights


def rma_stepping_stones(resolution: int, difficulty: float,
                         rng: np.random.RandomState) -> np.ndarray:
    """Stepping stones from RMA supplementary experiments.

    Stone size: 15–30 cm. Gap: 10–30 cm. Stones slightly raised above water.
    """
    n = resolution
    stone_size = max(3, int(n * (0.15 - 0.06 * difficulty)))   # smaller at high d
    gap_size = max(2, int(n * (0.05 + 0.10 * difficulty)))
    base_h = -0.10 * difficulty   # "water" level below stones
    heights = np.full((n, n), base_h, dtype=np.float32)
    ix = 0
    while ix < n:
        iy = 0
        while iy < n:
            sx = min(stone_size, n - ix)
            sy = min(stone_size, n - iy)
            stone_h = rng.uniform(0.0, 0.05 * difficulty)
            heights[ix:ix + sx, iy:iy + sy] = stone_h
            iy += stone_size + gap_size
        ix += stone_size + gap_size + rng.randint(0, max(1, gap_size))
    return heights


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (DreamWaQ — Nahrendra et al. 2023)
# ─────────────────────────────────────────────────────────────────────────────

def dreamwaq_rough(resolution: int, difficulty: float,
                   rng: np.random.RandomState) -> np.ndarray:
    """Rough terrain variant matching DreamWaQ training distribution.

    Uses two-scale noise: coarse + fine. Coarse σ: 0–5 cm.
    Fine σ: 0–1 cm. Matches the proprioceptive history encoding approach.
    """
    n = resolution
    coarse_sigma = 0.01 + 0.04 * difficulty
    fine_sigma = 0.002 + 0.008 * difficulty
    coarse = rng.normal(0, coarse_sigma, (n, n)).astype(np.float32)
    fine = rng.normal(0, fine_sigma, (n, n)).astype(np.float32)
    coarse = _smooth(coarse, sigma=4.0)
    fine = _smooth(fine, sigma=0.8)
    return coarse + fine


def dreamwaq_mixed(resolution: int, difficulty: float,
                   rng: np.random.RandomState) -> np.ndarray:
    """Mixed terrain: rough + isolated steps + small gaps.

    DreamWaQ paper uses a combination during curriculum phase 3.
    """
    n = resolution
    heights = dreamwaq_rough(n, difficulty, rng)
    # Add isolated steps (15 cm max)
    n_steps = int(5 + 10 * difficulty)
    for _ in range(n_steps):
        bx = rng.randint(0, n - 6)
        by = rng.randint(0, n - 6)
        bw = rng.randint(4, 10)
        bh = rng.uniform(0.02, 0.15 * difficulty + 0.02)
        heights[bx:bx + bw, by:by + bw] = bh
    # Add small trenches
    n_trenches = int(2 + 4 * difficulty)
    for _ in range(n_trenches):
        trench_x = rng.randint(n // 4, 3 * n // 4)
        trench_w = max(2, int(4 * difficulty))
        heights[trench_x:trench_x + trench_w, :] -= 0.05 * difficulty
    return heights


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (Walk These Ways — Margolis & Agrawal 2023)
# ─────────────────────────────────────────────────────────────────────────────

def walk_these_ways_rough(resolution: int, difficulty: float,
                           rng: np.random.RandomState) -> np.ndarray:
    """Rough terrain from Walk These Ways multi-gait training.

    Uniform noise ±h, smoothed. h = 0–8 cm. Used for walk/trot/run gait.
    """
    n = resolution
    h = 0.01 + 0.07 * difficulty
    heights = rng.uniform(-h, h, (n, n)).astype(np.float32)
    return _smooth(heights, sigma=1.5)


def walk_these_ways_stairs(resolution: int, difficulty: float,
                            rng: np.random.RandomState) -> np.ndarray:
    """Stair terrain from Walk These Ways.

    Step height: 10–20 cm. Step width: 20–40 cm.
    Direction: ascending in +x. Tested with walk/trot skills.
    """
    n = resolution
    step_h = 0.10 + 0.10 * difficulty
    step_w = max(4, int(n * (0.20 - 0.10 * difficulty)))
    heights = np.zeros((n, n), dtype=np.float32)
    max_height = 0.50  # cap at 50 cm regardless of resolution
    for i in range(n):
        step_idx = i // step_w
        heights[i, :] = min(step_idx * step_h, max_height)
    # Add slight roughness on each step
    roughness = rng.uniform(-0.005, 0.005, (n, n)).astype(np.float32)
    return heights + roughness


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (Hwangbo et al. 2019 — ANYmal)
# ─────────────────────────────────────────────────────────────────────────────

def anymal_rough(resolution: int, difficulty: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Rough terrain from Hwangbo 2019 ANYmal paper.

    2 cm RMS roughness, Gaussian bumps, matches ANYmal training environment.
    """
    n = resolution
    rms = 0.005 + 0.015 * difficulty    # 0.5–2 cm RMS
    heights = rng.normal(0.0, rms, (n, n)).astype(np.float32)
    return _smooth(heights, sigma=1.0)


def anymal_steps(resolution: int, difficulty: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Randomly placed step obstacles (Hwangbo 2019 step curriculum).

    Isolated elevated platforms 5–15 cm high, spread across terrain.
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    n_platforms = int(8 + 12 * difficulty)
    max_h = 0.05 + 0.10 * difficulty
    platform_w = max(4, int(n * 0.08))
    for _ in range(n_platforms):
        px = rng.randint(0, n - platform_w)
        py = rng.randint(0, n - platform_w)
        ph = rng.uniform(0.02, max_h)
        heights[px:px + platform_w, py:py + platform_w] = ph
    return heights


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (Parkour / Robot Parkour Learning — Zhuang 2024)
# ─────────────────────────────────────────────────────────────────────────────

def parkour_hurdle(resolution: int, difficulty: float,
                   rng: np.random.RandomState) -> np.ndarray:
    """Hurdle terrain — thin elevated bars perpendicular to travel direction.

    Hurdle height: 10–40 cm. Spacing: 50–100 cm.
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    hurdle_h = 0.10 + 0.30 * difficulty
    hurdle_spacing = max(10, int(n * (0.3 - 0.1 * difficulty)))
    hurdle_width = max(2, int(n * 0.03))
    pos = hurdle_spacing // 2
    while pos + hurdle_width < n:
        heights[pos:pos + hurdle_width, :] = hurdle_h
        pos += hurdle_spacing
    return heights


def parkour_gap(resolution: int, difficulty: float,
                rng: np.random.RandomState) -> np.ndarray:
    """Gap terrain — deep pits that robot must jump across.

    Gap width: 20–60 cm. Depth: 20–50 cm. Platform width: 60–100 cm.
    """
    n = resolution
    gap_w = max(3, int(n * (0.10 + 0.20 * difficulty)))
    depth = 0.20 + 0.30 * difficulty
    platform_w = max(8, int(n * (0.30 - 0.10 * difficulty)))
    heights = np.zeros((n, n), dtype=np.float32)
    pos = platform_w // 2
    while pos + gap_w < n:
        heights[pos:pos + gap_w, :] = -depth
        pos += gap_w + platform_w
    return heights


def parkour_wall(resolution: int, difficulty: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Vertical wall obstacles robot must climb over.

    Wall height: 20–60 cm. Thickness: 5–15 cm.
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    wall_h = 0.20 + 0.40 * difficulty
    wall_thickness = max(2, int(n * 0.05))
    wall_spacing = max(15, int(n * 0.35))
    pos = wall_spacing // 2
    while pos + wall_thickness < n:
        heights[pos:pos + wall_thickness, :] = wall_h
        pos += wall_spacing
    return heights


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (PGTT progressive curriculum — Yang 2023)
# ─────────────────────────────────────────────────────────────────────────────

def pgtt_progressive(resolution: int, difficulty: float,
                     rng: np.random.RandomState) -> np.ndarray:
    """Progressive terrain combining multiple types in sequence.

    Difficulty gates what sections appear:
      0.0–0.2 → flat
      0.2–0.4 → rough
      0.4–0.6 → slope + rough
      0.6–0.8 → stairs + rough
      0.8–1.0 → stairs + gaps + rough
    Each zone occupies 1/N of the heightfield.
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    n_zones = 4
    zone_size = n // n_zones

    if difficulty < 0.2:
        pass  # all flat
    elif difficulty < 0.4:
        d_norm = (difficulty - 0.2) / 0.2
        heights[zone_size:, :] = rma_rough(n, d_norm, rng)[zone_size:, :]
    elif difficulty < 0.6:
        d_norm = (difficulty - 0.4) / 0.2
        heights[zone_size:2 * zone_size, :] = rma_rough(n, d_norm, rng)[zone_size:2 * zone_size, :]
        slope = sloped_terrain(n, d_norm, rng)
        heights[2 * zone_size:, :] = slope[2 * zone_size:, :]
    elif difficulty < 0.8:
        d_norm = (difficulty - 0.6) / 0.2
        heights[:zone_size, :] = rma_rough(n, 0.3, rng)[:zone_size, :]
        stairs = walk_these_ways_stairs(n, d_norm, rng)
        heights[zone_size:3 * zone_size, :] = stairs[zone_size:3 * zone_size, :]
        heights[3 * zone_size:, :] = rma_rough(n, 0.5, rng)[3 * zone_size:, :]
    else:
        d_norm = (difficulty - 0.8) / 0.2
        heights[:zone_size, :] = anymal_rough(n, 0.5, rng)[:zone_size, :]
        stairs = walk_these_ways_stairs(n, 0.7, rng)
        heights[zone_size:2 * zone_size, :] = stairs[zone_size:2 * zone_size, :]
        gaps = parkour_gap(n, d_norm, rng)
        heights[2 * zone_size:, :] = gaps[2 * zone_size:, :]
    return heights


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TERRAINS  (CHRL — Nature Sci. Reports 2024)
# ─────────────────────────────────────────────────────────────────────────────

def chrl_mixed(resolution: int, difficulty: float,
               rng: np.random.RandomState) -> np.ndarray:
    """Mixed challenge terrain from CHRL survey paper.

    Combines: rough + raised platforms + shallow stairs + small gaps.
    """
    n = resolution
    base = rma_rough(n, difficulty * 0.5, rng)
    platforms = discrete_obstacles(n, difficulty, rng)
    return np.maximum(base, platforms * difficulty)


def chrl_challenge(resolution: int, difficulty: float,
                   rng: np.random.RandomState) -> np.ndarray:
    """Maximum challenge: stairs + gaps + rough (CHRL hardest curriculum stage)."""
    n = resolution
    heights = walk_these_ways_stairs(n, difficulty, rng)
    # Embed gaps every N steps
    gap_interval = max(20, int(n * 0.25 - 10 * difficulty))
    gap_w = max(2, int(4 * difficulty))
    for pos in range(gap_interval, n - gap_w, gap_interval):
        heights[pos:pos + gap_w, :] = -0.15 * difficulty
    noise = rng.normal(0, 0.01, (n, n)).astype(np.float32)
    return heights + noise


# ─────────────────────────────────────────────────────────────────────────────
# === CUSTOM TERRAINS ===
# ─────────────────────────────────────────────────────────────────────────────

def crater_field(resolution: int, difficulty: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Custom C1: Random bowl-shaped depressions + ridges.

    Tests lateral stability. Each crater is a 2D Gaussian dip.
    Difficulty scales crater depth (2–15 cm) and count (5–25).
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    n_craters = int(5 + 20 * difficulty)
    max_depth = 0.02 + 0.13 * difficulty
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    for _ in range(n_craters):
        cx = rng.randint(n // 8, 7 * n // 8)
        cy = rng.randint(n // 8, 7 * n // 8)
        r = rng.uniform(n * 0.05, n * 0.15)
        depth = rng.uniform(0.005, max_depth)
        sign = rng.choice([-1, 1])     # crater or mound
        heights += sign * depth * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2)).astype(np.float32)
    return heights.astype(np.float32)


def sand_dunes(resolution: int, difficulty: float,
               rng: np.random.RandomState) -> np.ndarray:
    """Custom C3: Smooth sinusoidal dunes with varying frequency.

    Tests energy efficiency. Amplitude: 2–12 cm. Multiple overlapping waves.
    """
    n = resolution
    amplitude = 0.02 + 0.10 * difficulty
    heights = np.zeros((n, n), dtype=np.float32)
    n_waves = rng.randint(2, 5)
    x = np.linspace(0, 2 * math.pi, n)
    y = np.linspace(0, 2 * math.pi, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    for _ in range(n_waves):
        freq = rng.uniform(0.5, 2.0 + 2.0 * difficulty)
        phase_x = rng.uniform(0, 2 * math.pi)
        phase_y = rng.uniform(0, 2 * math.pi)
        a = rng.uniform(0.3, 1.0) * amplitude
        heights += a * np.sin(freq * xx + phase_x) * np.cos(freq * yy + phase_y)
    return heights.astype(np.float32)


def rubble_field(resolution: int, difficulty: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Custom C4: Dense random blocks — densest obstacle environment.

    Block height: 1–15 cm. Very high density (~200 blocks at max difficulty).
    Inspired by disaster-response robotics terrain.
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    n_blocks = int(50 + 150 * difficulty)
    max_h = 0.01 + 0.14 * difficulty
    for _ in range(n_blocks):
        bx = rng.randint(0, n - 3)
        by = rng.randint(0, n - 3)
        bw = rng.randint(2, max(3, int(n * 0.06)))
        bd = rng.randint(2, max(3, int(n * 0.06)))
        bh = rng.uniform(0.005, max_h)
        heights[bx:min(bx + bw, n), by:min(by + bd, n)] = bh
    return heights


def asymmetric_slope(resolution: int, difficulty: float,
                     rng: np.random.RandomState) -> np.ndarray:
    """Custom C5: Left side slopes up, right side slopes down.

    Tests roll compensation. Creates a permanent lateral tilt inducement.
    """
    n = resolution
    angle_deg = 3.0 + 15.0 * difficulty
    slope = math.tan(math.radians(angle_deg))
    heights = np.zeros((n, n), dtype=np.float32)
    for j in range(n):
        # Left half: rises left→centre; right half: drops centre→right
        if j < n // 2:
            heights[:, j] = slope * (n // 2 - j) / n
        else:
            heights[:, j] = -slope * (j - n // 2) / n
    # Add a small random rough component
    noise = rng.normal(0, 0.005 * difficulty, (n, n)).astype(np.float32)
    return heights + noise


def frozen_lake(resolution: int, difficulty: float,
                rng: np.random.RandomState) -> np.ndarray:
    """Custom C6: Very low friction flat terrain.

    Terrain itself is flat; friction is handled by the wrapper via
    domain randomization override. Returns near-flat heightfield with
    tiny bumps to avoid perfect numerical zero contact.
    Friction target: 0.05–0.15 (set by wrapper, not this function).
    """
    n = resolution
    # Slight random texture so contacts are detected normally
    heights = rng.uniform(-0.002, 0.002, (n, n)).astype(np.float32)
    return heights


def trench_crossing(resolution: int, difficulty: float,
                    rng: np.random.RandomState) -> np.ndarray:
    """Custom C7: Parallel trenches perpendicular to motion (x-axis).

    Tests stride length adaptation. Trench depth: 5–20 cm.
    Trench width: 5–25 cm. Platform width: 20–40 cm.
    """
    n = resolution
    depth = 0.05 + 0.15 * difficulty
    trench_w = max(3, int(n * (0.05 + 0.12 * difficulty)))
    platform_w = max(8, int(n * (0.20 - 0.08 * difficulty)))
    heights = np.zeros((n, n), dtype=np.float32)
    pos = 0
    on_platform = True
    while pos < n:
        if on_platform:
            end = min(pos + platform_w, n)
        else:
            end = min(pos + trench_w, n)
            heights[pos:end, :] = -depth
        pos = end
        on_platform = not on_platform
    return heights


def tunnel_exit(resolution: int, difficulty: float,
                rng: np.random.RandomState) -> np.ndarray:
    """Custom C2: Flat approach → elevated corridor → drop.

    Tests height adaptation. Corridor height encodes via raised floor.
    First 1/3: flat. Middle 1/3: raised platform (5–20 cm). Last 1/3: step down.
    """
    n = resolution
    heights = np.zeros((n, n), dtype=np.float32)
    corridor_h = 0.05 + 0.15 * difficulty
    corridor_start = n // 3
    corridor_end = 2 * n // 3
    corridor_width_half = int(n * (0.2 - 0.1 * difficulty))
    cy = n // 2
    y0 = max(0, cy - corridor_width_half)
    y1 = min(n, cy + corridor_width_half)
    # Raised floor in corridor
    heights[corridor_start:corridor_end, y0:y1] = corridor_h
    # Narrow the corridor further at high difficulty
    if difficulty > 0.5:
        extra_raise = rng.uniform(0.03, 0.08)
        mid = (y0 + y1) // 2
        half_w = max(2, corridor_width_half // 2)
        heights[corridor_start:corridor_end, mid - half_w:mid + half_w] += extra_raise
    return heights
