"""
Unitree Go1 Advanced Terrain-Aware Quadruped Environment — v23.

v23: Complete redesign for robust, precise, terrain-aware locomotion.

Robot: Unitree Go1 (from mujoco_menagerie, BSD-3-Clause)

== KEY CHANGES FROM v22 ==
  - Perfect sensors: NO domain randomization, NO sensor noise, NO push perturbations
  - Terrain system: Procedural heightfield (slopes, stairs, rough, gaps, stepping stones)
  - Heightmap observation: 11x11 local terrain scan in body frame
  - Continuous height control: Walk/stand at any height 0.10–0.30m
  - Deterministic jump trajectory: Parabolic arc with smooth ramp
  - Foot height + contact force observations for terrain awareness
  - Terrain curriculum integration: Progressive difficulty
  - Mode redesign: stand, walk, run, jump (crouch = low-height walk/stand)

== DESIGN PRINCIPLES (from 80+ papers) ==
  - RMA (Kumar 2021): Privileged terrain info for training
  - legged_gym (Rudin 2022): Reward structure, termination, gait reward
  - Walk These Ways (Margolis 2023): Multi-skill, per-mode reward multipliers
  - DreamWaQ (2023): Terrain estimation from proprioception
  - PGTT (2023): Progressive terrain curriculum
  - DWL (2408.14472): Auxiliary world model prediction
  - SET (2410.13496): State estimation transformers with causal masking

Observation (196 dims):
  [0:12]    joint positions (rad)
  [12:24]   joint velocities (rad/s)
  [24:27]   base linear velocity (m/s, body frame)
  [27:30]   base angular velocity (rad/s, body frame)
  [30:33]   projected gravity vector (body frame)
  [33:45]   previous action (12)
  [45:49]   command (vx, vy, wz, target_height)
  [49:53]   skill one-hot (stand=0, walk=1, run=2, jump=3)
  [53:59]   command decomposition (speed, dir_sin, dir_cos, lat_frac, turn_frac, lat_sign)
  [59:60]   base height (m)
  [60:64]   foot contacts (binary)
  [64:66]   CPG phase (sin, cos)
  [66:71]   height control (h_err, vz, jump_progress, ramp_active, h_offset)
  [71:75]   foot heights (4, z-pos relative to terrain below each foot)
  [75:196]  local heightmap (11x11=121, heights relative to base, body frame)

Action (12 dims): delta joint position targets, scaled ±0.2 rad
"""
import os
import math
import collections
import numpy as np
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

# ══════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════

NUM_JOINTS = 12
ACT_DIM = 12

# Skill modes — crouch integrated as continuous height parameter
SKILL_MODES = ["stand", "walk", "run", "jump"]
SKILL_DIM = len(SKILL_MODES)       # 4-dim one-hot
SKILL_TO_IDX = {m: i for i, m in enumerate(SKILL_MODES)}

# Heightmap grid
HEIGHTMAP_ROWS = 11
HEIGHTMAP_COLS = 11
HEIGHTMAP_SIZE = HEIGHTMAP_ROWS * HEIGHTMAP_COLS  # 121
HEIGHTMAP_RESOLUTION = 0.1  # meters per cell (covers ±0.55m around robot)

OBS_DIM = 196
# Breakdown:
#   45 proprioception + 4 command + 4 skill + 6 cmd_decomp
#   + 1 base_height + 4 foot_contacts + 2 cpg_phase + 5 height_control
#   + 4 foot_heights + 121 heightmap = 196

# Manual observation normalization (fixed divisors for ~[-1,1] range)
OBS_SCALES = np.concatenate([
    np.array([1.0]*12),         # joint positions
    np.array([10.0]*12),        # joint velocities
    np.array([2.0]*3),          # base linear velocity
    np.array([5.0]*3),          # base angular velocity
    np.array([1.0]*3),          # gravity projection
    np.array([1.0]*12),         # previous action
    np.array([2.0, 0.5, 0.8, 0.35]),   # command (vx, vy, wz, h)
    np.array([1.0]*4),          # skill one-hot
    np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # command decomposition
    np.array([0.35]),           # base height
    np.array([1.0]*4),          # foot contacts
    np.array([1.0]*2),          # CPG phase
    np.array([0.2, 1.0, 1.0, 1.0, 1.0]),  # height control
    np.array([0.3]*4),          # foot heights
    np.array([0.5]*121),        # heightmap (heights relative to base)
]).astype(np.float32)

assert len(OBS_SCALES) == OBS_DIM, f"OBS_SCALES length {len(OBS_SCALES)} != OBS_DIM {OBS_DIM}"

# Default standing pose (Go1 home keyframe)
DEFAULT_STANCE = np.array([
    0.0,  0.9, -1.8,   # FR
    0.0,  0.9, -1.8,   # FL
    0.0,  0.9, -1.8,   # RR
    0.0,  0.9, -1.8,   # RL
], dtype=np.float32)

# ── Height targets ────────────────────────────────────────────────
# Now all modes support continuous height from HEIGHT_MIN to HEIGHT_MAX
HEIGHT_MIN = 0.10       # deep crouch
HEIGHT_MAX = 0.30       # tall standing
HEIGHT_DEFAULT = 0.27   # normal standing
HEIGHT_RAMP_STEPS = 25  # smooth transition over 0.5s

# Jump trajectory
JUMP_TRAJECTORY_STEPS = 60   # 1.2s at 50Hz
JUMP_PEAK_HEIGHT = 0.45      # peak of parabolic arc

# ── Terrain constants ─────────────────────────────────────────────
TERRAIN_TYPES = [
    "flat", "rough", "slope_up", "slope_down",
    "stairs_up", "stairs_down", "gaps", "stepping_stones",
    "random_blocks", "mixed",
]

# Heightfield grid dimensions (must match MuJoCo XML)
HFIELD_NROW = 200
HFIELD_NCOL = 200
HFIELD_SIZE = 5.0       # half-extent in meters (10m x 10m total)
HFIELD_Z_TOP = 2.0      # max elevation
HFIELD_Z_BOT = 0.5      # max depth below reference

# ── Gait constants ────────────────────────────────────────────────
FEET_AIR_TIME_THRESHOLD = 0.15
POSTURE_SIGMA = 0.5
BODY_HEIGHT_SIGMA = 0.10
TRACKING_SIGMA = 0.25
LATERAL_SIGMA = 0.15
HEADING_SIGMA = 0.08

ROBOT_MASS = 12.74  # Go1 approximate total mass (kg)

ABD_JOINT_INDICES  = np.array([0, 3, 6, 9],  dtype=np.int32)
HIP_JOINT_INDICES  = np.array([1, 4, 7, 10], dtype=np.int32)
KNEE_JOINT_INDICES = np.array([2, 5, 8, 11], dtype=np.int32)

# ── Posture targets ───────────────────────────────────────────────
POSTURE_TARGETS = {
    "stand":   {"hip": 0.9,  "knee": -1.8},
    "walk":    {"hip": 0.9,  "knee": -1.8},
    "run":     {"hip": 0.8,  "knee": -1.6},
    "jump":    {"hip": 0.9,  "knee": -1.8},
}

# ── Reward scales (legged_gym / RMA aligned) ─────────────────────
REWARD_SCALES = {
    # Positive tracking rewards
    "r_vel_x":          2.0,
    "r_vel_y":          2.0,
    "r_yaw":            2.5,
    "r_gait":           2.5,
    "r_posture":        1.0,
    "r_body_height":    1.5,
    "r_stillness":      1.5,
    "r_motion_penalty": -1.5,
    "r_vel_track_penalty": 0.0,    # DISABLED — unbounded, use exp() tracking instead
    "r_fwd_vel":        10.0,      # v23h: boosted forward velocity incentive
    "r_jump_phase":     5.0,
    "r_alive":          2.0,       # Increased — anchor rewards positive
    "r_terrain_progress": 1.0,
    "r_foot_clearance":  0.5,
    "r_energy":         -0.0005,   # Light energy penalty
    # Penalties (all bounded or small)
    "r_orientation":   -2.0,
    "r_torque":        -1e-5,
    "r_smooth":        -0.02,
    "r_ang_vel_xy":    -0.05,
    "r_joint_limit":  -10.0,
    "r_lin_vel_z":     -2.0,
    "r_dof_vel":       -5e-5,
    "r_abd_symmetry":  -1.0,
    "r_heading_drift": -1.0,       # Reduced from -3.0
    "r_stumble":        0.0,       # Disabled — heightfield produces non-vertical normals on flat terrain
    "r_standstill":    -3.0,       # v23i: reduced from -8.0 (too harsh with speed gating)
}

# ── Per-mode reward multipliers ──────────────────────────────────
MODE_REWARD_MULTIPLIERS = {
    "stand": {
        "r_vel_x": 0.0, "r_vel_y": 0.0, "r_yaw": 0.0, "r_gait": 0.0,
        "r_posture": 3.0, "r_body_height": 2.0, "r_stillness": 3.0,
        "r_motion_penalty": 1.0, "r_vel_track_penalty": 0.0,
        "r_fwd_vel": 0.0, "r_jump_phase": 0.0,
        "r_terrain_progress": 0.0, "r_foot_clearance": 0.0,
        "r_orientation": 2.0, "r_ang_vel_xy": 2.0, "r_lin_vel_z": 2.0,
        "r_energy": 0.5, "r_stumble": 0.0, "r_standstill": 0.0,
    },
    "walk": {
        "r_vel_x": 3.0, "r_vel_y": 1.5, "r_yaw": 3.0, "r_gait": 2.5,
        "r_posture": 1.0, "r_body_height": 1.0, "r_stillness": 0.0,
        "r_motion_penalty": 0.0, "r_vel_track_penalty": 0.0,
        "r_fwd_vel": 2.0, "r_jump_phase": 0.0,
        "r_alive": 1.5,  # v23i5: ungated alive offsets standstill (3.0 vs -3.0)
        "r_terrain_progress": 1.0, "r_foot_clearance": 1.0,
        "r_smooth": 1.0, "r_ang_vel_xy": 0.3, "r_lin_vel_z": 0.3,
        "r_torque": 0.3, "r_dof_vel": 0.3,
        "r_abd_symmetry": 1.0, "r_heading_drift": 1.0,
        "r_energy": 1.0, "r_stumble": 1.0, "r_standstill": 1.0,
    },
    "run": {
        "r_vel_x": 3.0, "r_vel_y": 1.5, "r_yaw": 3.0, "r_gait": 2.5,
        "r_posture": 1.0, "r_body_height": 1.0, "r_stillness": 0.0,
        "r_motion_penalty": 0.0, "r_vel_track_penalty": 0.0,
        "r_fwd_vel": 2.0, "r_jump_phase": 0.0,
        "r_alive": 0.5,
        "r_terrain_progress": 1.5, "r_foot_clearance": 1.5,
        "r_smooth": 0.5, "r_lin_vel_z": 0.3, "r_ang_vel_xy": 0.3,
        "r_torque": 0.3, "r_dof_vel": 0.3,
        "r_abd_symmetry": 2.0, "r_heading_drift": 1.0,
        "r_energy": 0.5, "r_stumble": 1.0, "r_standstill": 1.0,
    },
    "jump": {
        "r_vel_x": 0.2, "r_vel_y": 0.2, "r_yaw": 0.2, "r_gait": 0.0,
        "r_posture": 0.5, "r_body_height": 3.0,
        "r_stillness": 0.0, "r_motion_penalty": 0.0,
        "r_vel_track_penalty": 0.0, "r_fwd_vel": 0.0,
        "r_jump_phase": 1.0, "r_terrain_progress": 0.0,
        "r_foot_clearance": 0.0, "r_lin_vel_z": 0.1,
        "r_ang_vel_xy": 0.5, "r_energy": 0.0, "r_stumble": 0.0, "r_standstill": 0.0,
    },
}

# ── Termination ───────────────────────────────────────────────────
TERMINATION_GRACE_STEPS = 100      # 2s after reset
MODE_TRANSITION_GRACE_STEPS = 50   # 1s after mode change
COMMAND_RESAMPLE_INTERVAL = 200    # 4s at 50Hz


# ══════════════════════════════════════════════════════════════════════
#  Terrain Generator
# ══════════════════════════════════════════════════════════════════════

class TerrainGenerator:
    """Procedural terrain generation for MuJoCo heightfield.

    Generates 2D height arrays that are applied to MuJoCo hfield_data.
    Supports progressive difficulty via the difficulty parameter [0,1].

    Terrain types (from Isaac Gym / legged_gym conventions):
      - flat: zero-height baseline
      - rough: Gaussian-smoothed random bumps (σ = 0.02–0.08m)
      - slope_up/down: inclined plane (5°–25°)
      - stairs_up/down: rectangular steps (3–20cm height, variable depth)
      - gaps: platform-gap sequences (10–35cm gap width)
      - stepping_stones: discrete foothold patches
      - random_blocks: scattered rectangular obstacles
      - mixed: concatenated sections of different types
    """

    def __init__(self, size: float = 10.0, resolution: int = 200, seed: int = 0):
        self.size = size
        self.resolution = resolution
        self.rng = np.random.RandomState(seed)
        self._heightfield = None

    def generate(self, terrain_type: str = "flat", difficulty: float = 0.5
                 ) -> np.ndarray:
        """Generate heightfield data.

        Args:
            terrain_type: one of TERRAIN_TYPES
            difficulty: 0.0 (easy) to 1.0 (hard)

        Returns:
            heights: (resolution, resolution) float32 array of absolute heights
        """
        n = self.resolution
        heights = np.zeros((n, n), dtype=np.float32)

        if terrain_type == "flat":
            pass

        elif terrain_type == "rough":
            scale = 0.02 + 0.08 * difficulty
            heights = self.rng.uniform(-scale, scale, (n, n)).astype(np.float32)
            # Smooth with Gaussian filter
            try:
                from scipy.ndimage import gaussian_filter
                heights = gaussian_filter(heights, sigma=1.5).astype(np.float32)
            except ImportError:
                # Manual box filter fallback
                kernel = 3
                pad = kernel // 2
                padded = np.pad(heights, pad, mode='edge')
                for i in range(n):
                    for j in range(n):
                        heights[i, j] = padded[i:i+kernel, j:j+kernel].mean()

        elif terrain_type == "slope_up":
            angle = 5 + 20 * difficulty  # degrees
            slope = math.tan(math.radians(angle))
            for i in range(n):
                heights[i, :] = slope * (i / n) * self.size

        elif terrain_type == "slope_down":
            angle = 5 + 20 * difficulty
            slope = math.tan(math.radians(angle))
            for i in range(n):
                heights[i, :] = -slope * (i / n) * self.size

        elif terrain_type == "stairs_up":
            step_h = 0.03 + 0.17 * difficulty   # 3–20cm step height
            step_w = max(4, int(n / (5 + 10 * difficulty)))
            for i in range(n):
                step_idx = i // step_w
                heights[i, :] = step_idx * step_h

        elif terrain_type == "stairs_down":
            step_h = 0.03 + 0.17 * difficulty
            step_w = max(4, int(n / (5 + 10 * difficulty)))
            for i in range(n):
                step_idx = (n - 1 - i) // step_w
                heights[i, :] = step_idx * step_h

        elif terrain_type == "gaps":
            gap_w = max(2, int(3 + 5 * difficulty))
            platform_w = max(8, int(20 - 10 * difficulty))
            depth = 0.1 + 0.3 * difficulty
            pos = 0
            while pos < n:
                end_plat = min(pos + platform_w, n)
                pos = end_plat
                end_gap = min(pos + gap_w, n)
                heights[pos:end_gap, :] = -depth
                pos = end_gap

        elif terrain_type == "stepping_stones":
            stone_size = max(3, int(8 - 4 * difficulty))
            gap_size = max(2, int(2 + 4 * difficulty))
            base_height = -0.15 * difficulty
            heights[:] = base_height
            ix = 0
            while ix < n:
                iy = 0
                while iy < n:
                    sx = min(stone_size, n - ix)
                    sy = min(stone_size, n - iy)
                    h = self.rng.uniform(0.0, 0.1 * difficulty)
                    heights[ix:ix+sx, iy:iy+sy] = h
                    iy += stone_size + gap_size
                ix += stone_size + gap_size

        elif terrain_type == "random_blocks":
            n_blocks = int(10 + 30 * difficulty)
            for _ in range(n_blocks):
                bx = self.rng.randint(0, n - 5)
                by = self.rng.randint(0, n - 5)
                bw = self.rng.randint(3, min(15, n - bx))
                bh_size = self.rng.randint(3, min(15, n - by))
                h = self.rng.uniform(0.02, 0.15 * (1 + difficulty))
                heights[bx:bx+bw, by:by+bh_size] = h

        elif terrain_type == "mixed":
            section_types = self.rng.choice(
                ["rough", "stairs_up", "slope_up", "gaps", "flat"],
                size=4
            )
            quarter = n // 4
            for idx, st in enumerate(section_types):
                sub = self.generate(st, difficulty)
                start = idx * quarter
                end = start + quarter
                heights[start:end, :] = sub[start:end, :]

        self._heightfield = heights
        return heights

    def get_height_at(self, x: float, y: float) -> float:
        """Get terrain height at world coordinate (x, y).

        Coordinate mapping: world (x, y) → grid index.
        The heightfield is centered at origin, half-extent = self.size/2.
        """
        if self._heightfield is None:
            return 0.0
        n = self.resolution
        half = self.size / 2.0
        xi = int(np.clip((x / self.size + 0.5) * n, 0, n - 1))
        yi = int(np.clip((y / self.size + 0.5) * n, 0, n - 1))
        return float(self._heightfield[xi, yi])

    def sample_heightmap(self, x: float, y: float, yaw: float,
                         rows: int = HEIGHTMAP_ROWS,
                         cols: int = HEIGHTMAP_COLS,
                         resolution: float = HEIGHTMAP_RESOLUTION
                         ) -> np.ndarray:
        """Sample local heightmap centered on robot position, rotated into body frame.

        Returns (rows*cols,) flat array of heights relative to base position.
        Grid is oriented with rows along robot's forward direction.
        Vectorized with numpy for performance.

        Args:
            x, y: robot world position
            yaw: robot heading angle (rad)
            rows, cols: grid dimensions
            resolution: meters per cell

        Returns:
            heights: (rows*cols,) float32 array, heights relative to robot base z
        """
        if self._heightfield is None:
            return np.zeros(rows * cols, dtype=np.float32)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Build local grid offsets (vectorized)
        r_offsets = (np.arange(rows) - (rows - 1) / 2.0) * resolution
        c_offsets = (np.arange(cols) - (cols - 1) / 2.0) * resolution
        dx_local, dy_local = np.meshgrid(r_offsets, c_offsets, indexing='ij')

        # Rotate to world frame (vectorized)
        wx = x + cos_yaw * dx_local - sin_yaw * dy_local
        wy = y + sin_yaw * dx_local + cos_yaw * dy_local

        # Convert to grid indices (vectorized)
        n = self.resolution
        xi = np.clip(((wx / self.size + 0.5) * n).astype(np.int32), 0, n - 1)
        yi = np.clip(((wy / self.size + 0.5) * n).astype(np.int32), 0, n - 1)

        return self._heightfield[xi, yi].flatten().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
#  Terrain Curriculum
# ══════════════════════════════════════════════════════════════════════

class TerrainCurriculum:
    """Progressive terrain difficulty based on learning progress.

    Inspired by LP-ACRL (2601.17428) and PGTT (2023).
    Increases difficulty when performance exceeds threshold,
    decreases when performance drops significantly.

    Levels:
      0: flat
      1: slight rough (σ=0.02)
      2: rough (σ=0.05) + slight slope (5°)
      3: rough + moderate slope (15°)
      4: low stairs (5cm)
      5: medium stairs (10cm)
      6: high stairs (15cm) + rough
      7: mixed terrain (max difficulty)
    """

    LEVELS = [
        {"terrain_type": "flat",          "difficulty": 0.0},
        {"terrain_type": "rough",         "difficulty": 0.2},
        {"terrain_type": "rough",         "difficulty": 0.5},
        {"terrain_type": "slope_up",      "difficulty": 0.4},
        {"terrain_type": "stairs_up",     "difficulty": 0.2},
        {"terrain_type": "stairs_up",     "difficulty": 0.5},
        {"terrain_type": "stairs_up",     "difficulty": 0.8},
        {"terrain_type": "mixed",         "difficulty": 0.8},
    ]

    def __init__(self, n_envs: int, advance_threshold: float = 15000.0,
                 retreat_threshold: float = 5000.0, window: int = 50):
        self.n_envs = n_envs
        self.advance_threshold = advance_threshold
        self.retreat_threshold = retreat_threshold
        self.levels = np.zeros(n_envs, dtype=np.int32)
        self.reward_history = [[] for _ in range(n_envs)]
        self._window = window

    def record(self, env_id: int, ep_reward: float):
        """Record episode reward and potentially adjust difficulty."""
        history = self.reward_history[env_id]
        history.append(ep_reward)
        if len(history) > self._window * 2:
            history[:] = history[-self._window * 2:]

        if len(history) >= self._window:
            recent = np.mean(history[-self._window:])
            level = self.levels[env_id]
            if recent > self.advance_threshold and level < len(self.LEVELS) - 1:
                self.levels[env_id] = level + 1
            elif recent < self.retreat_threshold and level > 0:
                self.levels[env_id] = level - 1

    def get_config(self, env_id: int) -> dict:
        return self.LEVELS[self.levels[env_id]]


# ══════════════════════════════════════════════════════════════════════
#  Main Environment
# ══════════════════════════════════════════════════════════════════════

class MiniCheetahEnv(gym.Env):
    """Advanced terrain-aware Unitree Go1 environment — v23.

    Perfect sensors, no domain randomization. Designed for hierarchical
    transformer policy training with terrain awareness.
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str = "none",
        terrain_type: str = "flat",
        terrain_difficulty: float = 0.0,
        use_terrain: bool = True,
        episode_length: int = 2000,
        dt: float = 0.02,
        physics_dt: float = 0.002,
        curriculum: Optional[TerrainCurriculum] = None,
        env_id: int = 0,
        forced_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.terrain_type = terrain_type
        self.terrain_difficulty = terrain_difficulty
        self.use_terrain = use_terrain
        self.episode_length = episode_length
        self.dt = dt
        self.physics_dt = physics_dt
        self.n_substeps = int(dt / physics_dt)
        self.curriculum = curriculum
        self.env_id = env_id

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        # State variables
        self.step_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.command_mode = "stand"
        self.randomize_commands = True
        self.forced_mode = forced_mode  # override random mode selection
        self.target_height = HEIGHT_DEFAULT

        # Feet tracking
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._prev_base_linvel = np.zeros(3, dtype=np.float32)
        self._prev_foot_heights = np.zeros(4, dtype=np.float32)
        self._vx_ema = 0.0  # v23i5: smoothed forward velocity for reward
        self._vy_ema = 0.0

        # CPG phase
        self._cpg_phase = 0.0

        # Height control
        self._effective_target_height = HEIGHT_DEFAULT
        self._height_ramp_from = HEIGHT_DEFAULT
        self._height_ramp_to = HEIGHT_DEFAULT
        self._height_ramp_counter = HEIGHT_RAMP_STEPS

        # Jump trajectory
        self._jump_traj_step = 0
        self._jump_traj_active = False
        self._jump_max_height = 0.0

        # Grace periods
        self._last_mode_change_step = 0

        # PD gains (v23i8: softer PD for natural ground contacts)
        self.kp = 60.0   # was 100.0 — too stiff, kills compliance needed for walking
        self.kd = 0.5    # was 0.0 — damping smooths contact dynamics
        self.max_torque = 33.5

        # Terrain
        self._terrain_gen = None
        self._terrain_height_at_base = 0.0

        # Episode tracking
        self._ep_reward_sum = 0.0
        self._initial_base_pos = np.zeros(3)

        self._init_simulator()

    def _find_model(self):
        """Find the appropriate MuJoCo model file."""
        base = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base))
        if self.use_terrain:
            primary = os.path.join(project_root, "assets", "go1_terrain.xml")
        else:
            primary = os.path.join(project_root, "assets", "go1.xml")
        if os.path.exists(primary):
            return primary
        # Fallback
        fallback = os.path.join(project_root, "assets", "go1.xml")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"Cannot find Go1 model. Looked in: {primary}, {fallback}")

    def _init_simulator(self):
        import mujoco
        import mujoco.viewer
        self._mj = mujoco
        self._mj_viewer = mujoco.viewer
        model_path = self._find_model()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        self.renderer = None

        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Cache foot geom IDs
        _FOOT_GEOM_NAMES = ["FR", "FL", "RR", "RL"]
        self._foot_geom_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in _FOOT_GEOM_NAMES
        ], dtype=np.int32)

        # Cache foot site IDs
        _FOOT_SITE_NAMES = ["FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"]
        self._foot_site_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in _FOOT_SITE_NAMES
        ], dtype=np.int32)

        # Cache heightfield ID if terrain enabled
        self._hfield_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain"
        )
        self._has_hfield = self._hfield_id >= 0

        # Cache floor geom
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

    # ══════════════════════════════════════════════════════════════
    #  Gymnasium API
    # ══════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._prev_base_linvel = np.zeros(3, dtype=np.float32)
        self._prev_foot_heights = np.zeros(4, dtype=np.float32)
        self._vx_ema = 0.0
        self._vy_ema = 0.0
        self._cpg_phase = 0.0
        self._last_mode_change_step = 0
        self._ep_reward_sum = 0.0

        # Reset height control
        self._effective_target_height = HEIGHT_DEFAULT
        self._height_ramp_from = HEIGHT_DEFAULT
        self._height_ramp_to = HEIGHT_DEFAULT
        self._height_ramp_counter = HEIGHT_RAMP_STEPS
        self._jump_traj_step = 0
        self._jump_traj_active = False
        self._jump_max_height = 0.0

        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)

        # Generate terrain
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
        terrain_cfg = self._get_terrain_config(rng)
        self._generate_terrain(terrain_cfg, rng)

        # Compute starting height based on terrain at origin
        start_terrain_h = 0.0
        if self._terrain_gen is not None:
            start_terrain_h = self._terrain_gen.get_height_at(0.0, 0.0)

        # Set initial pose
        self.data.qpos[2] = HEIGHT_DEFAULT + start_terrain_h + 0.02  # small margin
        self.data.qpos[3] = 1.0  # quat w
        self.data.qpos[4:7] = 0.0
        self.data.qpos[7:7 + NUM_JOINTS] = DEFAULT_STANCE
        mujoco.mj_forward(self.model, self.data)

        # v23i9f: Bootstrap locomotion with initial forward velocity.
        # Without CPG, policy has zero gradient from standing→walking.
        # Give robot forward momentum so it must learn to step to stay upright.
        if self.randomize_commands:
            # Will be set after mode selection below, but set a default
            pass

        self._initial_base_pos = self.data.qpos[:3].copy()

        # Randomize command for training
        if self.randomize_commands:
            # Mode weights: train all modes, walk most common
            if self.forced_mode and self.forced_mode in SKILL_MODES:
                self.command_mode = self.forced_mode
            else:
                mode_weights = [0.08, 0.42, 0.25, 0.25]  # stand, walk, run, jump
                self.command_mode = str(rng.choice(SKILL_MODES, p=mode_weights))
            self._randomize_command_for_mode(rng)
            self.target_height = self._effective_target_height

            # v23i9f: Bootstrap locomotion — give initial forward velocity
            # so policy experiences non-zero vx and has gradient to learn from.
            if self.command_mode in ("walk", "run"):
                init_vx = float(rng.uniform(0.1, 0.4))
                self.data.qvel[0] = init_vx
                self._vx_ema = init_vx * 0.5  # pre-seed EMA

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Advance height control
        self._update_height_ramp()
        if self._jump_traj_active:
            self._effective_target_height = self._compute_jump_trajectory(self._jump_traj_step)
        self.target_height = self._effective_target_height

        # Scale action (corrections to posture-centered base)
        action_scaled = action * 0.5  # v23i: increased from 0.25 for walking authority

        # CPG phase tick (for timing observation signal)
        cpg = self._compute_cpg()  # returns zeros but updates _cpg_phase

        # v23i9f: Reference walking trajectory as action center.
        # Replaces disabled CPG. Produces ~0.3 m/s trot pattern.
        # Policy action is a CORRECTION on top of this reference.
        ref = self._compute_walk_reference_action()

        # v23g: Action center adapts to target height (crouch/stand/jump)
        posture = self._get_height_posture(self._effective_target_height)
        center = DEFAULT_STANCE.copy()
        center[HIP_JOINT_INDICES] = posture["hip"]
        center[KNEE_JOINT_INDICES] = posture["knee"]
        target_q = center + action_scaled + ref

        # PD control (perfect actuators)
        q = self.data.qpos[7:7 + NUM_JOINTS]
        qd = self.data.qvel[6:6 + NUM_JOINTS]
        tau = self.kp * (target_q - q) - self.kd * qd
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        self.data.ctrl[:NUM_JOINTS] = tau

        # Physics step
        for _ in range(self.n_substeps):
            self._mj.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        self._ep_reward_sum += reward
        terminated = self._check_done()
        truncated = self.step_count >= self.episode_length

        # Update state
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()
        self.step_count += 1

        # Mid-episode command re-randomization (velocity/height only, keep mode)
        # v23h: mode persists for entire episode — switching modes every 4s was
        # too hard and prevented the policy from learning any single mode well
        if (self.randomize_commands
                and self.step_count % COMMAND_RESAMPLE_INTERVAL == 0):
            rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
            self._randomize_command_for_mode(rng)  # re-randomize vel/height
            self.target_height = self._effective_target_height

        # Report to curriculum on episode end
        if (terminated or truncated) and self.curriculum is not None:
            self.curriculum.record(self.env_id, self._ep_reward_sum)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_info()

    # ══════════════════════════════════════════════════════════════
    #  Terrain generation
    # ══════════════════════════════════════════════════════════════

    def _get_terrain_config(self, rng):
        """Get terrain configuration (from curriculum or random)."""
        if self.curriculum is not None:
            return self.curriculum.get_config(self.env_id)
        if self.use_terrain:
            if self.terrain_type == "random":
                t_type = str(rng.choice(TERRAIN_TYPES))
                t_diff = float(rng.uniform(0.0, 0.8))
                return {"terrain_type": t_type, "difficulty": t_diff}
            return {"terrain_type": self.terrain_type, "difficulty": self.terrain_difficulty}
        return {"terrain_type": "flat", "difficulty": 0.0}

    def _generate_terrain(self, cfg: dict, rng):
        """Generate and apply terrain to MuJoCo heightfield."""
        seed = int(rng.integers(0, 2**31)) if hasattr(rng, 'integers') else int(rng.randint(0, 2**31))
        self._terrain_gen = TerrainGenerator(
            size=HFIELD_SIZE * 2,  # full extent = 2 * half_extent
            resolution=HFIELD_NROW,
            seed=seed,
        )
        heights = self._terrain_gen.generate(cfg["terrain_type"], cfg["difficulty"])

        if self._has_hfield:
            h_min = float(heights.min())
            h_max = float(heights.max())
            h_range = h_max - h_min

            if h_range > 1e-6:
                # Normalize to [0, 1] for MuJoCo
                normalized = (heights - h_min) / h_range
            else:
                normalized = np.full_like(heights, 0.2)  # flat at low reference

            self.model.hfield_data[:HFIELD_NROW * HFIELD_NCOL] = normalized.flatten().astype(np.float32)

            # Set z_top to actual height range
            self.model.hfield_size[self._hfield_id][2] = max(h_range, 0.001)
            self.model.hfield_size[self._hfield_id][3] = 0.01

            # Move floor geom z to account for h_min offset
            if self._floor_geom_id >= 0:
                self.model.geom_pos[self._floor_geom_id][2] = h_min

    # ══════════════════════════════════════════════════════════════
    #  CPG (Central Pattern Generator)
    # ══════════════════════════════════════════════════════════════

    def _compute_cpg(self) -> np.ndarray:
        """Compute CPG trot pattern for walk/run modes.

        Returns 12-dim array of joint offsets from DEFAULT_STANCE.
        """
        cpg = np.zeros(NUM_JOINTS, dtype=np.float32)
        if self.command_mode not in ("walk", "run"):
            return cpg

        t = self.step_count * self.dt
        freq = 2.0  # Hz trot frequency
        phase = 2.0 * math.pi * freq * t
        self._cpg_phase = phase

        amp_hip = 0.0   # v23i9: CPG DISABLED — policy learns gait from scratch
        amp_knee = 0.0  # v23i9: (phase still ticks for timing obs)
        return cpg      # v23i9: all zeros — no CPG offsets, only phase for obs

    def _compute_walk_reference_action(self) -> np.ndarray:
        """v23i9f: Reference walking action trajectory for imitation reward.

        Produces a sinusoidal trot pattern (freq=3Hz, hip=0.4, knee=0.4)
        that generates ~0.3 m/s forward velocity. Used as imitation target
        so the policy has a clear gradient from random actions to walking.
        """
        ref = np.zeros(NUM_JOINTS, dtype=np.float32)
        if self.command_mode not in ("walk", "run"):
            return ref

        t = self.step_count * self.dt
        freq = 3.0  # Hz — best walking frequency found empirically
        phase = 2.0 * math.pi * freq * t
        amp_hip = 0.4
        amp_knee = 0.4

        sin_p = math.sin(phase)
        sin_anti = math.sin(phase + math.pi)

        # Trot: FR+RL vs FL+RR in anti-phase
        ref[1]  = amp_hip * sin_p        # FR hip
        ref[2]  = amp_knee * sin_p       # FR knee
        ref[4]  = amp_hip * sin_anti     # FL hip
        ref[5]  = amp_knee * sin_anti    # FL knee
        ref[7]  = amp_hip * sin_anti     # RR hip
        ref[8]  = amp_knee * sin_anti    # RR knee
        ref[10] = amp_hip * sin_p        # RL hip
        ref[11] = amp_knee * sin_p       # RL knee
        return ref

        sin_p = math.sin(phase)
        sin_p_pi = math.sin(phase + math.pi)

        # Trot: diagonal pairs in anti-phase (FR+RL vs FL+RR)
        cpg[1]  = amp_hip * sin_p       # FR hip
        cpg[2]  = amp_knee * sin_p       # FR knee
        cpg[4]  = amp_hip * sin_p_pi     # FL hip
        cpg[5]  = amp_knee * sin_p_pi    # FL knee
        cpg[7]  = amp_hip * sin_p_pi     # RR hip
        cpg[8]  = amp_knee * sin_p_pi    # RR knee
        cpg[10] = amp_hip * sin_p        # RL hip
        cpg[11] = amp_knee * sin_p       # RL knee

        # Lateral CPG: abductor oscillation for lateral stepping
        amp_abd = 0.25 * min(abs(float(self.command[1])) / 0.3, 1.0)
        abd_sign = 1.0 if float(self.command[1]) > 0 else -1.0
        cos_p = math.cos(phase)
        cos_p_pi = math.cos(phase + math.pi)
        cpg[0]  = amp_abd * abd_sign * cos_p      # FR abductor
        cpg[3]  = amp_abd * abd_sign * cos_p_pi   # FL abductor
        cpg[6]  = amp_abd * abd_sign * cos_p_pi   # RR abductor
        cpg[9]  = amp_abd * abd_sign * cos_p      # RL abductor

        # Scale sagittal CPG based on command activity
        fwd_activity = abs(float(self.command[0])) + abs(float(self.command[2])) * 0.15
        fwd_scale_raw = min(fwd_activity / 0.3, 1.0)
        any_cmd = abs(float(self.command[0])) + abs(float(self.command[1])) + abs(float(self.command[2])) * 0.3
        lat_cmd = abs(float(self.command[1]))
        base_floor = 0.3 if any_cmd > 0.05 else 0.0
        if lat_cmd > 0.1:
            base_floor = max(base_floor, min(lat_cmd / 0.3, 1.0) * 0.6)
        fwd_scale = max(base_floor, fwd_scale_raw)

        for i in [1, 2, 4, 5, 7, 8, 10, 11]:
            cpg[i] *= fwd_scale

        # Yaw: differential abductor lean
        wz_abd = 0.06 * min(abs(float(self.command[2])) / 0.5, 1.0)
        if abs(float(self.command[2])) > 0.05:
            yaw_sign = 1.0 if float(self.command[2]) > 0 else -1.0
            cpg[0]  += wz_abd * yaw_sign
            cpg[3]  -= wz_abd * yaw_sign
            cpg[6]  += wz_abd * yaw_sign
            cpg[9]  -= wz_abd * yaw_sign

        return cpg

    # ══════════════════════════════════════════════════════════════
    #  Observations
    # ══════════════════════════════════════════════════════════════

    def _get_obs(self) -> np.ndarray:
        d = self.data
        qpos = d.qpos[7:7 + NUM_JOINTS].astype(np.float32)
        qvel = d.qvel[6:6 + NUM_JOINTS].astype(np.float32)

        quat = d.qpos[3:7]
        base_linvel = self._quat_rotate_inv(quat, d.qvel[:3]).astype(np.float32)
        base_angvel = self._quat_rotate_inv(quat, d.qvel[3:6]).astype(np.float32)
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))

        base_height = np.array([float(d.qpos[2])], dtype=np.float32)

        # Foot contacts (perfect sensing)
        foot_contacts = np.zeros(4, dtype=np.float32)
        if d.ncon > 0:
            geom1 = d.contact.geom1[:d.ncon]
            geom2 = d.contact.geom2[:d.ncon]
            for j, fid in enumerate(self._foot_geom_ids):
                foot_contacts[j] = float(np.any((geom1 == fid) | (geom2 == fid)))

        # Skill one-hot (4 modes)
        skill_onehot = np.zeros(SKILL_DIM, dtype=np.float32)
        skill_idx = SKILL_TO_IDX.get(self.command_mode, 0)
        skill_onehot[skill_idx] = 1.0

        # CPG phase signal
        if self.command_mode in ("walk", "run"):
            cpg_phase_signal = np.array([
                math.sin(self._cpg_phase),
                math.cos(self._cpg_phase),
            ], dtype=np.float32)
        else:
            cpg_phase_signal = np.zeros(2, dtype=np.float32)

        # Command decomposition
        vx_c = float(self.command[0])
        vy_c = float(self.command[1])
        wz_c = float(self.command[2])
        cmd_speed = math.sqrt(vx_c**2 + vy_c**2)
        cmd_heading = math.atan2(vy_c, vx_c) if cmd_speed > 0.01 else 0.0
        total_cmd = cmd_speed + abs(wz_c)
        cmd_decomp = np.array([
            cmd_speed,
            math.sin(cmd_heading),
            math.cos(cmd_heading),
            abs(vy_c) / max(abs(vx_c) + abs(vy_c), 0.01),
            abs(wz_c) / max(total_cmd, 0.01),
            (1.0 if vy_c > 0.05 else (-1.0 if vy_c < -0.05 else 0.0)),
        ], dtype=np.float32)

        # Height control vector
        height_error = self._effective_target_height - float(d.qpos[2])
        vertical_velocity = float(base_linvel[2])
        jump_phase_progress = (self._jump_traj_step / JUMP_TRAJECTORY_STEPS
                                if self._jump_traj_active else 0.0)
        height_ramp_active = 1.0 if self._height_ramp_counter < HEIGHT_RAMP_STEPS else 0.0
        target_height_offset = (self._effective_target_height - HEIGHT_DEFAULT) / 0.20
        height_control = np.array([
            height_error,
            vertical_velocity,
            jump_phase_progress,
            height_ramp_active,
            target_height_offset,
        ], dtype=np.float32)

        # Foot heights (absolute z position of each foot site)
        foot_heights = np.zeros(4, dtype=np.float32)
        for i, sid in enumerate(self._foot_site_ids):
            foot_z = float(d.site_xpos[sid, 2])
            # Subtract local terrain height to get height above ground
            foot_x = float(d.site_xpos[sid, 0])
            foot_y = float(d.site_xpos[sid, 1])
            terrain_h = self._get_terrain_height(foot_x, foot_y)
            foot_heights[i] = foot_z - terrain_h

        # Local heightmap (11x11 = 121, relative to base height)
        base_x = float(d.qpos[0])
        base_y = float(d.qpos[1])
        base_z = float(d.qpos[2])
        # Extract yaw from quaternion
        yaw = self._quat_to_yaw(quat)
        if self._terrain_gen is not None:
            heightmap = self._terrain_gen.sample_heightmap(base_x, base_y, yaw)
            # Make heights relative to base position
            heightmap = heightmap - base_z
        else:
            heightmap = np.zeros(HEIGHTMAP_SIZE, dtype=np.float32)

        obs = np.concatenate([
            qpos,                   # [0:12]
            qvel,                   # [12:24]
            base_linvel,            # [24:27]
            base_angvel,            # [27:30]
            gravity_body,           # [30:33]
            self.prev_action,       # [33:45]
            np.append(self.command, self._effective_target_height),  # [45:49]
            skill_onehot,           # [49:53]
            cmd_decomp,             # [53:59]
            base_height,            # [59:60]
            foot_contacts,          # [60:64]
            cpg_phase_signal,       # [64:66]
            height_control,         # [66:71]
            foot_heights,           # [71:75]
            heightmap,              # [75:196]
        ]).astype(np.float32)

        # Manual normalization (fixed scales, no running stats)
        obs = obs / OBS_SCALES

        # NO sensor noise — perfect sensors for advanced training
        return obs

    # ══════════════════════════════════════════════════════════════
    #  Reward
    # ══════════════════════════════════════════════════════════════

    def _compute_reward(self, action: np.ndarray) -> float:
        quat = self.data.qpos[3:7]
        base_linvel = self._quat_rotate_inv(quat, self.data.qvel[:3])
        base_angvel = self._quat_rotate_inv(quat, self.data.qvel[3:6])
        tau = self.data.ctrl[:NUM_JOINTS]
        joint_vel = self.data.qvel[6:6 + NUM_JOINTS]
        q = self.data.qpos[7:7 + NUM_JOINTS]
        base_z = float(self.data.qpos[2])

        vx_cmd, vy_cmd, wz_cmd = self.command
        cmd_speed = math.sqrt(vx_cmd**2 + vy_cmd**2)
        mode = self.command_mode

        # ── 1. Velocity tracking ────────────────────────────────
        vx_error = (base_linvel[0] - vx_cmd)**2
        vy_error = (base_linvel[1] - vy_cmd)**2

        vx_sigma_floor = 0.12 if mode == "walk" else TRACKING_SIGMA
        sigma_vx = max(vx_sigma_floor, abs(vx_cmd) * 0.5)
        sigma_vy = max(LATERAL_SIGMA, abs(vy_cmd) * 0.5)
        r_vel_x = math.exp(-vx_error / sigma_vx)
        r_vel_y = math.exp(-vy_error / sigma_vy)

        CMD_ACTIVE_THRESH = 0.1
        vx_cmd_scale = min(abs(vx_cmd) / CMD_ACTIVE_THRESH, 1.0)
        vy_cmd_scale = min(abs(vy_cmd) / CMD_ACTIVE_THRESH, 1.0)
        wz_cmd_scale = min(abs(wz_cmd) / CMD_ACTIVE_THRESH, 1.0)
        any_cmd_active = max(vx_cmd_scale, vy_cmd_scale, wz_cmd_scale)

        if mode in ("walk", "run"):
            # v23i: Only reward tracking when that specific direction is commanded.
            # Old code used any_cmd_active which gave free reward for tracking vy=0/wz=0.
            r_vel_x *= vx_cmd_scale
            r_vel_y *= vy_cmd_scale

        # ── 2. Yaw tracking ────────────────────────────────────
        ang_vel_error = (base_angvel[2] - wz_cmd)**2
        sigma_wz = max(HEADING_SIGMA, abs(wz_cmd) * 0.5)
        r_yaw = math.exp(-ang_vel_error / sigma_wz)
        if mode in ("walk", "run"):
            r_yaw *= wz_cmd_scale

        # ── 3. Gait reward ──────────────────────────────────────
        foot_contacts = np.zeros(4, dtype=bool)
        if self.data.ncon > 0:
            geom1 = self.data.contact.geom1[:self.data.ncon]
            geom2 = self.data.contact.geom2[:self.data.ncon]
            for j, fid in enumerate(self._foot_geom_ids):
                foot_contacts[j] = np.any((geom1 == fid) | (geom2 == fid))

        contact_filt = np.logical_or(foot_contacts, self._last_contacts)
        self._last_contacts = foot_contacts.copy()
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += self.dt

        air_time_reward = float(np.sum(
            (self._feet_air_time - FEET_AIR_TIME_THRESHOLD) * first_contact
        ))
        if cmd_speed < 0.1:
            air_time_reward = 0.0
        self._feet_air_time[contact_filt] = 0.0

        diag1 = float(contact_filt[0] + contact_filt[3])
        diag2 = float(contact_filt[1] + contact_filt[2])
        trot_symmetry = math.sqrt(abs(diag1 - diag2) / 2.0) if cmd_speed > 0.1 else 0.0

        # Foot clearance
        foot_site_heights = self.data.site_xpos[self._foot_site_ids, 2]
        swing_mask = ~contact_filt
        clearance_target = 0.08
        foot_clearance = 0.0
        if np.any(swing_mask) and cmd_speed > 0.1:
            swing_heights = foot_site_heights[swing_mask]
            # Get terrain height under each swing foot
            for idx in np.where(swing_mask)[0]:
                fx = float(self.data.site_xpos[self._foot_site_ids[idx], 0])
                fy = float(self.data.site_xpos[self._foot_site_ids[idx], 1])
                th = self._get_terrain_height(fx, fy)
                swing_heights[np.where(np.where(swing_mask)[0] == idx)[0]] -= th
            foot_clearance = float(np.mean(
                np.clip(swing_heights, 0, clearance_target) / clearance_target
            ))

        n_touchdowns = float(np.sum(first_contact))
        stride_freq_reward = math.exp(-0.5 * (n_touchdowns - 2.0)**2) if cmd_speed > 0.1 else 0.0

        r_gait = (0.3 * air_time_reward + 0.25 * trot_symmetry
                  + 0.25 * foot_clearance + 0.2 * stride_freq_reward)

        # ── 4. Posture tracking ─────────────────────────────────
        hip_q = q[HIP_JOINT_INDICES]
        knee_q = q[KNEE_JOINT_INDICES]
        posture_key = mode if mode in POSTURE_TARGETS else "stand"
        # Continuous height-based posture interpolation
        p_target = self._get_height_posture(self._effective_target_height)
        posture_err = (float(np.sum((hip_q - p_target["hip"])**2))
                       + float(np.sum((knee_q - p_target["knee"])**2)))
        r_posture = math.exp(-posture_err / POSTURE_SIGMA)

        # ── 5. Body height tracking ─────────────────────────────
        terrain_h = self._get_terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1]))
        effective_height = base_z - terrain_h  # height above local terrain
        r_body_height = math.exp(-(effective_height - self._effective_target_height)**2 / BODY_HEIGHT_SIGMA)

        # ── 6. Stillness reward ─────────────────────────────────
        r_stillness = 0.0
        if mode == "stand":
            joint_motion = float(np.mean(joint_vel**2))
            body_motion = float(np.sum(base_linvel[:2]**2))
            ang_motion = float(np.sum(base_angvel[:2]**2))
            r_stillness = 1.0 / (1.0 + joint_motion * 0.01 + body_motion * 2.0 + ang_motion * 0.5)

        # ── 6b. Motion penalty ──────────────────────────────────
        r_motion_penalty = 0.0
        if mode == "stand":
            body_speed_sq = float(np.sum(base_linvel[:2]**2))
            yaw_rate_sq = float(base_angvel[2]**2)
            r_motion_penalty = body_speed_sq + 0.5 * yaw_rate_sq

        # ── 6c. Forward velocity bonus ──────────────────────────
        # v23i5: Use EMA-smoothed velocity to prevent symmetric CPG oscillation
        # from getting free reward via max(0) on instantaneous velocity peaks.
        _EMA_ALPHA = 0.1  # v23i7b: ~10-step window (faster response, still filters CPG cycle)
        self._vx_ema = (1 - _EMA_ALPHA) * self._vx_ema + _EMA_ALPHA * float(base_linvel[0])
        self._vy_ema = (1 - _EMA_ALPHA) * self._vy_ema + _EMA_ALPHA * float(base_linvel[1])
        r_fwd_vel = 0.0
        if mode in ("walk", "run"):
            if abs(vx_cmd) > 0.05:
                r_fwd_vel += self._vx_ema * (1.0 if vx_cmd > 0 else -1.0)
            if abs(vy_cmd) > 0.05:
                r_fwd_vel += self._vy_ema * (1.0 if vy_cmd > 0 else -1.0)
            r_fwd_vel = max(r_fwd_vel, 0.0)

        # ── 6d. Velocity tracking penalty ───────────────────────
        r_vel_track_penalty = 0.0
        if mode in ("walk", "run"):
            r_vel_track_penalty = vx_error + vy_error + ang_vel_error

        # ── 7. Jump phase reward ────────────────────────────────
        r_jump_phase = self._advance_jump(base_z, base_linvel, foot_contacts, terrain_h)

        # ── 8. Alive bonus ──────────────────────────────────────
        r_alive = 1.0

        # ── 9. Orientation penalty ──────────────────────────────
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        r_orientation = float(np.sum(gravity_body[:2]**2))

        # ── 10. Angular velocity xy penalty ─────────────────────
        r_ang_vel_xy = float(base_angvel[0]**2 + base_angvel[1]**2)

        # ── 11. Torque penalty ──────────────────────────────────
        r_torque = float(np.sum(tau**2))

        # ── 12. Action smoothness (L2) ─────────────────────────
        r_smooth = float(np.sum((action - self.prev_action)**2))

        # ── 13. Joint limit proximity ───────────────────────────
        jnt_range = self.model.jnt_range[1:NUM_JOINTS + 1]
        margin = 0.1
        below = np.clip(jnt_range[:, 0] + margin - q, 0, None)
        above = np.clip(q - (jnt_range[:, 1] - margin), 0, None)
        r_joint_limit = float(np.sum(below**2 + above**2))

        # ── 14. Vertical bounce penalty ─────────────────────────
        r_lin_vel_z = float(base_linvel[2]**2)

        # ── 15. Joint velocity penalty ──────────────────────────
        r_dof_vel = float(np.sum(joint_vel**2))

        # ── 16. Abductor symmetry ───────────────────────────────
        abd_left = action[3] + action[9]
        abd_right = action[0] + action[6]
        abd_imbalance = (abd_left - abd_right)**2
        vy_symmetry_gate = 1.0 - min(abs(vy_cmd) / 0.3, 1.0)
        r_abd_symmetry = abd_imbalance * vy_symmetry_gate

        # ── 17. Heading drift penalty ───────────────────────────
        r_heading_drift = 0.0
        if mode in ("walk", "run"):
            r_heading_drift = abs(base_angvel[2] - wz_cmd)

        # ── 18. Terrain progress (v23) ──────────────────────────
        r_terrain_progress = 0.0
        if mode in ("walk", "run") and cmd_speed > 0.1:
            # Reward movement in commanded direction relative to terrain
            cmd_dir = math.atan2(vy_cmd, vx_cmd) if cmd_speed > 0.01 else 0.0
            actual_vel = math.sqrt(float(base_linvel[0])**2 + float(base_linvel[1])**2)
            actual_dir = math.atan2(float(base_linvel[1]), float(base_linvel[0]))
            dir_alignment = math.cos(actual_dir - cmd_dir)
            r_terrain_progress = actual_vel * max(0.0, dir_alignment)

        # ── 19. Foot clearance reward (v23) ─────────────────────
        r_foot_clearance_reward = foot_clearance  # already computed above

        # ── 20. Energy efficiency (v23) ─────────────────────────
        # Cost of transport: mechanical power / (mass * speed)
        mech_power = float(np.sum(np.abs(tau * joint_vel)))
        r_energy = mech_power

        # ── 21. Stumble penalty (v23) ───────────────────────────
        r_stumble = 0.0
        # Detect foot collisions with lateral forces (hitting stairs/edges)
        # Only count contacts between foot geoms and terrain/floor, not self-collisions
        if self.data.ncon > 0:
            floor_id = self._floor_geom_id
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = c.geom1, c.geom2
                # Check if contact involves a foot geom AND the floor/terrain
                foot_involved = g1 in self._foot_geom_ids or g2 in self._foot_geom_ids
                floor_involved = (g1 == floor_id) or (g2 == floor_id)
                if foot_involved and floor_involved:
                    # Large lateral normal = foot hitting edge/side of terrain
                    # Threshold 0.85: only steep lateral contacts count
                    # (heightfield triangulation can produce small normal tilts)
                    if abs(c.frame[0]) > 0.85 or abs(c.frame[1]) > 0.85:
                        r_stumble += 1.0

        # ── 22. Standstill penalty (v23h) ───────────────────────
        # v23i9d: Use EMA velocity, not instantaneous. Prevents oscillation gaming
        # (robot could oscillate ±0.3 m/s with zero net motion but zero penalty).
        r_standstill = 0.0
        if mode in ("walk", "run") and cmd_speed > 0.1:
            ema_speed = abs(self._vx_ema)  # v23i9d: EMA-smoothed forward speed
            speed_frac = min(ema_speed / cmd_speed, 1.0)
            # Quadratic penalty peaks at standstill, zero at full speed
            r_standstill = (1.0 - speed_frac) ** 2

        # Update state
        self._prev_base_linvel = base_linvel.copy()
        self._prev_foot_heights = foot_site_heights.copy()

        # ── Assemble reward ─────────────────────────────────────
        raw_components = {
            "r_vel_x": r_vel_x,
            "r_vel_y": r_vel_y,
            "r_yaw": r_yaw,
            "r_gait": r_gait,
            "r_posture": r_posture,
            "r_body_height": r_body_height,
            "r_stillness": r_stillness,
            "r_motion_penalty": r_motion_penalty,
            "r_vel_track_penalty": r_vel_track_penalty,
            "r_fwd_vel": r_fwd_vel,
            "r_jump_phase": r_jump_phase,
            "r_alive": r_alive,
            "r_terrain_progress": r_terrain_progress,
            "r_foot_clearance": r_foot_clearance_reward,
            "r_energy": r_energy,
            "r_orientation": r_orientation,
            "r_torque": r_torque,
            "r_smooth": r_smooth,
            "r_ang_vel_xy": r_ang_vel_xy,
            "r_joint_limit": r_joint_limit,
            "r_lin_vel_z": r_lin_vel_z,
            "r_dof_vel": r_dof_vel,
            "r_abd_symmetry": r_abd_symmetry,
            "r_heading_drift": r_heading_drift,
            "r_stumble": r_stumble,
            "r_standstill": r_standstill,
        }

        mode_mults = MODE_REWARD_MULTIPLIERS.get(mode, {})
        total = 0.0
        scaled_components = {}

        # v23i9e: Instant vx for learning + EMA for gating/tracking
        # v23i9f: Add alive bonus (+0.3) for stable exploration floor.
        # Bootstrap: initial velocity in reset provides non-zero vx signal.
        if mode in ("walk", "run"):
            vx = float(base_linvel[0])    # instantaneous
            vx_ema = self._vx_ema         # EMA-smoothed

            # Linear velocity: INSTANT for strong per-step gradient (learning)
            r_vx_lin = vx * (1.0 if vx_cmd > 0 else (-1.0 if vx_cmd < 0 else 0.0))
            # Exp tracking: EMA for sustained motion only (no oscillation bonus)
            r_vx_track = math.exp(-16.0 * (vx_ema - vx_cmd)**2)

            # Velocity-gated foot clearance: EMA gate (no exploit)
            vx_gate = min(max(vx_ema, 0.0) / 0.05, 1.0)
            gated_clearance = foot_clearance * vx_gate

            # v23i9f: Reference trajectory is now the action center (added in step()).
            # Zero-action policy automatically walks via the reference.
            # Imitation reward not needed — velocity reward guides corrections.

            total = (
                0.3                       # alive bonus — positive floor for exploration
                + 2.0 * r_vx_lin         # INSTANT forward velocity (strong learning signal)
                + 1.0 * r_vx_track       # EMA tracking (sustained motion bonus)
                + 0.15 * gated_clearance  # gait hint (EMA-gated, exploit-proof)
                - 0.2 * r_standstill     # standstill penalty
                - 2.0 * r_orientation    # stay upright
                - 5e-5 * r_torque        # energy
                - 0.005 * r_smooth       # smooth actions
                - 0.5 * r_lin_vel_z      # don't bounce
                - 0.02 * r_ang_vel_xy    # don't wobble
            )
            scaled_components = {
                "r_alive": 0.3,
                "r_vx_lin": 2.0 * r_vx_lin,
                "r_vx_track": 1.0 * r_vx_track,
                "r_clearance_gated": 0.15 * gated_clearance,
                "r_standstill": -0.2 * r_standstill,
                "r_orientation": -2.0 * r_orientation,
                "r_torque": -5e-5 * r_torque,
                "r_smooth": -0.005 * r_smooth,
                "r_lin_vel_z": -0.5 * r_lin_vel_z,
                "r_ang_vel_xy": -0.02 * r_ang_vel_xy,
                "r_vx_ema": vx_ema,
                "r_total": total,
            }
        else:
            # Original complex reward for stand/jump modes
            _UNGATED = {"r_standstill", "r_heading_drift", "r_fwd_vel", "r_alive"}
            if mode in ("walk", "run") and cmd_speed > 0.1:
                actual_speed = math.sqrt(float(base_linvel[0])**2 + float(base_linvel[1])**2)
                _speed_gate = min(actual_speed / cmd_speed, 1.0)
            else:
                _speed_gate = 1.0

            for k, raw_val in raw_components.items():
                base_scale = REWARD_SCALES.get(k, 0.0)
                mode_mult = mode_mults.get(k, 1.0)
                scaled = base_scale * mode_mult * raw_val
                if k not in _UNGATED:
                    scaled *= _speed_gate
                scaled_components[k] = scaled
                total += scaled

        scaled_components["r_total"] = total
        self._last_reward_components = scaled_components
        return float(total)

    # ══════════════════════════════════════════════════════════════
    #  Termination
    # ══════════════════════════════════════════════════════════════

    def _check_done(self) -> bool:
        if self.step_count < TERMINATION_GRACE_STEPS:
            return False
        steps_since_mode_change = self.step_count - self._last_mode_change_step
        if steps_since_mode_change < MODE_TRANSITION_GRACE_STEPS:
            return False

        base_z = self.data.qpos[2]
        terrain_h = self._get_terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1]))
        height_above_terrain = base_z - terrain_h

        # Terminate if too low (fallen)
        if height_above_terrain < 0.05:
            return True

        # Terminate if too tilted (>60° from vertical)
        quat = self.data.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        tilt = float(np.sum(gravity_body[:2]**2))
        if tilt > 0.75:  # ~60° tilt
            return True

        return False

    # ══════════════════════════════════════════════════════════════
    #  Jump trajectory
    # ══════════════════════════════════════════════════════════════

    def _advance_jump(self, base_z, base_linvel, foot_contacts, terrain_h):
        """Parabolic jump trajectory following with phase-specific bonuses."""
        if self.command_mode != "jump":
            if self._jump_traj_active:
                self._jump_traj_active = False
                self._jump_traj_step = 0
            return 0.0

        if not self._jump_traj_active:
            self._jump_traj_active = True
            self._jump_traj_step = 0
            self._jump_max_height = base_z - terrain_h
            return 0.0

        effective_h = base_z - terrain_h
        self._jump_max_height = max(self._jump_max_height, effective_h)

        traj_h = self._compute_jump_trajectory(self._jump_traj_step)
        phase = self._jump_traj_step / JUMP_TRAJECTORY_STEPS

        r = 0.0
        height_err_sq = (effective_h - traj_h)**2
        r += math.exp(-height_err_sq / 0.02) * 0.5

        # Launch phase: bonus for upward velocity
        if 0.15 < phase < 0.50:
            vz = float(base_linvel[2])
            r += max(0.0, vz) * 3.0  # v23i9f: increased from 2.0

        # Airborne phase: bonus for height
        if 0.30 < phase < 0.65:
            r += max(0.0, effective_h - HEIGHT_DEFAULT) * 5.0  # v23i9f: increased from 3.0
            n_contacts = int(np.sum(foot_contacts))
            if n_contacts == 0:
                r += 1.5  # v23i9f: increased from 1.0

        # Advance
        self._jump_traj_step += 1
        if self._jump_traj_step >= JUMP_TRAJECTORY_STEPS:
            r += max(0.0, self._jump_max_height - 0.30) * 10.0  # v23i9f: increased from 5.0
            self._jump_traj_active = False
            self._jump_traj_step = 0
            self._jump_max_height = base_z - terrain_h
            self._start_height_ramp(HEIGHT_DEFAULT)

        return float(r)

    def _compute_jump_trajectory(self, step):
        """Parabolic jump height trajectory.

        Phase breakdown (60 steps = 1.2s at 50Hz):
          [0,12):   Prep — smooth descent from 0.27 to 0.18
          [12,48):  Arc — parabola peaking at 0.45
          [48,60]:  Land — stabilize at 0.27
        """
        h_stand, h_crouch = HEIGHT_DEFAULT, HEIGHT_MIN + 0.08
        prep_end, arc_end = 12, 48

        if step < 0:
            return h_stand
        elif step < prep_end:
            t = step / prep_end
            s = 3 * t * t - 2 * t * t * t
            return h_stand + (h_crouch - h_stand) * s
        elif step < arc_end:
            t = (step - prep_end) / (arc_end - prep_end)
            return -0.90 * t * t + 0.99 * t + h_crouch
        elif step < JUMP_TRAJECTORY_STEPS:
            return h_stand
        else:
            return h_stand

    # ══════════════════════════════════════════════════════════════
    #  Height control
    # ══════════════════════════════════════════════════════════════

    def _start_height_ramp(self, new_target):
        new_target = float(np.clip(new_target, HEIGHT_MIN, HEIGHT_MAX))
        if abs(new_target - self._effective_target_height) > 0.005:
            self._height_ramp_from = self._effective_target_height
            self._height_ramp_to = new_target
            self._height_ramp_counter = 0
        else:
            self._height_ramp_to = new_target
            self._effective_target_height = new_target

    def _update_height_ramp(self):
        if self._jump_traj_active:
            return
        if self._height_ramp_counter < HEIGHT_RAMP_STEPS:
            self._height_ramp_counter += 1
            t = self._height_ramp_counter / HEIGHT_RAMP_STEPS
            s = 3 * t * t - 2 * t * t * t
            self._effective_target_height = (
                self._height_ramp_from + (self._height_ramp_to - self._height_ramp_from) * s
            )

    def _get_height_posture(self, target_height):
        """Interpolate posture targets based on continuous height."""
        stand_h = HEIGHT_DEFAULT  # 0.27
        deep_h = HEIGHT_MIN      # 0.10
        t = max(0.0, min(1.0, (stand_h - target_height) / (stand_h - deep_h)))
        return {
            "hip": 0.9 + (1.5 - 0.9) * t,      # 0.9 → 1.5 as height decreases
            "knee": -1.8 + (-2.5 - (-1.8)) * t,  # -1.8 → -2.5 as height decreases
        }

    # ══════════════════════════════════════════════════════════════
    #  Command interface
    # ══════════════════════════════════════════════════════════════

    def _randomize_command_for_mode(self, rng):
        mode = self.command_mode
        if mode == "stand":
            vx, vy, wz = 0.0, 0.0, 0.0
            # Random height for stand (tests crouching in place)
            height = float(rng.uniform(HEIGHT_MIN, HEIGHT_MAX))
            self._start_height_ramp(height)
        elif mode == "walk":
            vx = float(rng.uniform(0.05, 0.5))   # v23i9f: wide range for walking
            vy = 0.0   # v23i9b: pure forward walk first — no lateral confusion
            wz = 0.0   # v23i9b: no yaw — learn forward walking first
            # Random height during walking (tests crouched walking)
            height = float(rng.uniform(HEIGHT_MIN + 0.05, HEIGHT_MAX))
            self._start_height_ramp(height)
        elif mode == "run":
            vx = float(rng.uniform(1.5, 3.0))
            vy = float(rng.uniform(-0.5, 0.5))
            wz = float(rng.uniform(-0.5, 0.5))
            # Run height is more restricted (can't run fully crouched)
            height = float(rng.uniform(0.22, HEIGHT_MAX))
            self._start_height_ramp(height)
        elif mode == "jump":
            vx = float(rng.uniform(0.0, 0.5))
            vy = float(rng.uniform(-0.1, 0.1))
            wz = float(rng.uniform(-0.1, 0.1))
            self._jump_traj_active = True
            self._jump_traj_step = 0
        else:
            vx = float(rng.uniform(-0.5, 2.0))
            vy = float(rng.uniform(-0.5, 0.5))
            wz = float(rng.uniform(-0.5, 0.5))
            self._start_height_ramp(HEIGHT_DEFAULT)
        self.command = np.array([vx, vy, wz], dtype=np.float32)

    def set_command(self, vx: float, vy: float, wz: float,
                    mode: str = "walk", height: float = HEIGHT_DEFAULT):
        """Set command for interactive/eval use."""
        self.command = np.array([vx, vy, wz], dtype=np.float32)
        new_mode = mode if mode in SKILL_MODES else "walk"
        if new_mode != self.command_mode:
            self._last_mode_change_step = self.step_count
        self.command_mode = new_mode
        self._start_height_ramp(height)
        if new_mode == "jump" and not self._jump_traj_active:
            self._jump_traj_active = True
            self._jump_traj_step = 0

    def set_exploration_heading(self, heading_rad: float, speed: float = 1.5):
        vx = speed * math.cos(heading_rad)
        vy = speed * math.sin(heading_rad)
        self.set_command(vx, vy, 0.0, "walk")

    # ══════════════════════════════════════════════════════════════
    #  Terrain helpers
    # ══════════════════════════════════════════════════════════════

    def _get_terrain_height(self, x: float, y: float) -> float:
        """Get terrain height at world position."""
        if self._terrain_gen is not None:
            return self._terrain_gen.get_height_at(x, y)
        return 0.0

    # ══════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "step": self.step_count,
            "base_pos": self.data.qpos[:3].tolist(),
            "base_height": float(self.data.qpos[2]),
            "command": self.command.tolist(),
            "mode": self.command_mode,
            "terrain_type": self.terrain_type,
        }
        if hasattr(self, "_last_reward_components"):
            info["reward_components"] = self._last_reward_components
        return info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = self._mj_viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    @staticmethod
    def _quat_rotate_inv(quat, vec):
        """Rotate world-frame vector into body frame. MuJoCo quat = [w,x,y,z]."""
        w, x, y, z = quat
        v = np.array(vec, dtype=np.float64)
        q_vec = np.array([x, y, z], dtype=np.float64)
        t = 2.0 * np.cross(q_vec, v)
        result = v - w * t + np.cross(q_vec, t)
        return result.astype(np.float32)

    @staticmethod
    def _quat_to_yaw(quat):
        """Extract yaw angle from MuJoCo quaternion [w,x,y,z]."""
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
