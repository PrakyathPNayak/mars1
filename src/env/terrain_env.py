"""
Advanced Terrain Environment for Mini Cheetah.

Procedurally generates terrain types and supports multi-skill locomotion:
  - Walking / trotting / running over varied terrain
  - Jumping across gaps
  - Crouching under obstacles
  - Recovery from push perturbations

Terrain types generated via MuJoCo heightfield:
  - Flat, rough, slopes, stairs, gaps, stepping stones, mixed

Inspired by: Isaac Gym terrain generation (Rudin et al. 2022), DreamWaQ (2023),
PGTT (2023), RMA (Kumar et al. 2021).

Reward design references:
  - legged_gym (leggedrobotics/legged_gym): per-term scaling, exponential kernels
  - CHRL (Nature Sci. Rep. 2024): curriculum, domain randomization ranges
  - EIPO (ICRA 2024): constrained energy formulation
  - Kim et al. 2019: Mini Cheetah physical specs (17 Nm peak torque)
"""
import os
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

NUM_JOINTS = 12
OBS_DIM = 57  # Extended obs: base 48 + terrain encoding (8) + phase info (1)
ACT_DIM = 12
BASE_OBS_DIM = 48

DEFAULT_STANCE = np.array([
    0.0,  0.7, -1.4,    # FR
    0.0,  0.7, -1.4,    # FL
    0.0,  0.7, -1.4,    # RR
    0.0,  0.7, -1.4,    # RL
], dtype=np.float32)

# Skill modes with associated reward profiles
# energy_weight: higher = more energy-efficient gait expected (walk > trot > run)
# Ref: Adaptive Energy Regularization (DAEC, arXiv 2024)
SKILL_MODES = {
    "walk":   {"target_speed": 0.8,  "height_target": 0.32, "energy_weight": 0.3},
    "trot":   {"target_speed": 1.5,  "height_target": 0.32, "energy_weight": 0.2},
    "run":    {"target_speed": 2.5,  "height_target": 0.30, "energy_weight": 0.1},
    "jump":   {"target_speed": 0.5,  "height_target": 0.45, "energy_weight": 0.1},
    "crouch": {"target_speed": 0.5,  "height_target": 0.20, "energy_weight": 0.3},
    "stand":  {"target_speed": 0.0,  "height_target": 0.32, "energy_weight": 0.5},
}

# Reward weights — each term's scale is chosen so that at typical operating
# conditions the weighted contribution is O(0.1–1.0) per step.
# See _compute_reward() docstring for per-term analysis.
REWARD_SCALES = {
    "lin_vel_tracking": 1.0,    # max ~2.0 (Gaussian kernel, 2 axes)
    "ang_vel_tracking": 0.5,    # max ~0.5
    "height":          -1.0,    # -(z-target)^2, typ ~0.01 → -0.01
    "orientation":     -1.0,    # -sum(g_xy^2), typ ~0.01 → -0.01
    "lin_vel_z":       -2.0,    # -vz^2, typ ~0.04 → -0.08
    "ang_vel_xy":      -0.05,   # -sum(w_xy^2), typ ~1.0 → -0.05
    "torque":          -2e-5,   # -sum(tau^2), typ ~20000 → -0.4
    "action_rate":     -0.02,   # -sum(da^2), typ ~0.5 → -0.01
    "joint_acc":       -2.5e-7, # -sum(qdd^2), typ ~1e6 → -0.25
    "joint_limit":     -1.0,    # -sum(excess^2), typ ~0 unless near limit
    "contact":         -0.2,    # -(nc-2)^2, typ ~0–0.8
    "terrain":          0.2,    # terrain_diff * fwd_vel, typ ~0.1
    "collision":       -0.5,    # 0 or -0.5 per body collision
    "stumble":         -0.5,    # -0.5 per foot lateral hit
}


class TerrainGenerator:
    """Procedural terrain generation via MuJoCo heightfield.

    Generates terrain patches as 2D height arrays.
    """

    TERRAIN_TYPES = [
        "flat", "rough", "slope_up", "slope_down",
        "stairs_up", "stairs_down", "gaps", "stepping_stones",
        "random_blocks", "mixed",
    ]

    def __init__(self, size: float = 10.0, resolution: int = 200, seed: int = None):
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
            heights: (resolution, resolution) float32 array
        """
        n = self.resolution
        heights = np.zeros((n, n), dtype=np.float32)

        if terrain_type == "flat":
            pass

        elif terrain_type == "rough":
            scale = 0.02 + 0.08 * difficulty
            heights = self.rng.uniform(-scale, scale, (n, n)).astype(np.float32)
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
            step_h = 0.03 + 0.15 * difficulty
            step_w = max(4, int(n / (5 + 10 * difficulty)))
            for i in range(n):
                step_idx = i // step_w
                heights[i, :] = step_idx * step_h

        elif terrain_type == "stairs_down":
            step_h = 0.03 + 0.15 * difficulty
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

    def get_terrain_encoding(self, x: float, y: float, radius: float = 0.5
                             ) -> np.ndarray:
        """Sample terrain features around a position.

        Returns 8-dim encoding: [mean_h, std_h, max_h, min_h,
                                  slope_x, slope_y, roughness, max_gap]

        Coordinate mapping: world (x, y) → grid index via
          idx = clip((coord / size + 0.5) * resolution, 0, resolution-1)
        The heightfield is centered at the geom origin. The MuJoCo hfield
        size parameter defines half-extents, so world range is [-size/2, +size/2].
        """
        if self._heightfield is None:
            return np.zeros(8, dtype=np.float32)

        n = self.resolution
        half_size = self.size / 2.0
        # Clamp robot position to heightfield world bounds before mapping
        x_clamped = float(np.clip(x, -half_size, half_size))
        y_clamped = float(np.clip(y, -half_size, half_size))

        cx = int(np.clip((x_clamped / self.size + 0.5) * n, 0, n - 1))
        cy = int(np.clip((y_clamped / self.size + 0.5) * n, 0, n - 1))
        r = max(1, int(radius / self.size * n))

        x0, x1 = max(0, cx - r), min(n, cx + r)
        y0, y1 = max(0, cy - r), min(n, cy + r)

        patch = self._heightfield[x0:x1, y0:y1]
        if patch.size == 0:
            return np.zeros(8, dtype=np.float32)

        mean_h = patch.mean()
        std_h = patch.std()
        max_h = patch.max()
        min_h = patch.min()

        grad_x = np.gradient(patch, axis=0).mean() if patch.shape[0] > 1 else 0.0
        grad_y = np.gradient(patch, axis=1).mean() if patch.shape[1] > 1 else 0.0

        roughness = std_h
        max_gap = max_h - min_h

        return np.array([
            mean_h, std_h, max_h, min_h,
            grad_x, grad_y, roughness, max_gap
        ], dtype=np.float32)


class AdvancedTerrainEnv(gym.Env):
    """Advanced terrain environment with multi-skill locomotion for Mini Cheetah.

    Observation space (57 dims):
      [0:12]   joint positions (rad)
      [12:24]  joint velocities (rad/s)
      [24:27]  base linear velocity (body frame, m/s)
      [27:30]  base angular velocity (body frame, rad/s)
      [30:33]  projected gravity vector (body frame, unitless)
      [33:45]  previous action (rad)
      [45:48]  velocity command (vx_cmd, vy_cmd, wz_cmd)
      [48:56]  terrain encoding (8 features: mean_h, std_h, max_h, min_h,
               slope_x, slope_y, roughness, max_gap)
      [56]     episode phase (0 to 1)

    Action space (12 dims):
      Delta joint position targets in [-0.5, 0.5] rad, added to DEFAULT_STANCE.
      Tracked by PD controller at physics_dt rate.

    Reward terms (see _compute_reward docstring for details):
      Positive: velocity tracking (lin + ang), terrain traversal bonus
      Negative: height deviation, orientation tilt, z-velocity, xy angular velocity,
                torque, action rate, joint acceleration, joint limit violation,
                contact pattern, body collision, stumble

    Terrain types: flat, rough, slope_up, slope_down, stairs_up, stairs_down,
                   gaps, stepping_stones, random_blocks, mixed

    Skill modes: walk, trot, run, jump, crouch, stand
      Each mode has target_speed, height_target, and energy_weight.

    Domain randomization (when enabled):
      - Body masses: ±15% per body (RMA, DreamWaQ convention)
      - Floor friction: 0.3–1.5 (CHRL survey consensus)
      - Foot friction: 0.8–2.0
      - Joint damping: ±20% (CHRL, Dc-Gait)
      - Joint armature: ±20%
      - PD gains: kp ±15%, kd ±15% (DAEC, CHRL)
      - Motor strength (max_torque): ±10% (RMA)
      - Observation noise: Gaussian σ=0.02

    Push perturbation:
      Applied via xfrc_applied (proper external force through MuJoCo integrator).
      Horizontal only, magnitude configurable. Default: 50N every 200 steps (~4s).
      Ref: CHRL uses up to 80N every 5–10s.

    PD controller:
      kp=80, kd=2.0, max_torque=17 Nm
      Ref: Kim et al. 2019, Mini Cheetah proprietary actuators, 17 Nm peak.
           kd raised from 1.0 to 2.0 to reduce oscillation (legged_gym convention).

    Usage:
      env = AdvancedTerrainEnv(render_mode="human", skill_mode="trot")
      obs, info = env.reset()
      for _ in range(1000):
          action = env.action_space.sample()
          obs, reward, terminated, truncated, info = env.step(action)
          if terminated or truncated:
              obs, info = env.reset()
      env.close()
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str = "none",
        terrain_type: str = "flat",
        difficulty: float = 0.0,
        skill_mode: str = "trot",
        randomize_domain: bool = True,
        randomize_terrain: bool = True,
        randomize_skill: bool = True,
        episode_length: int = 2000,
        push_interval: int = 200,
        push_magnitude: float = 50.0,
        dt: float = 0.02,
        physics_dt: float = 0.002,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.terrain_type = terrain_type
        self.difficulty = difficulty
        self.skill_mode = skill_mode
        self.randomize_domain = randomize_domain
        self.randomize_terrain = randomize_terrain
        self.randomize_skill = randomize_skill
        self.episode_length = episode_length
        self.push_interval = push_interval
        self.push_magnitude = push_magnitude  # Newtons (was 0.5 m/s velocity hack)
        self.dt = dt
        self.physics_dt = physics_dt
        self.n_substeps = int(dt / physics_dt)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(ACT_DIM,), dtype=np.float32
        )

        self.step_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.command_mode = "trot"
        self.randomize_commands = True

        # PD gains — kd raised from 1.0 to 2.0 to suppress oscillation.
        # Ref: legged_gym uses kd=2.0–4.0 for similar-scale quadrupeds.
        # Kim et al. 2019: Mini Cheetah peak torque = 17 Nm.
        self.kp = 80.0
        self.kd = 2.0
        self.max_torque = 17.0

        # Base values saved for domain randomization reset
        self._base_kp = self.kp
        self._base_kd = self.kd
        self._base_max_torque = self.max_torque
        self._base_masses = None
        self._base_damping = None
        self._base_armature = None

        self.terrain_gen = TerrainGenerator()
        self._terrain_encoding = np.zeros(8, dtype=np.float32)

        # Foot contact tracking
        self._foot_contacts = np.zeros(4, dtype=np.float32)
        self._contact_history = []

        # Previous joint velocities for acceleration computation
        self._prev_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float64)

        # Body collision tracking (non-foot geoms touching ground)
        self._body_collision_count = 0
        self._stumble_count = 0

        # Cache foot geom ids and body (non-foot) geom ids after init
        self._foot_geom_ids = set()
        self._foot_geom_id_to_idx = {}  # geom_id → foot index (0..3)
        self._body_geom_ids = set()

        self._init_simulator()

    def _find_model(self):
        base = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base))
        terrain_xml = os.path.join(project_root, "assets", "mini_cheetah_terrain.xml")
        if os.path.exists(terrain_xml):
            return terrain_xml
        primary = os.path.join(project_root, "assets", "mini_cheetah.xml")
        if os.path.exists(primary):
            return primary
        raise FileNotFoundError(f"Cannot find mini_cheetah_terrain.xml or mini_cheetah.xml")

    def _init_simulator(self):
        import mujoco
        import mujoco.viewer
        self._mj = mujoco
        self._mj_viewer = mujoco.viewer
        model_path = self._find_model()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._base_masses = self.model.body_mass.copy()
        self._base_damping = self.model.dof_damping.copy()
        self._base_armature = self.model.dof_armature.copy()

        # Cache hfield id for terrain updates
        self._hfield_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain"
        )
        self._has_hfield = self._hfield_id >= 0
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        # Cache base body id for xfrc_applied push
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )

        # Build geom id sets for collision detection
        foot_names_ordered = ["FR_foot_collision", "FL_foot_collision",
                              "RR_foot_collision", "RL_foot_collision"]
        foot_names = set(foot_names_ordered)
        floor_names = {"floor"}
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name in foot_names:
                self._foot_geom_ids.add(i)
                self._foot_geom_id_to_idx[i] = foot_names_ordered.index(name)
            elif name and name not in floor_names and "visual" not in name:
                self._body_geom_ids.add(i)

        # Cache floor geom id as int for contact checks
        self._floor_geom_id_int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        self.viewer = None
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._foot_contacts = np.zeros(4, dtype=np.float32)
        self._contact_history = []
        self._prev_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float64)
        self._body_collision_count = 0
        self._stumble_count = 0

        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)

        # Reset all randomized model parameters to base values
        self.model.body_mass[:] = self._base_masses
        self.model.dof_damping[:] = self._base_damping
        self.model.dof_armature[:] = self._base_armature
        self.kp = self._base_kp
        self.kd = self._base_kd
        self.max_torque = self._base_max_torque

        # Randomize terrain
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
        if self.randomize_terrain:
            terrain_types = TerrainGenerator.TERRAIN_TYPES
            self.terrain_type = str(rng.choice(terrain_types))
            self.difficulty = float(rng.uniform(0.0, 1.0))
        self.terrain_gen = TerrainGenerator(seed=int(rng.integers(0, 2**31)) if hasattr(rng, 'integers') else int(rng.randint(0, 2**31)))
        heights = self.terrain_gen.generate(self.terrain_type, self.difficulty)

        # Apply heightfield to MuJoCo physics
        start_terrain_height = 0.0
        if self._has_hfield:
            h_min, h_max = float(heights.min()), float(heights.max())
            h_range = h_max - h_min
            if h_range > 1e-6:
                normalized = (heights - h_min) / h_range
            else:
                normalized = np.zeros_like(heights)
            self.model.hfield_data[:] = normalized.flatten().astype(np.float32)
            self.model.hfield_size[self._hfield_id][2] = max(h_range, 0.001)
            self.model.hfield_size[self._hfield_id][3] = 0.01
            if self._floor_geom_id >= 0:
                self.model.geom_pos[self._floor_geom_id][2] = h_min
            n = self.terrain_gen.resolution
            center = n // 2
            r = max(1, n // 20)
            patch = heights[center - r:center + r, center - r:center + r]
            start_terrain_height = float(patch.mean()) if patch.size > 0 else 0.0

        # Set initial pose — adjust for terrain height
        self.data.qpos[2] = 0.32 + start_terrain_height
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.qpos[7:7 + NUM_JOINTS] = DEFAULT_STANCE
        mujoco.mj_forward(self.model, self.data)

        if self.randomize_domain:
            self._apply_domain_randomization()

        # Randomize skill mode
        if self.randomize_skill:
            self.skill_mode = str(rng.choice(list(SKILL_MODES.keys())))

        # Randomize velocity command based on skill
        if self.randomize_commands:
            skill = SKILL_MODES[self.skill_mode]
            speed = skill["target_speed"]
            vx = float(rng.uniform(speed * 0.5, speed * 1.5))
            vy = float(rng.uniform(-0.3, 0.3))
            wz = float(rng.uniform(-0.3, 0.3))
            self.command = np.array([vx, vy, wz], dtype=np.float32)

        return self._get_obs(), {"skill_mode": self.skill_mode, "terrain_type": self.terrain_type}

    def step(self, action: np.ndarray):
        action = np.clip(action, -0.5, 0.5).astype(np.float32)
        target_q = DEFAULT_STANCE + action

        mujoco = self._mj
        q = self.data.qpos[7:7 + NUM_JOINTS]
        qd = self.data.qvel[6:6 + NUM_JOINTS]
        tau = self.kp * (target_q - q) - self.kd * qd
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        self.data.ctrl[:NUM_JOINTS] = tau

        # Apply random push perturbation at intervals via xfrc_applied
        # (proper external force through MuJoCo integrator, not velocity hack)
        if (self.push_interval > 0 and self.step_count > 0 and
                self.step_count % self.push_interval == 0):
            self._apply_push()

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Clear any applied external forces after stepping
        self.data.xfrc_applied[:] = 0.0

        # Update foot contacts and collision detection
        self._update_foot_contacts()
        self._update_collision_info()

        # Update terrain encoding based on robot position
        pos = self.data.qpos[:2]
        self._terrain_encoding = self.terrain_gen.get_terrain_encoding(
            float(pos[0]), float(pos[1])
        )

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_done()
        truncated = self.step_count >= self.episode_length

        # Save state for next step
        self._prev_joint_vel = self.data.qvel[6:6 + NUM_JOINTS].copy()
        self.prev_action = action.copy()
        self.step_count += 1

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Build 57-dim observation vector.

        Velocities are rotated into body frame for sim-to-real transfer
        consistency (DreamWaQ, RMA convention).
        """
        d = self.data
        qpos = d.qpos[7:7 + NUM_JOINTS].astype(np.float32)
        qvel = d.qvel[6:6 + NUM_JOINTS].astype(np.float32)
        quat = d.qpos[3:7]

        # Rotate base velocities into body frame (critical for sim-to-real)
        world_linvel = d.qvel[:3]
        world_angvel = d.qvel[3:6]
        base_linvel = self._quat_rotate_inv(quat, world_linvel).astype(np.float32)
        base_angvel = self._quat_rotate_inv(quat, world_angvel).astype(np.float32)

        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))

        # Phase: 0 to 1 within episode
        phase = np.array([self.step_count / max(self.episode_length, 1)], dtype=np.float32)

        obs = np.concatenate([
            qpos, qvel, base_linvel, base_angvel,
            gravity_body, self.prev_action, self.command,
            self._terrain_encoding,  # 8 terrain features
            phase,                   # 1 phase feature
        ]).astype(np.float32)

        if self.randomize_domain:
            obs[:BASE_OBS_DIM] += self.np_random.standard_normal(BASE_OBS_DIM).astype(np.float32) * 0.02

        return obs

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute per-step reward as a weighted sum of terms.

        All terms are designed so their weighted contribution at typical
        operating conditions is O(0.1–1.0) per step. Weights are stored
        in REWARD_SCALES.

        Term-by-term analysis at convergence (trot, 1.5 m/s, flat terrain):
        ───────────────────────────────────────────────────────────────────
        r_linvel:      exp kernel per axis, max 2.0. At convergence ~1.8.
                       Scale 1.0 → contribution ~1.8
        r_yaw:         exp kernel, max 0.5. At convergence ~0.45.
                       Scale 0.5 → contribution ~0.23
        r_height:      (z - 0.32)², typ ~0.001–0.01.
                       Scale -1.0 → contribution ~-0.005
        r_orientation: sum(g_xy²), typ ~0.01 upright.
                       Scale -1.0 → contribution ~-0.01
        r_lin_vel_z:   vz², typ ~0.02–0.04 during trot.
                       Scale -2.0 → contribution ~-0.06
        r_ang_vel_xy:  sum(w²), typ ~0.5–2.0.
                       Scale -0.05 → contribution ~-0.05
        r_torque:      sum(τ²), 12 joints at ~10 Nm → ~1200.
                       Scale -2e-5 → contribution ~-0.024
        r_action_rate: sum((a-a_prev)²), typ ~0.1–0.5.
                       Scale -0.02 → contribution ~-0.005
        r_joint_acc:   sum(qdd²), typ ~1e5–1e6.
                       Scale -2.5e-7 → contribution ~-0.1
        r_joint_limit: sum(excess²), typ 0 unless near limit.
                       Scale -1.0 → contribution ~0.0
        r_contact:     (nc-2)², range 0–4.
                       Scale -0.2 → contribution ~-0.1
        r_terrain:     diff * fwd_vel, typ 0–0.5.
                       Scale 0.2 → contribution ~0.05
        r_collision:   0 or 1 body collision events.
                       Scale -0.5 → contribution ~0.0
        r_stumble:     0 or 1 lateral foot hits.
                       Scale -0.5 → contribution ~0.0
        ───────────────────────────────────────────────────────────────────
        Net per-step reward at convergence: ~1.5–2.0
        Net per-step reward at start (random policy): ~0.2–0.5

        Removed terms:
        - r_time: was -(episode_length - 500)*0.1 = -450 per step for first
          500 steps. This dominated all other terms by 2–3 orders of magnitude,
          making early-episode learning impossible. Removed entirely.
          Ref: legged_gym, RMA, Isaac Gym all omit survival/time bonuses.
        """
        # Body-frame velocities (consistent with observation frame).
        # Isaac Gym / legged_gym convention: commands are body-frame-relative,
        # so reward tracking must also be in body frame.
        quat = self.data.qpos[3:7]
        base_linvel = self._quat_rotate_inv(quat, self.data.qvel[:3])
        base_angvel = self._quat_rotate_inv(quat, self.data.qvel[3:6])
        tau = self.data.ctrl[:NUM_JOINTS]
        qd = self.data.qvel[6:6 + NUM_JOINTS]
        skill = SKILL_MODES[self.skill_mode]

        vx_cmd, vy_cmd, wz_cmd = self.command
        scales = REWARD_SCALES

        # -- Velocity tracking (exp kernel, legged_gym standard) --
        # sigma²=0.25 → at 0.5 m/s error, reward ≈ 0.37; at 0 error, reward = 1.0
        # Computed in body frame: command vx/vy are body-relative.
        r_linvel = scales["lin_vel_tracking"] * (
            math.exp(-((base_linvel[0] - vx_cmd) ** 2) / 0.25) +
            math.exp(-((base_linvel[1] - vy_cmd) ** 2) / 0.25)
        )
        r_yaw = scales["ang_vel_tracking"] * math.exp(
            -((base_angvel[2] - wz_cmd) ** 2) / 0.25
        )

        # -- Height tracking (skill-dependent, squared penalty) --
        base_z = float(self.data.qpos[2])
        target_h = skill["height_target"]
        r_height = scales["height"] * (base_z - target_h) ** 2

        # -- Orientation: penalise body tilt (projected gravity xy components) --
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        r_orientation = scales["orientation"] * float(np.sum(gravity_body[:2] ** 2))

        # -- Z-velocity penalty: suppress vertical bouncing --
        r_lin_vel_z = scales["lin_vel_z"] * float(base_linvel[2] ** 2)

        # -- XY angular velocity penalty: suppress roll/pitch oscillation --
        r_ang_vel_xy = scales["ang_vel_xy"] * float(np.sum(base_angvel[:2] ** 2))

        # -- Energy efficiency: torque² penalty, scaled by skill energy_weight --
        # Ref: EIPO, DAEC use |τ·ω| (mechanical power). τ² is simpler and
        # correlates well for position-controlled joints.
        energy_w = skill["energy_weight"]
        r_torque = scales["torque"] * energy_w * float(np.sum(tau ** 2))

        # -- Action rate: penalise jerky actions (smoothness) --
        r_action_rate = scales["action_rate"] * float(np.sum((action - self.prev_action) ** 2))

        # -- Joint acceleration penalty: suppress high-frequency oscillation --
        # Ref: legged_gym dof_acc penalty. Computed as Δqvel / dt.
        joint_acc = (qd - self._prev_joint_vel) / self.dt
        r_joint_acc = scales["joint_acc"] * float(np.sum(joint_acc ** 2))

        # -- Joint limit penalty: penalise proximity to joint limits --
        joint_pos = self.data.qpos[7:7 + NUM_JOINTS]
        joint_lower = self.model.jnt_range[1:1 + NUM_JOINTS, 0]  # skip freejoint
        joint_upper = self.model.jnt_range[1:1 + NUM_JOINTS, 1]
        margin = 0.1  # rad, start penalising 0.1 rad from limit
        lower_violation = np.clip(joint_lower + margin - joint_pos, 0.0, None)
        upper_violation = np.clip(joint_pos - (joint_upper - margin), 0.0, None)
        r_joint_limit = scales["joint_limit"] * float(
            np.sum(lower_violation ** 2) + np.sum(upper_violation ** 2)
        )

        # -- Foot contact reward: encourage rhythmic stepping --
        contact_sum = float(np.sum(self._foot_contacts))
        if self.skill_mode in ("walk", "trot", "run"):
            # Prefer 2 feet on ground (diagonal for trot)
            r_contact = scales["contact"] * (contact_sum - 2.0) ** 2
        elif self.skill_mode == "jump":
            # Reward airborne phases
            r_contact = -scales["contact"] * max(0, 1.0 - contact_sum)
        elif self.skill_mode == "crouch":
            # Reward all feet on ground (stability)
            r_contact = -scales["contact"] * min(contact_sum, 4.0) / 4.0
        else:
            r_contact = 0.0

        # -- Terrain adaptability: reward forward progress on difficult terrain --
        # Uses continuous body-frame forward velocity (consistent with tracking reward)
        terrain_difficulty = self._terrain_encoding[1] + self._terrain_encoding[7]
        fwd_vel = float(np.clip(base_linvel[0], 0.0, 3.0))
        r_terrain = scales["terrain"] * terrain_difficulty * fwd_vel

        # -- Body collision penalty: trunk/thigh touching ground --
        r_collision = scales["collision"] * float(self._body_collision_count > 0)

        # -- Stumble penalty: foot lateral contact force --
        r_stumble = scales["stumble"] * float(self._stumble_count > 0)

        total = (r_linvel + r_yaw + r_height + r_orientation +
                 r_lin_vel_z + r_ang_vel_xy + r_torque + r_action_rate +
                 r_joint_acc + r_joint_limit + r_contact + r_terrain +
                 r_collision + r_stumble)

        self._last_reward_components = {
            "r_linvel": r_linvel,
            "r_yaw": r_yaw,
            "r_height": r_height,
            "r_orientation": r_orientation,
            "r_lin_vel_z": r_lin_vel_z,
            "r_ang_vel_xy": r_ang_vel_xy,
            "r_torque": r_torque,
            "r_action_rate": r_action_rate,
            "r_joint_acc": r_joint_acc,
            "r_joint_limit": r_joint_limit,
            "r_contact": r_contact,
            "r_terrain": r_terrain,
            "r_collision": r_collision,
            "r_stumble": r_stumble,
            "r_total": total,
        }

        return total

    def _check_done(self) -> bool:
        """Terminate episode if robot has fallen.

        Conditions (legged_gym / CHRL convention):
          1. Base height < 0.15 m (body hitting ground)
          2. Body tilt > ~45° (gravity_body[2] > -0.7)
             When upright, gravity_body = [0,0,-1], so gz = -1.0.
             At 45° tilt, gz ≈ -cos(45°) ≈ -0.707.
             Threshold -0.7 ≈ 45° tilt.
        Ref: CHRL uses roll > 60°; legged_gym uses ~45°; fall recovery papers
             use 60° with 0.1–0.3s temporal filter. We use 45° without filter
             for clean termination signal during initial training.
        """
        base_z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        return bool(base_z < 0.15 or gravity_body[2] > -0.7)

    def _update_foot_contacts(self):
        """Detect foot-ground contacts using cached geom id lookup (no string comparison)."""
        self._foot_contacts = np.zeros(4, dtype=np.float32)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 in self._foot_geom_id_to_idx:
                self._foot_contacts[self._foot_geom_id_to_idx[g1]] = 1.0
            if g2 in self._foot_geom_id_to_idx:
                self._foot_contacts[self._foot_geom_id_to_idx[g2]] = 1.0

    def _update_collision_info(self):
        """Detect body (non-foot) collisions with ground and foot stumbles.

        Body collision: any non-foot geom (trunk, thigh, calf) touching floor.
        Stumble: foot hitting a non-floor obstacle with a mostly-horizontal
        contact normal. Floor contacts are excluded because heightfield
        surfaces naturally produce shallow contact normals on rough terrain.
        """
        self._body_collision_count = 0
        self._stumble_count = 0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            is_floor_g1 = (g1 == self._floor_geom_id_int)
            is_floor_g2 = (g2 == self._floor_geom_id_int)
            is_floor = is_floor_g1 or is_floor_g2

            # Body-ground collision: non-foot robot geom touching floor
            if is_floor:
                other = g2 if is_floor_g1 else g1
                if other in self._body_geom_ids:
                    self._body_collision_count += 1

            # Stumble: foot hitting a NON-FLOOR object with horizontal normal
            # (e.g. hitting the side of a stair step). Floor contacts excluded.
            is_foot_g1 = g1 in self._foot_geom_ids
            is_foot_g2 = g2 in self._foot_geom_ids
            if (is_foot_g1 or is_foot_g2) and not is_floor:
                normal = contact.frame[:3]
                if abs(normal[2]) < 0.3:
                    self._stumble_count += 1

    def _apply_push(self):
        """Apply random horizontal force perturbation to the base body.

        Uses MuJoCo xfrc_applied (6D: force + torque in world frame) which
        is properly integrated by the physics engine. This replaces the
        previous velocity-impulse hack (qvel[:3] += force) which bypassed
        the integrator.

        Ref: CHRL uses up to 80N pushes. We default to 50N.
        """
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        force_xy = rng.uniform(-self.push_magnitude, self.push_magnitude, size=2)
        # xfrc_applied shape: (nbody, 6) — [fx, fy, fz, tx, ty, tz]
        self.data.xfrc_applied[self._base_body_id, 0] = float(force_xy[0])
        self.data.xfrc_applied[self._base_body_id, 1] = float(force_xy[1])
        self.data.xfrc_applied[self._base_body_id, 2] = 0.0  # no vertical force

    def _apply_domain_randomization(self):
        """Apply domain randomization to model parameters.

        Randomized parameters and ranges (consensus from RMA, DreamWaQ, CHRL):
          - Body masses: ±15% per body
          - Floor friction: 0.3–1.5
          - Foot friction: 0.8–2.0
          - Joint damping: ±20%
          - Joint armature: ±20%
          - PD gains (kp, kd): ±15%
          - Motor strength (max_torque): ±10%
        """
        rng = self.np_random if hasattr(self, 'np_random') else np.random

        # Mass randomization: ±15% per body
        for i in range(self.model.nbody):
            body_scale = rng.uniform(0.85, 1.15)
            self.model.body_mass[i] = self._base_masses[i] * body_scale

        # Friction randomization: floor and foot geoms separately
        floor_friction = rng.uniform(0.3, 1.5)
        foot_friction = rng.uniform(0.8, 2.0)
        for i in range(self.model.ngeom):
            name = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_GEOM, i)
            if name and "floor" in name:
                self.model.geom_friction[i][0] = floor_friction
            elif name and "foot_collision" in name:
                self.model.geom_friction[i][0] = foot_friction

        # Joint damping randomization: ±20%
        for i in range(self.model.nv):
            damp_scale = rng.uniform(0.8, 1.2)
            self.model.dof_damping[i] = self._base_damping[i] * damp_scale

        # Joint armature randomization: ±20%
        for i in range(self.model.nv):
            arm_scale = rng.uniform(0.8, 1.2)
            self.model.dof_armature[i] = self._base_armature[i] * arm_scale

        # PD gain randomization: ±15%
        self.kp = self._base_kp * rng.uniform(0.85, 1.15)
        self.kd = self._base_kd * rng.uniform(0.85, 1.15)

        # Motor strength randomization: ±10%
        self.max_torque = self._base_max_torque * rng.uniform(0.9, 1.1)

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "step": self.step_count,
            "base_pos": self.data.qpos[:3].tolist(),
            "base_height": float(self.data.qpos[2]),
            "command": self.command.tolist(),
            "mode": self.command_mode,
            "skill_mode": self.skill_mode,
            "terrain_type": self.terrain_type,
            "difficulty": self.difficulty,
            "foot_contacts": self._foot_contacts.tolist(),
        }
        if hasattr(self, "_last_reward_components"):
            info["reward_components"] = self._last_reward_components
        return info

    def set_command(self, vx: float, vy: float, wz: float, mode: str = "trot"):
        self.command = np.array([vx, vy, wz], dtype=np.float32)
        self.command_mode = mode

    def set_skill(self, skill: str):
        if skill in SKILL_MODES:
            self.skill_mode = skill

    def set_terrain(self, terrain_type: str, difficulty: float = 0.5):
        self.terrain_type = terrain_type
        self.difficulty = difficulty

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
        """Rotate a world-frame vector into body frame. MuJoCo quat = [w,x,y,z].

        Computes q_conj * v * q (inverse/passive rotation).
        """
        w, x, y, z = quat
        v = np.array(vec, dtype=np.float64)
        q_vec = np.array([x, y, z], dtype=np.float64)
        t = 2.0 * np.cross(q_vec, v)
        result = v - w * t + np.cross(q_vec, t)
        return result.astype(np.float32)
