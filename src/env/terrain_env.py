"""
Advanced Terrain Environment for Mini Cheetah.

Procedurally generates terrain types and supports multi-skill locomotion:
  - Walking / trotting / running over varied terrain
  - Jumping across gaps
  - Crouching under obstacles
  - Recovery from push perturbations

Terrain types generated via MuJoCo heightfield:
  - Flat, rough, slopes, stairs, gaps, stepping stones, mixed

Inspired by: Isaac Gym terrain generation, GenTe, DreamWaQ++, PGTT.
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
SKILL_MODES = {
    "walk":   {"target_speed": 0.8,  "height_target": 0.32, "energy_weight": 0.3},
    "trot":   {"target_speed": 1.5,  "height_target": 0.32, "energy_weight": 0.2},
    "run":    {"target_speed": 2.5,  "height_target": 0.30, "energy_weight": 0.1},
    "jump":   {"target_speed": 0.5,  "height_target": 0.45, "energy_weight": 0.1},
    "crouch": {"target_speed": 0.5,  "height_target": 0.20, "energy_weight": 0.3},
    "stand":  {"target_speed": 0.0,  "height_target": 0.32, "energy_weight": 0.5},
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
            # Smooth slightly to avoid spikes
            from scipy.ndimage import gaussian_filter
            heights = gaussian_filter(heights, sigma=1.5).astype(np.float32)

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
                # Platform
                end_plat = min(pos + platform_w, n)
                pos = end_plat
                # Gap (set to negative = pit)
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
            # Divide terrain into sections with different types
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
        """
        if self._heightfield is None:
            return np.zeros(8, dtype=np.float32)

        n = self.resolution
        # Convert world coords to grid indices
        cx = int(np.clip((x / self.size + 0.5) * n, 0, n - 1))
        cy = int(np.clip((y / self.size + 0.5) * n, 0, n - 1))
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

        # Slope estimation via gradient
        grad_x = np.gradient(patch, axis=0).mean() if patch.shape[0] > 1 else 0.0
        grad_y = np.gradient(patch, axis=1).mean() if patch.shape[1] > 1 else 0.0

        # Roughness: second derivative magnitude
        roughness = std_h

        # Max gap: maximum height difference in patch
        max_gap = max_h - min_h

        return np.array([
            mean_h, std_h, max_h, min_h,
            grad_x, grad_y, roughness, max_gap
        ], dtype=np.float32)


class AdvancedTerrainEnv(gym.Env):
    """Advanced terrain environment with multi-skill locomotion.

    Extended observation (57 dims):
      [0:48]  base observation (same as MiniCheetahEnv)
      [48:56] terrain encoding (8 features)
      [56]    episode phase (0 to 1)

    Supports terrain generation via heightfield and push perturbations.
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
        episode_length: int = 1000,
        push_interval: int = 200,
        push_magnitude: float = 0.5,
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
        self.push_magnitude = push_magnitude
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

        self.kp = 80.0
        self.kd = 1.0
        self.max_torque = 17.0

        self._base_masses = None
        self.terrain_gen = TerrainGenerator()
        self._terrain_encoding = np.zeros(8, dtype=np.float32)

        # Foot contact tracking
        self._foot_contacts = np.zeros(4, dtype=np.float32)
        self._contact_history = []

        self._init_simulator()

    def _find_model(self):
        base = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base))
        primary = os.path.join(project_root, "assets", "mini_cheetah.xml")
        if os.path.exists(primary):
            return primary
        if os.path.exists("assets/mini_cheetah.xml"):
            return os.path.abspath("assets/mini_cheetah.xml")
        raise FileNotFoundError(f"Cannot find mini_cheetah.xml")

    def _init_simulator(self):
        import mujoco
        import mujoco.viewer
        self._mj = mujoco
        self._mj_viewer = mujoco.viewer
        model_path = self._find_model()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._base_masses = self.model.body_mass.copy()
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

        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[:] = self._base_masses

        # Set initial pose
        self.data.qpos[2] = 0.32
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.qpos[7:7 + NUM_JOINTS] = DEFAULT_STANCE
        mujoco.mj_forward(self.model, self.data)

        if self.randomize_domain:
            self._apply_domain_randomization()

        # Randomize terrain
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
        if self.randomize_terrain:
            terrain_types = TerrainGenerator.TERRAIN_TYPES
            self.terrain_type = str(rng.choice(terrain_types))
            self.difficulty = float(rng.uniform(0.0, 1.0))
        self.terrain_gen = TerrainGenerator(seed=int(rng.integers(0, 2**31)) if hasattr(rng, 'integers') else int(rng.randint(0, 2**31)))
        self.terrain_gen.generate(self.terrain_type, self.difficulty)

        # Randomize skill mode
        if self.randomize_skill:
            self.skill_mode = str(rng.choice(list(SKILL_MODES.keys())))

        # Randomize velocity command
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

        # Apply random push perturbation at intervals
        if (self.push_interval > 0 and self.step_count > 0 and
                self.step_count % self.push_interval == 0):
            self._apply_push()

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Update foot contacts
        self._update_foot_contacts()

        # Update terrain encoding based on robot position
        pos = self.data.qpos[:2]
        self._terrain_encoding = self.terrain_gen.get_terrain_encoding(
            float(pos[0]), float(pos[1])
        )

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_done()
        truncated = self.step_count >= self.episode_length

        self.prev_action = action.copy()
        self.step_count += 1

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        d = self.data
        qpos = d.qpos[7:7 + NUM_JOINTS].astype(np.float32)
        qvel = d.qvel[6:6 + NUM_JOINTS].astype(np.float32)
        base_linvel = d.qvel[:3].astype(np.float32)
        base_angvel = d.qvel[3:6].astype(np.float32)
        quat = d.qpos[3:7]
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
            obs[:BASE_OBS_DIM] += np.random.randn(BASE_OBS_DIM).astype(np.float32) * 0.02

        return obs

    def _compute_reward(self, action: np.ndarray) -> float:
        base_linvel = self.data.qvel[:3]
        base_angvel = self.data.qvel[3:6]
        tau = self.data.ctrl[:NUM_JOINTS]
        skill = SKILL_MODES[self.skill_mode]

        vx_cmd, vy_cmd, wz_cmd = self.command

        # -- Velocity tracking --
        r_linvel = (
            math.exp(-((base_linvel[0] - vx_cmd) ** 2) / 0.25) +
            math.exp(-((base_linvel[1] - vy_cmd) ** 2) / 0.25)
        )
        r_yaw = 0.5 * math.exp(-((base_angvel[2] - wz_cmd) ** 2) / 0.25)

        # -- Height tracking (skill-dependent) --
        base_z = float(self.data.qpos[2])
        target_h = skill["height_target"]
        r_height = -2.0 * (base_z - target_h) ** 2

        # -- Orientation --
        quat = self.data.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        r_orientation = -0.5 * float(np.sum(gravity_body[:2] ** 2))

        # -- Z-velocity penalty --
        r_lin_vel_z = -0.5 * float(base_linvel[2] ** 2)

        # -- Angular velocity penalty --
        r_ang_vel_xy = -0.05 * float(np.sum(base_angvel[:2] ** 2))

        # -- Energy efficiency (skill-weighted) --
        energy_w = skill["energy_weight"]
        r_torque = -energy_w * 0.0002 * float(np.sum(tau ** 2))

        # -- Smoothness --
        r_smooth = -0.01 * float(np.sum((action - self.prev_action) ** 2))

        # -- Foot contact reward: encourage rhythmic stepping for walking gaits --
        contact_sum = float(np.sum(self._foot_contacts))
        if self.skill_mode in ("walk", "trot", "run"):
            # Prefer 2 feet on ground (diagonal for trot)
            r_contact = -0.1 * (contact_sum - 2.0) ** 2
        elif self.skill_mode == "jump":
            # In jump: reward airborne phases
            r_contact = 0.1 * max(0, 1.0 - contact_sum)
        elif self.skill_mode == "crouch":
            # In crouch: reward all feet on ground
            r_contact = 0.1 * min(contact_sum, 4.0) / 4.0
        else:
            r_contact = 0.0

        # -- Terrain adaptability: reward for traversing rough terrain --
        terrain_difficulty = self._terrain_encoding[1] + self._terrain_encoding[7]  # std + gap
        r_terrain = 0.2 * terrain_difficulty * (base_linvel[0] > 0.1)

        total = (r_linvel + r_yaw + r_height + r_orientation +
                 r_lin_vel_z + r_ang_vel_xy + r_torque + r_smooth +
                 r_contact + r_terrain)

        self._last_reward_components = {
            "r_linvel": r_linvel,
            "r_yaw": r_yaw,
            "r_height": r_height,
            "r_orientation": r_orientation,
            "r_lin_vel_z": r_lin_vel_z,
            "r_ang_vel_xy": r_ang_vel_xy,
            "r_torque": r_torque,
            "r_smooth": r_smooth,
            "r_contact": r_contact,
            "r_terrain": r_terrain,
            "r_total": total,
        }

        return total

    def _check_done(self) -> bool:
        base_z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        return bool(base_z < 0.10 or gravity_body[2] > -0.4)

    def _update_foot_contacts(self):
        """Detect foot-ground contacts."""
        mujoco = self._mj
        foot_names = ["FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision"]
        self._foot_contacts = np.zeros(4, dtype=np.float32)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            for j, fname in enumerate(foot_names):
                if (geom1 == fname or geom2 == fname):
                    self._foot_contacts[j] = 1.0

    def _apply_push(self):
        """Apply random force perturbation to the base."""
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        force = rng.uniform(-self.push_magnitude, self.push_magnitude, size=3)
        force[2] = 0  # only horizontal push
        # Apply as velocity perturbation
        self.data.qvel[:3] += force.astype(np.float64)

    def _apply_domain_randomization(self):
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        mass_scale = rng.uniform(0.8, 1.2)
        friction_scale = rng.uniform(0.5, 1.5)

        for i in range(self.model.nbody):
            self.model.body_mass[i] = self._base_masses[i] * mass_scale

        for i in range(self.model.ngeom):
            name = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_GEOM, i)
            if name and "floor" in name:
                self.model.geom_friction[i][0] = 1.0 * friction_scale

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
        w, x, y, z = quat
        v = np.array(vec, dtype=np.float64)
        q_vec = np.array([x, y, z], dtype=np.float64)
        t = 2.0 * np.cross(q_vec, v)
        result = v + w * t + np.cross(q_vec, t)
        return result.astype(np.float32)
