"""
MIT Mini Cheetah Gymnasium Environment.

Observation (48 dims):
  [0:12]  joint positions (rad)
  [12:24] joint velocities (rad/s)
  [24:27] base linear velocity (m/s, body frame)
  [27:30] base angular velocity (rad/s, body frame)
  [30:33] projected gravity vector (body frame)
  [33:45] previous action (rad)
  [45:48] velocity command (vx, vy, wz)

Action (12 dims): delta joint position targets, clipped ±0.5 rad
"""
import os
import math
import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces

NUM_JOINTS = 12
OBS_DIM = 48
ACT_DIM = 12

# Default standing pose: [abduct, hip, knee] × 4 legs (FR, FL, RR, RL)
# With legs straight down (all-zero) the robot stands at ~0.37m
# Mild crouch with hip forward, knee back gives a more natural pose
DEFAULT_STANCE = np.array([
    0.0,  0.7, -1.4,  # FR
    0.0,  0.7, -1.4,  # FL
    0.0,  0.7, -1.4,  # RR
    0.0,  0.7, -1.4,  # RL
], dtype=np.float32)


class MiniCheetahEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str = "none",
        terrain_type: str = "flat",
        control_mode: str = "direct",
        randomize_domain: bool = True,
        episode_length: int = 1000,
        dt: float = 0.02,
        physics_dt: float = 0.002,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.terrain_type = terrain_type
        self.control_mode = control_mode
        self.randomize_domain = randomize_domain
        self.episode_length = episode_length
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
        self.command_mode = "stand"
        self.randomize_commands = True  # randomize command on each reset for training

        # PD gains
        self.kp = 80.0
        self.kd = 1.0
        self.max_torque = 17.0

        # Domain randomization
        self._base_mass = None
        self._base_friction = None

        self._init_simulator()

    def _find_model(self):
        # Use absolute path derived from this file's location (works in subprocesses)
        base = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base))
        primary = os.path.join(project_root, "assets", "mini_cheetah.xml")
        if os.path.exists(primary):
            return primary
        # Fallback: try relative to cwd
        if os.path.exists("assets/mini_cheetah.xml"):
            return os.path.abspath("assets/mini_cheetah.xml")
        raise FileNotFoundError(
            f"Cannot find mini_cheetah.xml. Looked in: {primary} and assets/mini_cheetah.xml")

    def _init_simulator(self):
        import mujoco
        import mujoco.viewer  # must be imported explicitly — not auto-imported with mujoco
        self._mj = mujoco
        self._mj_viewer = mujoco.viewer
        model_path = self._find_model()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Cache body mass for DR reset
        self._base_masses = self.model.body_mass.copy()

        self.viewer = None
        self.renderer = None

        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    # ── Gymnasium API ──────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)

        # Reset masses before DR
        self.model.body_mass[:] = self._base_masses

        # Set initial pose
        self.data.qpos[2] = 0.32  # height
        self.data.qpos[3] = 1.0   # quat w
        self.data.qpos[4:7] = 0.0
        self.data.qpos[7:7 + NUM_JOINTS] = DEFAULT_STANCE
        mujoco.mj_forward(self.model, self.data)

        if self.randomize_domain:
            self._apply_domain_randomization()

        # Randomize velocity command for training diversity
        if self.randomize_commands:
            rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
            vx = float(rng.uniform(-0.5, 2.0))
            vy = float(rng.uniform(-0.5, 0.5))
            wz = float(rng.uniform(-0.5, 0.5))
            self.command = np.array([vx, vy, wz], dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -0.5, 0.5).astype(np.float32)
        target_q = DEFAULT_STANCE + action

        mujoco = self._mj
        q = self.data.qpos[7:7 + NUM_JOINTS]
        qd = self.data.qvel[6:6 + NUM_JOINTS]
        tau = self.kp * (target_q - q) - self.kd * qd
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        self.data.ctrl[:NUM_JOINTS] = tau

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_done()
        truncated = self.step_count >= self.episode_length

        self.prev_action = action.copy()
        self.step_count += 1

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_info()

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

    # ── Observations ────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        d = self.data
        qpos = d.qpos[7:7 + NUM_JOINTS].astype(np.float32)
        qvel = d.qvel[6:6 + NUM_JOINTS].astype(np.float32)
        base_linvel = d.qvel[:3].astype(np.float32)
        base_angvel = d.qvel[3:6].astype(np.float32)
        quat = d.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))

        obs = np.concatenate([
            qpos, qvel, base_linvel, base_angvel,
            gravity_body, self.prev_action, self.command
        ]).astype(np.float32)

        if self.randomize_domain:
            obs += self.np_random.standard_normal(OBS_DIM).astype(np.float32) * 0.02

        return obs

    # ── Reward ──────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        base_linvel = self.data.qvel[:3]
        base_angvel = self.data.qvel[3:6]
        tau = self.data.ctrl[:NUM_JOINTS]

        vx_cmd, vy_cmd, wz_cmd = self.command

        # ── Tracking rewards (only positive signal — ETH/legged_gym standard) ──
        r_linvel = (
            math.exp(-((base_linvel[0] - vx_cmd) ** 2) / 0.25) +
            math.exp(-((base_linvel[1] - vy_cmd) ** 2) / 0.25)
        )
        r_yaw = 0.5 * math.exp(-((base_angvel[2] - wz_cmd) ** 2) / 0.25)

        # ── Orientation & stability penalties ──
        quat = self.data.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))

        # Lateral gravity penalty: penalize tilt (g_x² + g_y²) — bounded [0, 2]
        r_orientation = -0.5 * float(np.sum(gravity_body[:2] ** 2))

        # Z-velocity penalty: prevents vertical bouncing/hopping (RMA, Isaac Gym)
        r_lin_vel_z = -0.5 * float(base_linvel[2] ** 2)

        # XY angular velocity penalty: prevents unwanted rolling/pitching
        r_ang_vel_xy = -0.05 * float(np.sum(base_angvel[:2] ** 2))

        # Height penalty: squared deviation from standing height (Isaac Gym style)
        # Not a Gaussian reward — only tracking should provide positive signal
        base_z = float(self.data.qpos[2])
        r_height = -1.0 * (base_z - 0.32) ** 2

        # ── Energy efficiency & smoothness ──
        r_torque = -0.0002 * float(np.sum(tau ** 2))
        r_smooth = -0.01 * float(np.sum((action - self.prev_action) ** 2))

        # No survival bonus — tracking reward is the only incentive to stay alive.
        # Papers (legged_gym, RMA, Isaac Gym) omit survival or use ≤0.1.

        total = (r_linvel + r_yaw + r_orientation + r_lin_vel_z
                 + r_ang_vel_xy + r_height + r_torque + r_smooth)

        # Store components for logging callbacks
        self._last_reward_components = {
            "r_linvel": r_linvel,
            "r_yaw": r_yaw,
            "r_orientation": r_orientation,
            "r_lin_vel_z": r_lin_vel_z,
            "r_ang_vel_xy": r_ang_vel_xy,
            "r_height": r_height,
            "r_torque": r_torque,
            "r_smooth": r_smooth,
            "r_total": total,
        }

        return total

    # ── Terminal ────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        base_z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        # Fallen: too low or tilted > ~60 degrees
        return bool(base_z < 0.12 or gravity_body[2] > -0.5)

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "step": self.step_count,
            "base_pos": self.data.qpos[:3].tolist(),
            "base_height": float(self.data.qpos[2]),
            "command": self.command.tolist(),
            "mode": self.command_mode,
        }
        if hasattr(self, "_last_reward_components"):
            info["reward_components"] = self._last_reward_components
        return info

    # ── Command interface ───────────────────────────────────────────

    def set_command(self, vx: float, vy: float, wz: float, mode: str = "trot"):
        self.command = np.array([vx, vy, wz], dtype=np.float32)
        self.command_mode = mode

    def set_exploration_heading(self, heading_rad: float, speed: float = 1.5):
        vx = speed * math.cos(heading_rad)
        vy = speed * math.sin(heading_rad)
        self.set_command(vx, vy, 0.0, "explore")

    # ── Domain Randomization ────────────────────────────────────────

    def _apply_domain_randomization(self):
        rng = self.np_random if hasattr(self, 'np_random') else np.random
        friction_scale = rng.uniform(0.5, 1.5)

        # Per-body mass randomization for more realistic variation
        for i in range(self.model.nbody):
            body_scale = rng.uniform(0.85, 1.15)
            self.model.body_mass[i] = self._base_masses[i] * body_scale

        # Floor friction
        for i in range(self.model.ngeom):
            name = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_GEOM, i)
            if name and "floor" in name:
                self.model.geom_friction[i][0] = 1.0 * friction_scale

    # ── Quaternion math ─────────────────────────────────────────────

    @staticmethod
    def _quat_rotate_inv(quat, vec):
        """Rotate a world-frame vector into body frame. MuJoCo quat = [w,x,y,z].

        Computes q_conj * v * q (inverse/passive rotation).
        Uses Rodrigues formula with negated q_vec for conjugate.
        """
        w, x, y, z = quat
        v = np.array(vec, dtype=np.float64)
        q_vec = np.array([x, y, z], dtype=np.float64)
        t = 2.0 * np.cross(q_vec, v)
        result = v - w * t + np.cross(q_vec, t)
        return result.astype(np.float32)
