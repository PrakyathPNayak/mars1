"""
Unitree Go1 Quadruped Gymnasium Environment.

Robot: Unitree Go1 (from mujoco_menagerie, BSD-3-Clause)
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
# Unitree Go1 home keyframe from mujoco_menagerie
DEFAULT_STANCE = np.array([
    0.0,  0.9, -1.8,  # FR
    0.0,  0.9, -1.8,  # FL
    0.0,  0.9, -1.8,  # RR
    0.0,  0.9, -1.8,  # RL
], dtype=np.float32)

# Reward scale dictionary — signed weights (ETH/legged_gym convention).
# Penalty scales reduced from legged_gym defaults to compensate for
# 25 CPU-parallel envs (vs 4096 GPU envs).  Tracking rewards use
# exp(-error²/σ) kernel so they are always in [0,1].
REWARD_SCALES = {
    "r_linvel":      1.5,      # velocity tracking (exp kernel), max 3.0
    "r_yaw":         0.75,     # yaw-rate tracking, max 0.75
    "r_alive":       0.5,      # constant alive bonus
    "r_orientation": -0.2,     # gravity-tilt penalty (was -0.5)
    "r_lin_vel_z":   -1.0,     # vertical bounce penalty (was -2.0)
    "r_ang_vel_xy":  -0.005,   # roll/pitch rate penalty (was -0.05, 10× reduction)
    "r_height":      -0.5,     # height deviation penalty (was -1.0)
    "r_torque":      -5e-6,    # torque squared penalty (was -2e-5, 4× reduction)
    "r_smooth":      -0.01,    # action rate penalty
    "r_joint_acc":   -2.5e-8,  # joint accel penalty (was -2.5e-7, 10× reduction)
    "r_joint_limit": -5.0,     # joint limit proximity penalty (was -10.0)
}

# Clip total reward to >= 0 (legged_gym only_positive_rewards convention).
# Prevents the early-termination death spiral where the policy learns
# that dying quickly minimises accumulated penalties.
ONLY_POSITIVE_REWARDS = True


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
        self.prev_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.command_mode = "stand"
        self.randomize_commands = True  # randomize command on each reset for training

        # PD gains — Go1 XML already has damping=2 on joints, so kd=0 here
        # to avoid double-damping. The total damping matches the menagerie
        # position actuator behavior: passive damping only.
        self.kp = 100.0
        self.kd = 0.0
        self.max_torque = 33.5  # Go1: 23.7 Nm hip/thigh, 35.55 Nm knee

        # Domain randomization
        self._base_mass = None
        self._base_friction = None

        self._init_simulator()

    def _find_model(self):
        # Use absolute path derived from this file's location (works in subprocesses)
        base = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base))
        primary = os.path.join(project_root, "assets", "go1.xml")
        if os.path.exists(primary):
            return primary
        # Fallback: try relative to cwd
        if os.path.exists("assets/go1.xml"):
            return os.path.abspath("assets/go1.xml")
        raise FileNotFoundError(
            f"Cannot find go1.xml. Looked in: {primary} and assets/go1.xml")

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
        self.prev_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)

        mujoco = self._mj
        mujoco.mj_resetData(self.model, self.data)

        # Reset masses before DR
        self.model.body_mass[:] = self._base_masses

        # Set initial pose — Go1 standing height ~0.27m
        self.data.qpos[2] = 0.27  # height
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

        # Track state for next-step penalties
        self.prev_joint_vel = self.data.qvel[6:6 + NUM_JOINTS].copy().astype(np.float32)
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

        # Body-frame velocities (rotate world-frame qvel into body frame)
        quat = d.qpos[3:7]
        base_linvel = self._quat_rotate_inv(quat, d.qvel[:3]).astype(np.float32)
        base_angvel = self._quat_rotate_inv(quat, d.qvel[3:6]).astype(np.float32)
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
        quat = self.data.qpos[3:7]
        # Body-frame velocities — consistent with obs
        base_linvel = self._quat_rotate_inv(quat, self.data.qvel[:3])
        base_angvel = self._quat_rotate_inv(quat, self.data.qvel[3:6])
        tau = self.data.ctrl[:NUM_JOINTS]
        joint_vel = self.data.qvel[6:6 + NUM_JOINTS]

        vx_cmd, vy_cmd, wz_cmd = self.command

        # ── Tracking rewards (exp kernel — only positive signal) ──
        r_linvel = (
            math.exp(-((base_linvel[0] - vx_cmd) ** 2) / 0.25) +
            math.exp(-((base_linvel[1] - vy_cmd) ** 2) / 0.25)
        )
        r_yaw = math.exp(-((base_angvel[2] - wz_cmd) ** 2) / 0.25)

        # ── Orientation & stability penalties ──
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        r_orientation = float(np.sum(gravity_body[:2] ** 2))
        r_lin_vel_z = float(base_linvel[2] ** 2)
        r_ang_vel_xy = float(np.sum(base_angvel[:2] ** 2))

        # Height penalty: squared deviation from Go1 standing height
        base_z = float(self.data.qpos[2])
        r_height = (base_z - 0.27) ** 2

        # ── Energy efficiency & smoothness ──
        r_torque = float(np.sum(tau ** 2))
        r_smooth = float(np.sum((action - self.prev_action) ** 2))

        # Joint acceleration penalty: penalize high-frequency jitter
        joint_acc = (joint_vel - self.prev_joint_vel) / self.dt
        r_joint_acc = float(np.sum(joint_acc ** 2))

        # Joint limit proximity penalty: soft penalty near limits
        q = self.data.qpos[7:7 + NUM_JOINTS]
        jnt_range = self.model.jnt_range[1:NUM_JOINTS + 1]  # skip freejoint
        margin = 0.1  # rad
        below = np.clip(jnt_range[:, 0] + margin - q, 0, None)
        above = np.clip(q - (jnt_range[:, 1] - margin), 0, None)
        r_joint_limit = float(np.sum(below ** 2 + above ** 2))

        # ── Alive bonus (constant positive signal for surviving) ──
        r_alive = 1.0  # raw value; scaled by REWARD_SCALES["r_alive"]

        # ── Assemble with REWARD_SCALES ──
        components = {
            "r_linvel": r_linvel,
            "r_yaw": r_yaw,
            "r_alive": r_alive,
            "r_orientation": r_orientation,
            "r_lin_vel_z": r_lin_vel_z,
            "r_ang_vel_xy": r_ang_vel_xy,
            "r_height": r_height,
            "r_torque": r_torque,
            "r_smooth": r_smooth,
            "r_joint_acc": r_joint_acc,
            "r_joint_limit": r_joint_limit,
        }

        total = sum(REWARD_SCALES[k] * v for k, v in components.items())

        if ONLY_POSITIVE_REWARDS:
            total = max(total, 0.0)

        # Store components (with scales applied) for logging callbacks
        self._last_reward_components = {
            k: REWARD_SCALES[k] * v for k, v in components.items()
        }
        self._last_reward_components["r_total"] = total

        return total

    # ── Terminal ────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        base_z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        # Fallen: too low or tilted > ~45 degrees (cos 45° ≈ 0.707)
        return bool(base_z < 0.13 or gravity_body[2] > -0.7)

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
