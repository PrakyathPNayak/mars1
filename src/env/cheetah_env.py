"""
Unitree Go1 Quadruped Gymnasium Environment.

Robot: Unitree Go1 (from mujoco_menagerie, BSD-3-Clause)
Observation (49 dims):
  [0:12]  joint positions (rad)
  [12:24] joint velocities (rad/s)
  [24:27] base linear velocity (m/s, body frame)
  [27:30] base angular velocity (rad/s, body frame)
  [30:33] projected gravity vector (body frame)
  [33:45] previous action (rad)
  [45:49] command (vx, vy, wz, target_height)

Action (12 dims): delta joint position targets, clipped ±0.5 rad

Reward design (based on ETH Zurich legged_gym):
  + exp tracking for xy velocity (combined kernel, σ=0.15)
  + exp tracking for yaw rate
  + feet air-time gait reward (incentivises stepping, gated by cmd)
  - squared velocity command error (penalises ignoring commands)
  - orientation/stability penalties
  - energy smoothness penalties
"""
import os
import math
import collections
import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces

NUM_JOINTS = 12
OBS_DIM = 49  # 48 + 1 target_height appended to command
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
#
# CALIBRATION (from 20M-step training log analysis):
#   Raw penalty magnitudes at typical behavior:
#     body_acc²≈320, joint_acc²≈8M, smooth²≈4.4, ang_vel_xy²≈70,
#     cmd_vel_error≈1.5, lin_vel_z²≈0.19, torque²≈~10000
#   Scales are set so each penalty contributes at most ~0.2-0.5/step.
#   Positive rewards (tracking + alive) should sum to ~4.0-5.0/step.
#   Ref: legged_gym (Rudin et al. 2022), Walk These Ways (Margolis et al. 2023)
#
# Height targets for different command modes
HEIGHT_TARGETS = {
    "stand": 0.27, "walk": 0.27, "trot": 0.27, "run": 0.27, "explore": 0.27,
    "crouch": 0.18,
    "jump": 0.35,
}

# Anti-crouch: rolling window to detect sustained unwanted crouching
CROUCH_DETECT_WINDOW = 10       # steps (~0.2s at 50 Hz)
CROUCH_HEIGHT_THRESHOLD = 0.22  # below this counts as "crouching"

# Go1 approximate total mass (kg) — for cost-of-transport scaling
ROBOT_MASS = 12.74

REWARD_SCALES = {
    # ── Positive rewards (should sum to ~4-5/step for good behavior) ──
    "r_alive":           2.0,      # survival bonus (legged_gym standard, prevents death spiral)
    "r_linvel":          2.0,      # exp tracking for xy vel (combined kernel, max 1.0)
    "r_yaw":             1.0,      # exp tracking for yaw rate (max 1.0)
    "r_feet_air_time":   1.5,      # gait reward: incentivises leg swing (legged_gym)
    "r_gait_phase":      0.5,      # trot diagonal contact symmetry reward
    "r_foot_clearance":  0.3,      # swing-foot height reward
    "r_posture":         0.5,      # hip-knee angle target per mode (exp kernel)
    "r_jump_phase":      1.0,      # jump FSM phase-specific rewards
    # ── Penalties (each should contribute at most ~0.2-0.5/step) ──
    "r_cmd_vel_error":  -0.5,      # squared velocity error (was -2.0, raw≈1.5 → -0.75)
    "r_orientation":    -0.5,      # gravity-tilt penalty (raw≈0.17 → -0.085)
    "r_lin_vel_z":      -2.0,      # vertical bounce penalty (raw≈0.19 → -0.38)
    "r_ang_vel_xy":     -0.005,    # roll/pitch rate penalty (raw≈70 → -0.35)
    "r_height":         -1.0,      # height deviation from mode-dependent target
    "r_torque":         -5e-6,     # torque squared penalty (raw≈10k → -0.05)
    "r_smooth":         -0.05,     # action rate penalty (raw≈4.4 → -0.22)
    "r_joint_acc":      -5e-8,     # joint accel penalty (raw≈8M → -0.40)
    "r_joint_limit":    -5.0,      # joint limit proximity penalty (raw≈0.002 → -0.01)
    "r_crouch_penalty": -1.0,      # anti-crouch: penalize sustained low height
    "r_power":          -1e-5,     # mechanical power |tau·qvel| (raw≈4000 → -0.04)
    "r_stand_still":    -0.5,      # penalize joint deviation at zero command
    "r_foot_strike_vel":-0.005,    # foot velocity at contact (raw≈1.8 → -0.009)
    "r_body_acc":       -0.0005,   # body linear accel penalty (raw≈320 → -0.16)
    "r_action_jerk":    -0.005,    # second-order action smoothness (raw≈12 → -0.06)
}

# Tracking σ for exp kernel: legged_gym default is 0.25.
# At σ=0.25, standing with cmd=0.5 → exp≈0.37 (decent gradient for early training).
# Was 0.15 which made early exploration get near-zero tracking reward.
TRACKING_SIGMA = 0.25

# Feet air-time threshold (seconds).  First-contact reward = (air_time - threshold).
# Positive for steps longer than threshold (natural gait), negative for chattering.
FEET_AIR_TIME_THRESHOLD = 0.15

# Clip total reward to >= 0 (legged_gym only_positive_rewards convention).
# Prevents the early-termination death spiral where the policy learns
# that dying quickly minimises accumulated penalties.
ONLY_POSITIVE_REWARDS = True

# ── Jump finite state machine ──────────────────────────────────────
JUMP_PHASE_IDLE = 0
JUMP_PHASE_CROUCH = 1     # lower body before launch
JUMP_PHASE_LAUNCH = 2     # push upward
JUMP_PHASE_AIRBORNE = 3   # in the air
JUMP_PHASE_LANDING = 4    # stabilize after touchdown
JUMP_CROUCH_STEPS = 10    # 0.2s at 50 Hz
JUMP_LAUNCH_STEPS = 8     # 0.16s
JUMP_LANDING_STEPS = 15   # 0.3s
JUMP_AIRBORNE_MAX = 50    # 1s max airborne before forced landing

# ── Command re-randomization during training ──────────────────────
# Re-randomize velocity commands every N steps so the policy sees diverse
# commands within a single episode (ref: Walk These Ways, Margolis 2023).
COMMAND_RESAMPLE_INTERVAL = 200  # 4 seconds at 50 Hz

# ── Joint-angle posture targets (hip, knee) per mode ──────────────
# Based on ETH/RMA legged_gym conventions for Unitree Go1.
# Hip joint = rotation joint (indices 1,4,7,10), Knee joint (indices 2,5,8,11).
HIP_JOINT_INDICES = np.array([1, 4, 7, 10], dtype=np.int32)
KNEE_JOINT_INDICES = np.array([2, 5, 8, 11], dtype=np.int32)
POSTURE_TARGETS = {
    "stand":   {"hip": 0.9,  "knee": -1.8},
    "walk":    {"hip": 0.9,  "knee": -1.8},
    "trot":    {"hip": 0.9,  "knee": -1.8},
    "run":     {"hip": 0.8,  "knee": -1.6},
    "explore": {"hip": 0.9,  "knee": -1.8},
    "crouch":  {"hip": 1.5,  "knee": -2.5},
    "jump":    {"hip": 0.9,  "knee": -1.8},
}


class MiniCheetahEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str = "none",
        terrain_type: str = "flat",
        control_mode: str = "direct",
        randomize_domain: bool = True,
        episode_length: int = 2000,
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

        # Feet-air-time tracking for gait reward (legged_gym formulation)
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contacts = np.zeros(4, dtype=bool)

        # Height history for anti-crouch time-series detection
        self._height_history = collections.deque(maxlen=CROUCH_DETECT_WINDOW)
        self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)

        # Jump FSM state
        self._jump_phase = JUMP_PHASE_IDLE
        self._jump_step_counter = 0
        self._jump_max_height = 0.0

        # Tracking for smoothness rewards
        self.prev_prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_base_linvel = np.zeros(3, dtype=np.float32)
        self._prev_foot_heights = np.zeros(4, dtype=np.float32)

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

        # Cache foot geom IDs for contact detection (feet_air_time reward)
        _FOOT_GEOM_NAMES = ["FR", "FL", "RR", "RL"]
        self._foot_geom_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in _FOOT_GEOM_NAMES
        ], dtype=np.int32)

        # Cache foot site IDs for clearance reward
        _FOOT_SITE_NAMES = ["FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"]
        self._foot_site_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in _FOOT_SITE_NAMES
        ], dtype=np.int32)

    # ── Gymnasium API ──────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.prev_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._feet_air_time = np.zeros(4, dtype=np.float32)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._height_history.clear()
        self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)
        self._jump_phase = JUMP_PHASE_IDLE
        self._jump_step_counter = 0
        self._jump_max_height = 0.0
        self.prev_prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_base_linvel = np.zeros(3, dtype=np.float32)
        self._prev_foot_heights = np.zeros(4, dtype=np.float32)

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
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()
        self.step_count += 1

        # ── Mid-episode command re-randomization (Walk These Ways convention) ──
        if (self.randomize_commands
                and self.step_count % COMMAND_RESAMPLE_INTERVAL == 0):
            rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
            vx = float(rng.uniform(-0.5, 2.0))
            vy = float(rng.uniform(-0.5, 0.5))
            wz = float(rng.uniform(-0.5, 0.5))
            self.command = np.array([vx, vy, wz], dtype=np.float32)

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
            gravity_body, self.prev_action,
            np.append(self.command, self.target_height),
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

        # ── Tracking rewards (exp kernel, combined error — legged_gym style) ──
        lin_vel_error = (base_linvel[0] - vx_cmd) ** 2 + (base_linvel[1] - vy_cmd) ** 2
        r_linvel = math.exp(-lin_vel_error / TRACKING_SIGMA)

        ang_vel_error = (base_angvel[2] - wz_cmd) ** 2
        r_yaw = math.exp(-ang_vel_error / TRACKING_SIGMA)

        # ── Command velocity error penalty (squared) ──
        r_cmd_vel_error = float(lin_vel_error)

        # ── Feet air-time reward (legged_gym formulation) ──
        # Vectorized contact detection (replaces O(ncon×4) Python loop)
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
        r_feet_air_time = float(np.sum(
            (self._feet_air_time - FEET_AIR_TIME_THRESHOLD) * first_contact
        ))
        cmd_speed = math.sqrt(vx_cmd ** 2 + vy_cmd ** 2)
        if cmd_speed < 0.1:
            r_feet_air_time = 0.0
        self._feet_air_time[contact_filt] = 0.0

        # ── Gait phase reward: trot diagonal symmetry (legged_gym inspired) ──
        # Trot: diagonal pairs (FR+RL) and (FL+RR) should alternate.
        # Reward = |diag1_contact - diag2_contact| when moving.
        diag1 = float(contact_filt[0] + contact_filt[3])  # FR + RL
        diag2 = float(contact_filt[1] + contact_filt[2])  # FL + RR
        r_gait_phase = abs(diag1 - diag2) / 2.0  # max 1.0
        if cmd_speed < 0.1:
            r_gait_phase = 0.0  # no gait reward when standing

        # ── Foot clearance reward: swing feet should lift above ground ──
        foot_heights = self.data.site_xpos[self._foot_site_ids, 2]
        swing_mask = ~contact_filt  # feet NOT in contact = swing phase
        clearance_target = 0.05  # 5 cm target clearance
        r_foot_clearance = 0.0
        if np.any(swing_mask) and cmd_speed > 0.1:
            swing_heights = foot_heights[swing_mask]
            r_foot_clearance = float(np.mean(np.clip(swing_heights, 0, clearance_target) / clearance_target))

        # ── Orientation & stability penalties ──
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        r_orientation = float(np.sum(gravity_body[:2] ** 2))
        r_lin_vel_z = float(base_linvel[2] ** 2)
        r_ang_vel_xy = float(np.sum(base_angvel[:2] ** 2))

        # ── Height penalty: squared deviation from mode-dependent target ──
        base_z = float(self.data.qpos[2])
        r_height = (base_z - self.target_height) ** 2

        # ── Anti-crouch penalty (time-series check) ──
        self._height_history.append(base_z)
        r_crouch_penalty = 0.0
        if len(self._height_history) == CROUCH_DETECT_WINDOW:
            mean_height = sum(self._height_history) / CROUCH_DETECT_WINDOW
            if mean_height < CROUCH_HEIGHT_THRESHOLD and self.command_mode != "crouch":
                r_crouch_penalty = (CROUCH_HEIGHT_THRESHOLD - mean_height) ** 2

        # ── Energy efficiency & smoothness ──
        r_torque = float(np.sum(tau ** 2))
        r_smooth = float(np.sum((action - self.prev_action) ** 2))

        # Mechanical power: |tau · joint_vel| — true energy cost
        r_power = float(np.sum(np.abs(tau * joint_vel)))

        # Joint acceleration penalty
        joint_acc = (joint_vel - self.prev_joint_vel) / self.dt
        r_joint_acc = float(np.sum(joint_acc ** 2))

        # Joint limit proximity penalty
        q = self.data.qpos[7:7 + NUM_JOINTS]
        jnt_range = self.model.jnt_range[1:NUM_JOINTS + 1]
        margin = 0.1
        below = np.clip(jnt_range[:, 0] + margin - q, 0, None)
        above = np.clip(q - (jnt_range[:, 1] - margin), 0, None)
        r_joint_limit = float(np.sum(below ** 2 + above ** 2))

        # ── Stand-still penalty: penalize joint motion at zero command (legged_gym) ──
        r_stand_still = 0.0
        if cmd_speed < 0.1 and abs(wz_cmd) < 0.1:
            r_stand_still = float(np.sum(np.abs(
                self.data.qpos[7:7 + NUM_JOINTS] - DEFAULT_STANCE
            )))

        # ── Joint-angle posture reward (hip-knee angle targets per mode) ──
        # Ref: ETH legged_gym / RMA target joint configurations
        hip_q = q[HIP_JOINT_INDICES]
        knee_q = q[KNEE_JOINT_INDICES]
        posture_key = self.command_mode if self.command_mode in POSTURE_TARGETS else "stand"
        p_target = POSTURE_TARGETS[posture_key]
        hip_err = float(np.sum((hip_q - p_target["hip"]) ** 2))
        knee_err = float(np.sum((knee_q - p_target["knee"]) ** 2))
        r_posture = math.exp(-(hip_err + knee_err) / 0.5)

        # ── Foot-strike velocity penalty (penalize fast downward impact at touchdown) ──
        foot_heights_now = self.data.site_xpos[self._foot_site_ids, 2]
        r_foot_strike_vel = 0.0
        if np.any(first_contact):
            foot_z_vel = (foot_heights_now - self._prev_foot_heights) / self.dt
            r_foot_strike_vel = float(np.sum(foot_z_vel[first_contact] ** 2))
        self._prev_foot_heights = foot_heights_now.copy()

        # ── Body linear acceleration penalty (anti-shake) ──
        body_acc = (base_linvel - self._prev_base_linvel) / self.dt
        r_body_acc = float(np.sum(body_acc ** 2))
        self._prev_base_linvel = base_linvel.copy()

        # ── Second-order action smoothness (jerk penalty) ──
        action_jerk = action - 2.0 * self.prev_action + self.prev_prev_action
        r_action_jerk = float(np.sum(action_jerk ** 2))

        # ── Jump FSM phase reward ──
        r_jump_phase = self._advance_jump_fsm(base_z, base_linvel, foot_contacts)

        # ── Assemble with REWARD_SCALES ──
        components = {
            "r_alive": 1.0,          # constant survival bonus (legged_gym convention)
            "r_linvel": r_linvel,
            "r_yaw": r_yaw,
            "r_feet_air_time": r_feet_air_time,
            "r_cmd_vel_error": r_cmd_vel_error,
            "r_orientation": r_orientation,
            "r_lin_vel_z": r_lin_vel_z,
            "r_ang_vel_xy": r_ang_vel_xy,
            "r_height": r_height,
            "r_torque": r_torque,
            "r_smooth": r_smooth,
            "r_joint_acc": r_joint_acc,
            "r_joint_limit": r_joint_limit,
            "r_crouch_penalty": r_crouch_penalty,
            "r_power": r_power,
            "r_stand_still": r_stand_still,
            "r_gait_phase": r_gait_phase,
            "r_foot_clearance": r_foot_clearance,
            "r_posture": r_posture,
            "r_foot_strike_vel": r_foot_strike_vel,
            "r_body_acc": r_body_acc,
            "r_action_jerk": r_action_jerk,
            "r_jump_phase": r_jump_phase,
        }

        total = sum(REWARD_SCALES[k] * v for k, v in components.items())

        if ONLY_POSITIVE_REWARDS:
            total = max(total, 0.0)

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
        # Adaptive min-height: lower threshold when crouching
        min_height = 0.08 if self.command_mode == "crouch" else 0.13
        # Fallen: too low or tilted > ~45 degrees (cos 45° ≈ 0.707)
        return bool(base_z < min_height or gravity_body[2] > -0.7)

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

    # ── Jump finite state machine ──────────────────────────────────

    def _advance_jump_fsm(self, base_z, base_linvel, foot_contacts):
        """Advance jump FSM through crouch→launch→airborne→land phases.

        Returns a phase-specific reward scalar. The FSM dynamically
        adjusts self.target_height so the policy sees the current phase
        target in its observation.
        """
        # Reset FSM if not in jump mode
        if self.command_mode != "jump":
            if self._jump_phase != JUMP_PHASE_IDLE:
                self._jump_phase = JUMP_PHASE_IDLE
                self._jump_step_counter = 0
            return 0.0

        r = 0.0

        if self._jump_phase == JUMP_PHASE_IDLE:
            # Enter crouch preparation
            self._jump_phase = JUMP_PHASE_CROUCH
            self._jump_step_counter = 0
            self._jump_max_height = base_z
            self.target_height = 0.18

        elif self._jump_phase == JUMP_PHASE_CROUCH:
            self.target_height = 0.18
            self._jump_step_counter += 1
            # Reward lowering body toward crouch height
            r = max(0.0, 0.27 - base_z) * 2.0
            if self._jump_step_counter >= JUMP_CROUCH_STEPS:
                self._jump_phase = JUMP_PHASE_LAUNCH
                self._jump_step_counter = 0

        elif self._jump_phase == JUMP_PHASE_LAUNCH:
            self.target_height = 0.40
            self._jump_step_counter += 1
            # Reward upward velocity
            r = max(0.0, float(base_linvel[2])) * 3.0
            if self._jump_step_counter >= JUMP_LAUNCH_STEPS:
                self._jump_phase = JUMP_PHASE_AIRBORNE
                self._jump_step_counter = 0

        elif self._jump_phase == JUMP_PHASE_AIRBORNE:
            self.target_height = 0.40
            self._jump_step_counter += 1
            self._jump_max_height = max(self._jump_max_height, base_z)
            # Reward height achieved above standing
            r = max(0.0, base_z - 0.27) * 2.0
            # Transition to landing: descending + feet touching, or timeout
            n_contacts = int(np.sum(foot_contacts))
            descending = float(base_linvel[2]) < -0.1
            if (descending and n_contacts >= 2) or self._jump_step_counter > JUMP_AIRBORNE_MAX:
                self._jump_phase = JUMP_PHASE_LANDING
                self._jump_step_counter = 0

        elif self._jump_phase == JUMP_PHASE_LANDING:
            self.target_height = 0.27
            self._jump_step_counter += 1
            # Reward stability: height near normal + low velocity
            height_err = (base_z - 0.27) ** 2
            vel_err = float(np.sum(base_linvel ** 2))
            r = math.exp(-(height_err + 0.1 * vel_err) / 0.2)
            if self._jump_step_counter >= JUMP_LANDING_STEPS:
                # Bonus for maximum height achieved
                r += max(0.0, self._jump_max_height - 0.30) * 5.0
                self._jump_phase = JUMP_PHASE_IDLE
                self._jump_step_counter = 0
                self.target_height = HEIGHT_TARGETS.get("stand", 0.27)

        return float(r)

    # ── Command interface ───────────────────────────────────────────

    def set_command(self, vx: float, vy: float, wz: float, mode: str = "trot"):
        self.command = np.array([vx, vy, wz], dtype=np.float32)
        self.command_mode = mode
        self.target_height = HEIGHT_TARGETS.get(mode, 0.27)

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
                self.model.geom_friction[i][0] = 1.5 * friction_scale

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
