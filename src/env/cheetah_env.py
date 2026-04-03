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
# CALIBRATION v3 — informed by paper research + diagnostic analysis:
#   legged_gym (Rudin 2022): alive=0, tracking is main positive signal,
#       _reward_dof_vel (joint velocity penalty), _reward_stand_still
#   Walk These Ways (Margolis 2023): no alive bonus, tracking dominant
#   RMA (Kumar 2021): task reward = velocity tracking only
#   Humanoid-Gym (Gu 2024): body height as exp-kernel reward, dof_vel=-5e-4
#   Chi 2025: base height, contact timing, orientation penalty
#
# Previous issues (v2):
#   1. Missing r_dof_vel: no joint velocity penalty → shaking unpunished
#   2. r_body_height exp kernel always positive → rewards wrong height
#      (at 9cm off target, still gets +0.40/step)
#   3. r_posture exp kernel σ=0.5 too narrow → zero gradient for crouch
#      (at crouch distance, exp(-6.8) = 0.001 → no incentive)
#   4. r_stand_still fires in crouch mode → opposes crouch posture target
#   5. command_mode never randomized → model never trains on crouch/run
#   6. r_smooth/-0.05 and r_body_acc/-0.0005 too weak to deter oscillation
#   7. Net stand reward +8.3/step with stand_still only -0.14 (1.7%)
#
# v3 fixes: add r_dof_vel, center body_height at 0, flip posture to
#   negative penalty, gate stand_still by mode, randomize command_mode,
#   strengthen anti-shake penalties.
# Design: tracking rewards still dominate, but penalties are meaningful.
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
    # ── Positive rewards (tracking should dominate: ~6-8/step at convergence) ──
    "r_alive":           0.2,      # minimal survival tie-breaker (legged_gym uses 0)
    "r_linvel":          4.0,      # exp kernel xy-vel tracking (main signal)
    "r_yaw":             2.0,      # exp kernel yaw-rate tracking
    "r_feet_air_time":   2.0,      # gait: leg-swing reward gated by cmd speed
    "r_gait_phase":      1.5,      # trot diagonal contact symmetry
    "r_foot_clearance":  0.5,      # swing-foot height reward
    "r_body_height":     2.0,      # exp-kernel body height (centered at 0: exp-1)
    "r_jump_phase":      1.0,      # jump FSM phase-specific rewards
    # ── Penalties (v3: strengthened anti-shake, fixed crouch) ──
    "r_cmd_vel_error":  -1.0,      # squared velocity error
    "r_orientation":    -2.0,      # gravity-tilt penalty (anti-flip)
    "r_lin_vel_z":      -2.0,      # vertical bounce penalty
    "r_ang_vel_xy":     -0.01,     # roll/pitch rate penalty
    "r_height":         -5.0,      # height deviation squared
    "r_torque":         -5e-6,     # torque squared
    "r_smooth":         -0.3,      # action rate penalty (was -0.05; 6× for anti-shake)
    "r_joint_acc":      -5e-8,     # joint accel penalty
    "r_joint_limit":    -5.0,      # joint limit proximity penalty
    "r_crouch_penalty": -5.0,      # anti-crouch: sustained low height
    "r_power":          -1e-5,     # mechanical power |tau·qvel|
    "r_stand_still":    -3.0,      # joint deviation at zero cmd (was -0.5; gated by mode)
    "r_foot_strike_vel":-0.005,    # foot velocity at contact
    "r_body_acc":       -0.005,    # body linear accel (was -0.0005; 10× anti-shake)
    "r_action_jerk":    -0.02,     # second-order smoothness (was -0.005; 4× stronger)
    "r_abduction":      -2.0,      # penalize leg splay (abd joints 0,3,6,9 from 0)
    "r_hip_excess":     -3.0,      # penalize excessive hip flexion (anti-belly-sit)
    "r_posture":        -1.5,      # v3: negative penalty for joint deviation from target
    "r_dof_vel":        -0.1,      # v3 NEW: joint velocity penalty (legged_gym key anti-shake)
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
# Abduction joints (indices 0,3,6,9): default 0.0 — keep legs under body.
# Hip joint = rotation joint (indices 1,4,7,10), Knee joint (indices 2,5,8,11).
ABD_JOINT_INDICES  = np.array([0, 3, 6, 9],  dtype=np.int32)
HIP_JOINT_INDICES  = np.array([1, 4, 7, 10], dtype=np.int32)
KNEE_JOINT_INDICES = np.array([2, 5, 8, 11], dtype=np.int32)

# Body-height exp-kernel sigma (Humanoid-Gym, Chi 2025).
# v3: widened from 0.005 → 0.02 for better gradient at crouch distance.
# At ±5 cm: exp(-0.0025/0.02) = exp(-0.125) ≈ 0.88
# At ±9 cm (crouch→stand): exp(-0.0081/0.02) = exp(-0.405) ≈ 0.67
# The kernel is now centered at 0 (exp-1) so these become -0.12 and -0.33.
BODY_HEIGHT_SIGMA = 0.02

# Hip flexion limit for non-crouch modes (rad).  Go1 default stance = 0.9.
# Beyond this threshold, the robot is sitting on its belly.
HIP_EXCESS_THRESHOLD = 1.3
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
            # v3: randomize command_mode so the policy trains on all behaviors
            modes = ["stand", "walk", "trot", "run", "explore", "crouch"]
            mode_weights = [0.15, 0.15, 0.25, 0.15, 0.10, 0.20]
            self.command_mode = str(rng.choice(modes, p=mode_weights))
            self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)
            # Mode-appropriate velocity ranges
            if self.command_mode == "crouch":
                vx = float(rng.uniform(-0.1, 0.3))
                vy = float(rng.uniform(-0.1, 0.1))
                wz = float(rng.uniform(-0.2, 0.2))
            elif self.command_mode == "run":
                vx = float(rng.uniform(0.5, 2.0))
                vy = float(rng.uniform(-0.3, 0.3))
                wz = float(rng.uniform(-0.3, 0.3))
            elif self.command_mode == "stand":
                # 50% chance of truly zero command for stand practice
                if rng.random() < 0.5:
                    vx, vy, wz = 0.0, 0.0, 0.0
                else:
                    vx = float(rng.uniform(-0.3, 0.5))
                    vy = float(rng.uniform(-0.3, 0.3))
                    wz = float(rng.uniform(-0.3, 0.3))
            else:
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
            # v3: also re-randomize command_mode mid-episode
            modes = ["stand", "walk", "trot", "run", "explore", "crouch"]
            mode_weights = [0.15, 0.15, 0.25, 0.15, 0.10, 0.20]
            self.command_mode = str(rng.choice(modes, p=mode_weights))
            self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)
            if self.command_mode == "crouch":
                vx = float(rng.uniform(-0.1, 0.3))
                vy = float(rng.uniform(-0.1, 0.1))
                wz = float(rng.uniform(-0.2, 0.2))
            elif self.command_mode == "run":
                vx = float(rng.uniform(0.5, 2.0))
                vy = float(rng.uniform(-0.3, 0.3))
                wz = float(rng.uniform(-0.3, 0.3))
            elif self.command_mode == "stand":
                if rng.random() < 0.5:
                    vx, vy, wz = 0.0, 0.0, 0.0
                else:
                    vx = float(rng.uniform(-0.3, 0.5))
                    vy = float(rng.uniform(-0.3, 0.3))
                    wz = float(rng.uniform(-0.3, 0.3))
            else:
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

        # ── Anti-crouch penalty (time-series + joint-angle check) ──
        # The old v1 check only looked at height.  The robot exploited this by
        # splaying abduction joints outward while keeping hip/knee near target.
        # v2: also fires when abduction or hip angles indicate belly-sitting.
        self._height_history.append(base_z)
        r_crouch_penalty = 0.0
        if self.command_mode != "crouch":
            if len(self._height_history) == CROUCH_DETECT_WINDOW:
                mean_height = sum(self._height_history) / CROUCH_DETECT_WINDOW
                if mean_height < CROUCH_HEIGHT_THRESHOLD:
                    r_crouch_penalty = (CROUCH_HEIGHT_THRESHOLD - mean_height) ** 2
            # Instantaneous check: if body very low right now, penalize immediately
            if base_z < CROUCH_HEIGHT_THRESHOLD:
                instant_penalty = (CROUCH_HEIGHT_THRESHOLD - base_z) ** 2
                r_crouch_penalty = max(r_crouch_penalty, instant_penalty)

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
        # v3: gate by command_mode — do NOT penalize in crouch/jump where joints
        # should deviate from DEFAULT_STANCE to achieve lower/higher postures.
        r_stand_still = 0.0
        if (cmd_speed < 0.1 and abs(wz_cmd) < 0.1
                and self.command_mode not in ("crouch", "jump")):
            r_stand_still = float(np.sum(np.abs(
                self.data.qpos[7:7 + NUM_JOINTS] - DEFAULT_STANCE
            )))

        # ── Joint-angle posture penalty (v3: negative squared error, not exp kernel) ──
        # v2 used exp(-error/0.5) which gave zero gradient at crouch distance
        # (error=3.4: exp(-6.8)=0.001). v3 uses squared error directly so the
        # policy always has gradient toward the mode-dependent posture target.
        hip_q = q[HIP_JOINT_INDICES]
        knee_q = q[KNEE_JOINT_INDICES]
        posture_key = self.command_mode if self.command_mode in POSTURE_TARGETS else "stand"
        p_target = POSTURE_TARGETS[posture_key]
        hip_err = float(np.sum((hip_q - p_target["hip"]) ** 2))
        knee_err = float(np.sum((knee_q - p_target["knee"]) ** 2))
        r_posture = hip_err + knee_err

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

        # ── Body height tracking (v3: centered exp kernel, Humanoid-Gym / Chi 2025) ──
        # v2 used exp(-err²/σ) which is always ∈ (0,1] — positive even at wrong height.
        # v3 subtracts 1.0: zero at target, negative elsewhere.  This prevents the
        # +0.40/step reward at 9 cm off target that was overwhelming the r_height penalty.
        r_body_height = math.exp(-(base_z - self.target_height) ** 2 / BODY_HEIGHT_SIGMA) - 1.0

        # ── NEW v3: Abduction joint penalty (anti-splay) ──
        # Default stance has abduction=0.  Penalize deviation to prevent the
        # robot from splaying legs outward to lower its body.
        abd_q = q[ABD_JOINT_INDICES]
        r_abduction = float(np.sum(abd_q ** 2))

        # ── NEW v3: Joint velocity penalty (legged_gym _reward_dof_vel) ──
        # Directly penalizes joint oscillation — the key anti-shake mechanism
        # that was missing in v2.  legged_gym uses sum(dof_vel²).
        r_dof_vel = float(np.sum(joint_vel ** 2))

        # ── NEW: Excessive hip flexion penalty (anti-belly-sit) ──
        # If hip (thigh) joints flex beyond HIP_EXCESS_THRESHOLD (1.3 rad) in
        # non-crouch modes, the robot is folding its legs underneath → belly sit.
        r_hip_excess = 0.0
        if self.command_mode != "crouch":
            hip_over = np.clip(hip_q - HIP_EXCESS_THRESHOLD, 0, None)
            r_hip_excess = float(np.sum(hip_over ** 2))

        # ── Assemble with REWARD_SCALES ──
        components = {
            "r_alive": 1.0,
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
            "r_body_height": r_body_height,
            "r_abduction": r_abduction,
            "r_hip_excess": r_hip_excess,
            "r_dof_vel": r_dof_vel,
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
        min_height = 0.08 if self.command_mode == "crouch" else 0.18
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
