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
ACT_DIM = 12

# ── Skill modes the robot must learn on command ──────────────────────
SKILL_MODES = ["stand", "walk", "run", "crouch", "jump"]
SKILL_DIM = len(SKILL_MODES)       # 5-dim one-hot in observation
SKILL_TO_IDX = {m: i for i, m in enumerate(SKILL_MODES)}

OBS_DIM = 54  # 49 (base) + 5 (skill one-hot)
#   [0:12]  joint positions
#   [12:24] joint velocities
#   [24:27] base linear velocity (body frame)
#   [27:30] base angular velocity (body frame)
#   [30:33] projected gravity
#   [33:45] previous action
#   [45:49] command (vx, vy, wz, target_height)
#   [49:54] skill one-hot encoding

# Default standing pose: [abduct, hip, knee] × 4 legs (FR, FL, RR, RL)
# Unitree Go1 home keyframe from mujoco_menagerie
DEFAULT_STANCE = np.array([
    0.0,  0.9, -1.8,  # FR
    0.0,  0.9, -1.8,  # FL
    0.0,  0.9, -1.8,  # RR
    0.0,  0.9, -1.8,  # RL
], dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════
#  Reward v6 — Gait-focused, smoothness-fixed, training-tuned
#
#  Problems fixed from v5 (10M step training analysis):
#    1. r_smooth L2 at -0.05 caused -0.15 to -0.19/step → choppy gait
#       Fix: L1 at -0.01 (legged_gym uses -0.01; tolerates cyclic swings)
#    2. r_gait only 0.35-0.53/step: clearance_target=5cm too low,
#       trot_symmetry binary, no stride frequency incentive
#       Fix: clearance→8cm, sqrt symmetry, +stride_freq, scale 2.0→3.5
#    3. LR decayed to 1.4e-6 → clip_fraction→0, std frozen at 0.637
#       Fix: min_lr=1e-5 floor, log_std_init=-1.0
#    4. Only 40 grad steps/rollout (n_epochs=5, batch=4096)
#       Fix: n_epochs=10, batch=2048 → 160 grad steps/rollout
#    5. r_dof_vel constant -0.315/step dominated penalties
#       Fix: scale -2e-4 → -1e-4 (halved)
#    6. r_linvel too low for walk/run modes
#       Fix: scale 4.0→5.0, walk mult 1.2, run mult 2.0
#
#  Research references (quadruped gait, analogous to bipedal LIPM/LOM):
#    - SLIP model (Spring-Loaded Inverted Pendulum): stride frequency
#    - legged_gym (Rudin 2022): r_smooth scale, clearance targets
#    - Walk These Ways (Margolis 2023): per-mode multipliers, gait phase
#    - DreamWaQ (Nahrendra 2023): gait oscillator / phase variables
# ══════════════════════════════════════════════════════════════════════

# Height targets per skill mode
HEIGHT_TARGETS = {
    "stand":  0.27,
    "walk":   0.27,
    "run":    0.27,
    "crouch": 0.18,
    "jump":   0.35,   # overridden dynamically by jump FSM
}

# Anti-crouch: rolling window to detect sustained unwanted crouching
CROUCH_DETECT_WINDOW = 10       # steps (~0.2s at 50 Hz)
CROUCH_HEIGHT_THRESHOLD = 0.22  # below this counts as "crouching"

# Go1 approximate total mass (kg) — for cost-of-transport scaling
ROBOT_MASS = 12.74

# Survival multiplier: scales total reward by sqrt(t / T), encouraging longer
# episodes. Capped so it never dominates. At t=0 → 1.0×; at t=T → ALIVE_SCALE_MAX×.
ALIVE_SCALE_MAX = 3.0  # cap (e.g. 3× at full episode length)

# ── Base reward scales (apply to all modes) ──────────────────────────
REWARD_SCALES = {
    # Positive rewards (r_alive removed — now a multiplicative survival factor)
    "r_linvel":         5.0,       # exp-kernel xy-vel tracking (up from 4.0)
    "r_yaw":            1.5,       # exp-kernel yaw-rate tracking
    "r_gait":           3.5,       # combined: trot symmetry + foot clearance + air time (up from 2.0)
    "r_posture":        2.0,       # exp-kernel joint-angle tracking (mode-dependent target)
    "r_body_height":    2.0,       # exp-kernel height tracking (centered at 0)
    "r_stillness":      1.0,       # reward for minimal motion (stand/crouch)
    "r_jump_phase":     3.0,       # jump FSM phase-specific rewards
    # Penalties
    "r_orientation":   -1.0,       # gravity-tilt squared
    "r_torque":        -5e-6,      # torque squared
    "r_smooth":        -0.01,      # action rate (L1); was -0.05 L2 — over-penalised gait swings
    "r_joint_limit":   -2.0,       # joint limit proximity squared
    "r_lin_vel_z":     -0.5,       # vertical bounce squared
    "r_dof_vel":       -1e-4,      # joint velocity squared (halved; was -2e-4)
}

# ── Per-mode reward multipliers (Walk These Ways / multi-skill style) ──
# Keys must match REWARD_SCALES. Values multiply the base scale.
# 0.0 = term disabled for that mode; >1.0 = amplified.
MODE_REWARD_MULTIPLIERS = {
    "stand": {
        "r_linvel": 0.3,       # low: standing doesn't need velocity tracking
        "r_yaw": 0.3,
        "r_gait": 0.0,         # no gait reward when standing
        "r_posture": 3.0,      # high: maintain standing posture
        "r_body_height": 2.0,
        "r_stillness": 3.0,    # high: stay still
        "r_jump_phase": 0.0,
    },
    "walk": {
        "r_linvel": 1.2,
        "r_yaw": 1.0,
        "r_gait": 2.0,         # amplified: proper gait is key for walking
        "r_posture": 1.0,
        "r_body_height": 1.0,
        "r_stillness": 0.0,    # no stillness reward when walking
        "r_jump_phase": 0.0,
    },
    "run": {
        "r_linvel": 2.0,       # strongly amplified: speed is the main goal
        "r_yaw": 1.0,
        "r_gait": 1.5,
        "r_posture": 0.5,      # relaxed: running posture differs
        "r_body_height": 0.5,
        "r_stillness": 0.0,
        "r_jump_phase": 0.0,
        "r_smooth": 0.5,       # halve smoothness penalty for fast motion
    },
    "crouch": {
        "r_linvel": 0.3,       # low speed tracking
        "r_yaw": 0.3,
        "r_gait": 0.0,
        "r_posture": 3.0,      # high: maintain crouch posture
        "r_body_height": 3.0,  # high: reach target crouch height
        "r_stillness": 2.0,    # mostly still
        "r_jump_phase": 0.0,
    },
    "jump": {
        "r_linvel": 0.2,
        "r_yaw": 0.2,
        "r_gait": 0.0,
        "r_posture": 0.5,
        "r_body_height": 1.0,  # tracks FSM phase height
        "r_stillness": 0.0,
        "r_jump_phase": 1.0,   # jump FSM is the main signal
        "r_lin_vel_z": -0.1,   # relax vertical bounce penalty during jump
    },
}

# Tracking σ for exp kernel: legged_gym default is 0.25.
TRACKING_SIGMA = 0.25

# Posture exp-kernel σ — wider than tracking for smoother gradient
POSTURE_SIGMA = 0.5

# Feet air-time threshold (seconds).
FEET_AIR_TIME_THRESHOLD = 0.15

# v6: No ONLY_POSITIVE_REWARDS. Survival multiplier and exp-kernel
# rewards keep most transitions net positive. Removing the clip lets the
# policy learn from negative episodes instead of getting zero gradient.
ONLY_POSITIVE_REWARDS = False

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
    "run":     {"hip": 0.8,  "knee": -1.6},
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
            # v5: 5 skill modes with jump included
            mode_weights = [0.20, 0.25, 0.20, 0.15, 0.20]
            self.command_mode = str(rng.choice(SKILL_MODES, p=mode_weights))
            self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)
            self._randomize_command_for_mode(rng)

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
            # v5: 5 skill modes with jump included
            mode_weights = [0.20, 0.25, 0.20, 0.15, 0.20]
            self.command_mode = str(rng.choice(SKILL_MODES, p=mode_weights))
            self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)
            self._randomize_command_for_mode(rng)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_info()

    def _randomize_command_for_mode(self, rng):
        """Set velocity commands appropriate for the current command_mode."""
        mode = self.command_mode
        if mode == "stand":
            if rng.random() < 0.5:
                vx, vy, wz = 0.0, 0.0, 0.0
            else:
                vx = float(rng.uniform(-0.2, 0.3))
                vy = float(rng.uniform(-0.2, 0.2))
                wz = float(rng.uniform(-0.3, 0.3))
        elif mode == "walk":
            vx = float(rng.uniform(0.2, 1.0))
            vy = float(rng.uniform(-0.3, 0.3))
            wz = float(rng.uniform(-0.5, 0.5))
        elif mode == "run":
            vx = float(rng.uniform(1.0, 2.5))
            vy = float(rng.uniform(-0.3, 0.3))
            wz = float(rng.uniform(-0.3, 0.3))
        elif mode == "crouch":
            vx = float(rng.uniform(-0.1, 0.2))
            vy = float(rng.uniform(-0.1, 0.1))
            wz = float(rng.uniform(-0.2, 0.2))
        elif mode == "jump":
            # Mostly forward, small lateral
            vx = float(rng.uniform(0.0, 0.5))
            vy = float(rng.uniform(-0.1, 0.1))
            wz = float(rng.uniform(-0.1, 0.1))
        else:
            vx = float(rng.uniform(-0.5, 2.0))
            vy = float(rng.uniform(-0.5, 0.5))
            wz = float(rng.uniform(-0.5, 0.5))
        self.command = np.array([vx, vy, wz], dtype=np.float32)

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

        # Skill one-hot encoding (5 dims)
        skill_onehot = np.zeros(SKILL_DIM, dtype=np.float32)
        skill_idx = SKILL_TO_IDX.get(self.command_mode, 0)
        skill_onehot[skill_idx] = 1.0

        obs = np.concatenate([
            qpos, qvel, base_linvel, base_angvel,
            gravity_body, self.prev_action,
            np.append(self.command, self.target_height),
            skill_onehot,
        ]).astype(np.float32)

        if self.randomize_domain:
            # Add noise only to sensor dims (0:49), NOT to the skill encoding
            noise = np.zeros(OBS_DIM, dtype=np.float32)
            noise[:49] = self.np_random.standard_normal(49).astype(np.float32) * 0.02
            obs += noise

        return obs

    # ── Reward ──────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        """Reward v6: Gait-focused, L1 smoothness, stride frequency."""
        quat = self.data.qpos[3:7]
        base_linvel = self._quat_rotate_inv(quat, self.data.qvel[:3])
        base_angvel = self._quat_rotate_inv(quat, self.data.qvel[3:6])
        tau = self.data.ctrl[:NUM_JOINTS]
        joint_vel = self.data.qvel[6:6 + NUM_JOINTS]
        q = self.data.qpos[7:7 + NUM_JOINTS]
        base_z = float(self.data.qpos[2])

        vx_cmd, vy_cmd, wz_cmd = self.command
        cmd_speed = math.sqrt(vx_cmd ** 2 + vy_cmd ** 2)
        mode = self.command_mode

        # ── 1. Velocity tracking (exp kernel) ──
        lin_vel_error = (base_linvel[0] - vx_cmd) ** 2 + (base_linvel[1] - vy_cmd) ** 2
        r_linvel = math.exp(-lin_vel_error / TRACKING_SIGMA)

        # ── 2. Yaw rate tracking (exp kernel) ──
        ang_vel_error = (base_angvel[2] - wz_cmd) ** 2
        r_yaw = math.exp(-ang_vel_error / TRACKING_SIGMA)

        # ── 3. Combined gait reward (trot symmetry + clearance + air time) ──
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

        # Air time sub-reward
        air_time_reward = float(np.sum(
            (self._feet_air_time - FEET_AIR_TIME_THRESHOLD) * first_contact
        ))
        if cmd_speed < 0.1:
            air_time_reward = 0.0
        self._feet_air_time[contact_filt] = 0.0

        # Trot symmetry sub-reward: diagonal pairs should alternate.
        # In trot, when FR+RL are in stance, FL+RR are in swing (|diff|=2).
        # sqrt makes gradient smoother for partial trot patterns.
        diag1 = float(contact_filt[0] + contact_filt[3])  # FR + RL
        diag2 = float(contact_filt[1] + contact_filt[2])  # FL + RR
        trot_symmetry = math.sqrt(abs(diag1 - diag2) / 2.0) if cmd_speed > 0.1 else 0.0

        # Foot clearance sub-reward (raised target for proper leg swings)
        foot_heights = self.data.site_xpos[self._foot_site_ids, 2]
        swing_mask = ~contact_filt
        clearance_target = 0.08  # 8cm (was 5cm — too low for natural gait)
        foot_clearance = 0.0
        if np.any(swing_mask) and cmd_speed > 0.1:
            swing_heights = foot_heights[swing_mask]
            foot_clearance = float(np.mean(
                np.clip(swing_heights, 0, clearance_target) / clearance_target
            ))

        # Stride frequency sub-reward: encourages regular stepping cadence.
        # Counts feet that touched down this step; ideal trot = 2 touchdowns
        # per cycle. Reward peaks when exactly 2 feet make first contact.
        n_touchdowns = float(np.sum(first_contact))
        stride_freq_reward = math.exp(-0.5 * (n_touchdowns - 2.0) ** 2) if cmd_speed > 0.1 else 0.0

        # Combined gait = weighted sum of four sub-components
        r_gait = (0.3 * air_time_reward
                  + 0.25 * trot_symmetry
                  + 0.25 * foot_clearance
                  + 0.2 * stride_freq_reward)

        # ── 4. Posture tracking (exp kernel on joint angles) ──
        hip_q = q[HIP_JOINT_INDICES]
        knee_q = q[KNEE_JOINT_INDICES]
        posture_key = mode if mode in POSTURE_TARGETS else "stand"
        p_target = POSTURE_TARGETS[posture_key]
        posture_err = (float(np.sum((hip_q - p_target["hip"]) ** 2))
                       + float(np.sum((knee_q - p_target["knee"]) ** 2)))
        r_posture = math.exp(-posture_err / POSTURE_SIGMA)

        # ── 5. Body height tracking (centered exp kernel) ──
        r_body_height = math.exp(-(base_z - self.target_height) ** 2 / BODY_HEIGHT_SIGMA) - 1.0
        self._height_history.append(base_z)

        # ── 6. Stillness reward (for stand/crouch modes) ──
        r_stillness = 0.0
        if mode in ("stand", "crouch"):
            joint_motion = float(np.sum(joint_vel ** 2))
            body_motion = float(np.sum(base_linvel[:2] ** 2))
            r_stillness = math.exp(-(joint_motion * 0.001 + body_motion) / 0.5)

        # ── 7. Jump FSM phase reward ──
        r_jump_phase = self._advance_jump_fsm(base_z, base_linvel, foot_contacts)

        # ── 8. Orientation penalty ──
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        r_orientation = float(np.sum(gravity_body[:2] ** 2))

        # ── 9. Torque penalty ──
        r_torque = float(np.sum(tau ** 2))

        # ── 10. Action smoothness penalty (L1 — tolerates cyclic gait swings) ──
        r_smooth = float(np.sum(np.abs(action - self.prev_action)))

        # ── 11. Joint limit proximity penalty ──
        jnt_range = self.model.jnt_range[1:NUM_JOINTS + 1]
        margin = 0.1
        below = np.clip(jnt_range[:, 0] + margin - q, 0, None)
        above = np.clip(q - (jnt_range[:, 1] - margin), 0, None)
        r_joint_limit = float(np.sum(below ** 2 + above ** 2))

        # ── 12. Vertical bounce penalty ──
        r_lin_vel_z = float(base_linvel[2] ** 2)

        # ── 13. Joint velocity penalty (anti-shake) ──
        r_dof_vel = float(np.sum(joint_vel ** 2))

        # ── Update tracking state ──
        self._prev_base_linvel = base_linvel.copy()
        self._prev_foot_heights = foot_heights.copy()

        # ── Assemble with mode-dependent reward scales ──
        raw_components = {
            "r_linvel":      r_linvel,
            "r_yaw":         r_yaw,
            "r_gait":        r_gait,
            "r_posture":     r_posture,
            "r_body_height": r_body_height,
            "r_stillness":   r_stillness,
            "r_jump_phase":  r_jump_phase,
            "r_orientation": r_orientation,
            "r_torque":      r_torque,
            "r_smooth":      r_smooth,
            "r_joint_limit": r_joint_limit,
            "r_lin_vel_z":   r_lin_vel_z,
            "r_dof_vel":     r_dof_vel,
        }

        # Apply base scales, then mode-specific multipliers
        mode_mults = MODE_REWARD_MULTIPLIERS.get(mode, {})
        total = 0.0
        scaled_components = {}
        for k, raw_val in raw_components.items():
            base_scale = REWARD_SCALES[k]
            mode_mult = mode_mults.get(k, 1.0)
            scaled = base_scale * mode_mult * raw_val
            scaled_components[k] = scaled
            total += scaled

        if ONLY_POSITIVE_REWARDS:
            total = max(total, 0.0)

        # ── Survival multiplier: sqrt(t / T), capped at ALIVE_SCALE_MAX ──
        # Grows from 1.0 at step 0 to ALIVE_SCALE_MAX at t=episode_length,
        # encouraging the policy to stay alive longer without a fixed additive bonus.
        t_frac = self.step_count / max(self.episode_length, 1)
        survival_mult = min(
            1.0 + math.sqrt(t_frac) * (ALIVE_SCALE_MAX - 1.0),
            ALIVE_SCALE_MAX,
        )
        total *= survival_mult

        scaled_components["survival_mult"] = survival_mult
        scaled_components["r_total"] = total
        self._last_reward_components = scaled_components

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

    def set_command(self, vx: float, vy: float, wz: float, mode: str = "walk"):
        self.command = np.array([vx, vy, wz], dtype=np.float32)
        self.command_mode = mode if mode in SKILL_MODES else "walk"
        self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)

    def set_exploration_heading(self, heading_rad: float, speed: float = 1.5):
        vx = speed * math.cos(heading_rad)
        vy = speed * math.sin(heading_rad)
        self.set_command(vx, vy, 0.0, "walk")

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
