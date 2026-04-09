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

Action (12 dims): delta joint position targets, scaled +-0.2 rad (v12)

Reward v8 (based on ETH Zurich legged_gym, Walk These Ways, RMA):
  + exp tracking for vx velocity (σ=0.25)
  + exp tracking for vy velocity (σ=0.15, tighter for lateral precision)
  + exp tracking for yaw rate (σ=0.25, scale 2.0 — 4x stronger than v7)
  + combined gait reward (air time + symmetry + clearance + stride freq)
  + posture tracking (joint-angle exp kernel)
  + body height tracking (exp kernel, proper [0,1] reward)
  + stillness reward (stand/crouch modes)
  + motion penalty (explicit penalty for motion in stand/crouch — NEW v8)
  + constant alive bonus (0.5/step, replaces survival multiplier)
  - orientation penalty (gravity tilt squared)
  - angular velocity xy penalty (roll/pitch rate squared)
  - torque penalty (squared)
  - action smoothness penalty (L2 action rate, legged_gym standard)
  - joint limit proximity penalty
  - vertical bounce penalty (v_z squared)
  - joint velocity penalty (squared)
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

OBS_DIM = 61  # 49 (base) + 5 (augmented) + 5 (skill one-hot) + 2 (cpg phase)
#   [0:12]  joint positions
#   [12:24] joint velocities
#   [24:27] base linear velocity (body frame)
#   [27:30] base angular velocity (body frame)
#   [30:33] projected gravity
#   [33:45] previous action
#   [45:49] command (vx, vy, wz, target_height)
#   [49:50] base height (critical for crouch/jump/height tracking)
#   [50:54] foot contacts (binary; critical for gait timing, standard in legged_gym)
#   [54:59] skill one-hot encoding
#   [59:61] CPG phase [sin(phase), cos(phase)] — enables phase-dependent residuals

# ── Manual observation normalization (replaces VecNormalize) ──────────
# Fixed divisors so each observation dimension is roughly in [-1, 1].
# This avoids VecNormalize's running-statistics drift that causes
# catastrophic forgetting when normalization stats change mid-training.
OBS_SCALES = np.array(
    [1.0]*12        # joint positions: already ~[-1.5, 1.5]
  + [10.0]*12       # joint velocities: ~[-10, 10] → [-1, 1]
  + [2.0]*3         # base linear velocity: ~[-2, 2] → [-1, 1]
  + [5.0]*3         # base angular velocity: ~[-5, 5] → [-1, 1]
  + [1.0]*3         # gravity: already [-1, 1]
  + [1.0]*12        # previous action: [-1, 1] → [-1, 1] (v12: action space is [-1,1])
  + [2.0, 0.5, 0.8, 0.35]  # commands: vx/2, vy/0.5, wz/0.8, h/0.35
  + [0.35]          # base height: ~0.3 → ~0.86
  + [1.0]*4         # foot contacts: already [0, 1]
  + [1.0]*5         # skill one-hot: already [0, 1]
  + [1.0]*2,        # CPG phase [sin, cos]: already [-1, 1]
  dtype=np.float32
)

# Default standing pose: [abduct, hip, knee] × 4 legs (FR, FL, RR, RL)
# Unitree Go1 home keyframe from mujoco_menagerie
DEFAULT_STANCE = np.array([
    0.0,  0.9, -1.8,  # FR
    0.0,  0.9, -1.8,  # FL
    0.0,  0.9, -1.8,  # RR
    0.0,  0.9, -1.8,  # RL
], dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════
#  Reward v7 — Paper-aligned, final calibration
#
#  Problems fixed from v6 (10M step CSV analysis):
#    1. r_body_height formula exp(...)-1.0 was ALWAYS ≤0 (penalty, not
#       reward). Fix: remove -1.0 offset → proper [0,1] reward.
#    2. r_stillness exp kernel saturated to ~0 (avg 0.045). Fix:
#       1/(1+x) rational kernel for usable gradient at all motion levels.
#    3. Survival multiplier sqrt(t/T) was non-standard, distorted gradient
#       signal (early steps undervalued, amplified noise at late steps).
#       Fix: replace with constant r_alive=0.5/step (legged_gym standard).
#    4. Missing r_ang_vel_xy: roll/pitch angular velocity penalty is
#       standard in legged_gym (-0.05). Prevents wobbling.
#    5. r_smooth was L1; papers universally use L2 at -0.01.
#    6. Stand mode gave random small velocities 50% of the time.
#       Fix: stand always zero velocity (vx=vy=wz=0).
#    7. 69% of episodes died at step 50-100 (right after grace period).
#       Fix: lower min_height 0.18→0.15, tilt 45°→60°.
#    8. set_command() didn't update mode grace period, so interactive
#       control had no transition grace. Fix: track mode changes.
#    9. Reward scales were much larger than paper standard (5.0 linvel
#       vs legged_gym 1.0). Fix: align with legged_gym/RMA values.
#
#  Scale calibration sources:
#    - legged_gym (Rudin 2022): smoothness -0.01 L2, ang_vel_xy -0.05,
#      lin_vel_z -2.0, joint_limit -10.0, tracking σ=0.25
#    - RMA (Kumar 2021): alive 0.5, torque -1e-4, action_rate -0.01
#    - Walk These Ways (Margolis 2023): per-mode multipliers, 4096 envs
#    - Solo12/SAC: velocity exp(-4×err²), orientation exp(-3×err²)
#    - Gait-heuristic 2024: walk 0–0.8 m/s, trot 0.8–2.0, gallop 2.0–3.5
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

# ── Base reward scales (apply to all modes) ──────────────────────────
# Aligned with legged_gym (Rudin 2022) and RMA (Kumar 2021) conventions.
# All positive rewards output [0,1] before scaling; penalties are raw squared.
REWARD_SCALES = {
    # Positive rewards — v10: command-proportional, adaptive sigma, tracking penalty
    "r_vel_x":          1.5,       # exp(-vx_err²/σ²) * cmd_scale. Now 0 when cmd≈0
    "r_vel_y":          1.5,       # exp(-vy_err²/σ_y²) * cmd_scale. Now 0 when cmd≈0
    "r_yaw":            2.0,       # exp(-ang_err²/σ²) * cmd_scale. Now 0 when cmd≈0
    "r_gait":           1.0,       # combined: symmetry+clearance+airtime+stride_freq. legged_gym=1.0
    "r_posture":        1.0,       # exp(-joint_err/σ), mode-dependent targets
    "r_body_height":    1.0,       # exp(-(z-h)²/σ²), proper [0,1] reward
    "r_stillness":      1.5,       # 1/(1+motion) rational kernel for stand/crouch
    "r_motion_penalty": -1.5,      # penalty for any motion in stand/crouch
    "r_vel_track_penalty": -2.0,   # v10: explicit penalty for not following velocity commands in walk/run
    "r_fwd_vel":        10.0,      # v11: strong linear velocity bonus (was 1.5; 6.7x increase)
    "r_jump_phase":     3.0,       # jump FSM phase-specific rewards
    "r_alive":          0.5,       # constant per-step survival bonus (RMA=0.5, legged_gym standard)
    # Penalties (raw_value × scale)
    "r_orientation":   -2.0,       # gravity_xy². legged_gym -1 to -5; -2 balances strictness
    "r_torque":        -1e-5,      # Στ². Typical Στ²≈5K-8K → -0.05 to -0.08/step
    "r_smooth":        -0.01,      # |a_t-a_{t-1}|² L2. legged_gym=-0.01 (consensus across papers)
    "r_ang_vel_xy":    -0.05,      # ω_x²+ω_y². legged_gym=-0.05. Prevents roll/pitch wobble
    "r_joint_limit":  -10.0,       # limit_proximity². legged_gym=-10.0 (hard constraint)
    "r_lin_vel_z":     -2.0,       # v_z². legged_gym=-2.0 (was -0.5; too lenient on bounce)
    "r_dof_vel":       -5e-5,      # Σq̇². Halved from -1e-4; was -0.16/step, now ~-0.08
}

# ── Per-mode reward multipliers (Walk These Ways / multi-skill style) ──
# Keys must match REWARD_SCALES. Values multiply the base scale.
# 0.0 = term disabled for that mode; >1.0 = amplified.
MODE_REWARD_MULTIPLIERS = {
    # Stand: zero velocity, maximum stillness, strict orientation
    "stand": {
        "r_vel_x": 0.0,        # disabled: stand means no velocity tracking
        "r_vel_y": 0.0,        # disabled: no velocity tracking when standing
        "r_yaw": 0.0,          # disabled: no yaw tracking when standing
        "r_gait": 0.0,         # disabled: no gait reward when standing
        "r_posture": 3.0,      # high: maintain standing joint angles
        "r_body_height": 2.0,  # high: maintain standing height
        "r_stillness": 3.0,    # high: stay perfectly still
        "r_motion_penalty": 1.0,  # active: penalize any motion
        "r_vel_track_penalty": 0.0,  # disabled: no velocity commands to track
        "r_fwd_vel": 0.0,      # disabled: no forward motion reward for stand
        "r_jump_phase": 0.0,
        "r_orientation": 2.0,  # strict: upright orientation critical for stand
        "r_ang_vel_xy": 2.0,  # strict: no wobbling when standing
        "r_lin_vel_z": 2.0,   # strict: no vertical bounce when standing
    },
    # Walk: v13 — balanced multi-axis tracking. No command-proportional zeroing.
    # Strong tracking penalty penalizes all axes including zero-commanded.
    "walk": {
        "r_vel_x": 3.0,       # v12c: track vx (also penalizes unwanted forward when cmd=0)
        "r_vel_y": 3.0,       # v12c: match vx weight for lateral
        "r_yaw": 3.0,         # v12c: increased from 1.5 (yaw tracking)
        "r_gait": 0.0,
        "r_posture": 1.0,
        "r_body_height": 2.0,
        "r_stillness": 0.0,
        "r_motion_penalty": 0.0,
        "r_vel_track_penalty": 3.0,  # v13: 3x stronger tracking penalty (was 1.5)
        "r_fwd_vel": 0.0,      # v12c: DISABLED — was causing forward-only bias
        "r_jump_phase": 0.0,
        "r_smooth": 0.5,
        "r_ang_vel_xy": 0.0,
        "r_lin_vel_z": 0.0,
        "r_torque": 0.0,
        "r_dof_vel": 0.0,
    },
    # Run: v13 — same balanced tracking as walk.
    "run": {
        "r_vel_x": 3.0,
        "r_vel_y": 2.0,
        "r_yaw": 2.0,
        "r_gait": 0.0,
        "r_posture": 1.0,
        "r_body_height": 2.0,
        "r_stillness": 0.0,
        "r_motion_penalty": 0.0,
        "r_vel_track_penalty": 3.0,  # v13: 3x stronger (was 1.5)
        "r_fwd_vel": 0.0,      # v12c: DISABLED
        "r_jump_phase": 0.0,
        "r_smooth": 0.5,
        "r_lin_vel_z": 0.0,
        "r_ang_vel_xy": 0.0,
        "r_torque": 0.0,
        "r_dof_vel": 0.0,
    },
    # Crouch: low position, stillness, strict orientation
    "crouch": {
        "r_vel_x": 0.0,        # disabled: crouch is stationary
        "r_vel_y": 0.0,        # disabled
        "r_yaw": 0.0,
        "r_gait": 0.0,         # disabled
        "r_posture": 3.0,      # high: maintain crouch joint angles
        "r_body_height": 3.0,  # high: reach and hold crouch height
        "r_stillness": 2.0,    # mostly still
        "r_motion_penalty": 0.5,  # moderate penalty for motion
        "r_vel_track_penalty": 0.0,  # disabled: no velocity commands to track
        "r_fwd_vel": 0.0,      # disabled: no forward motion reward for crouch
        "r_jump_phase": 0.0,
        "r_orientation": 2.0,  # strict: stay level when crouched
        "r_ang_vel_xy": 2.0,  # strict: no wobbling
    },
    # Jump: FSM-driven, relax penalties for dynamic aerial motion
    "jump": {
        "r_vel_x": 0.2,
        "r_vel_y": 0.2,
        "r_yaw": 0.2,
        "r_gait": 0.0,
        "r_posture": 0.5,
        "r_body_height": 1.0,  # tracks FSM phase height
        "r_stillness": 0.0,
        "r_motion_penalty": 0.0,  # disabled
        "r_vel_track_penalty": 0.0,  # disabled
        "r_fwd_vel": 0.0,      # disabled: jump doesn't need forward velocity
        "r_jump_phase": 1.0,   # jump FSM is the main signal
        "r_lin_vel_z": 0.1,   # heavily relax: jumping needs vertical velocity
        "r_ang_vel_xy": 0.5,  # relax wobble for aerial phase
    },
}

# Tracking σ for exp kernel: legged_gym default is 0.25.
TRACKING_SIGMA = 0.25
LATERAL_SIGMA = 0.15   # tighter for lateral precision (smaller commands)

# Posture exp-kernel σ — wider than tracking for smoother gradient
POSTURE_SIGMA = 0.5

# Feet air-time threshold (seconds).
FEET_AIR_TIME_THRESHOLD = 0.15

# No ONLY_POSITIVE_REWARDS. Constant alive bonus and exp-kernel rewards
# keep most transitions net-positive. Removing the clip lets the policy
# learn from negative episodes instead of getting zero gradient.
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

# ── Termination grace periods (legged_gym convention) ─────────────
# Don't terminate early while the robot is settling or adapting to a
# new mode.  This prevents the untrained policy from dying in <30 steps
# and gives the mode-transition dynamics time to stabilise.
TERMINATION_GRACE_STEPS = 100       # 2 s after episode reset (was 50; untrained policy needs
                                     # more time to learn balance before termination kicks in)
MODE_TRANSITION_GRACE_STEPS = 50    # 1 s after mid-episode mode change (was 25;
                                     # 0.5s not enough for crouch→stand transition)

# ── Command re-randomization during training ──────────────────────
# Re-randomize velocity commands every N steps so the policy sees diverse
# commands within a single episode (ref: Walk These Ways, Margolis 2023).
COMMAND_RESAMPLE_INTERVAL = 200  # 4 seconds at 50 Hz

# ── Push perturbations for robustness (legged_gym / RMA convention) ──
PUSH_INTERVAL = 100       # Apply random push every 100 steps (2 seconds)
PUSH_FORCE_MAX = 0.3      # Max velocity impulse (m/s) per axis. Conservative; legged_gym uses 0.5-1.0

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
BODY_HEIGHT_SIGMA = 0.10  # v12: broadened from 0.02 — old value had zero gradient when >5cm from target

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
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
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

        # CPG phase (updated in step(), exposed in obs for phase-dependent residuals)
        self._cpg_phase = 0.0

        # Grace period tracking for termination
        self._last_mode_change_step = 0

        # PD gains — Go1 XML already has damping=2 on joints, so kd=0 here
        # to avoid double-damping. The total damping matches the menagerie
        # position actuator behavior: passive damping only.
        self.kp = 100.0
        self.kd = 0.0
        self.max_torque = 33.5  # Go1: 23.7 Nm hip/thigh, 35.55 Nm knee

        # Domain randomization
        self._base_mass = None
        self._base_friction = None
        self._motor_strength = np.ones(NUM_JOINTS, dtype=np.float32)

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
        self._cpg_phase = 0.0
        self._last_mode_change_step = 0

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
            # v10b: heavily favor locomotion modes to accelerate walk/run discovery
            # Stand/crouch are easy (just be still); walk/run need concentrated experience
            mode_weights = [0.10, 0.40, 0.30, 0.05, 0.15]
            self.command_mode = str(rng.choice(SKILL_MODES, p=mode_weights))
            self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)
            self._randomize_command_for_mode(rng)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # v12: reduce action scale 0.5→0.2 rad so policy makes small corrections
        # to CPG rather than overpowering it. Prevents action-std explosion.
        action_scaled = action * 0.2

        # ── CPG: base trot pattern for walk/run modes (v11) ──────────
        # Without CPG, Gaussian noise on 12 independent joints CANNOT
        # discover coordinated locomotion gaits. The CPG provides a base
        # trotting rhythm; the policy learns residual adjustments on top.
        # In stand/crouch/jump modes, cpg_scale=0 and only policy output acts.
        cpg = np.zeros(NUM_JOINTS, dtype=np.float32)
        if self.command_mode in ("walk", "run"):
            t = self.step_count * self.dt
            freq = 2.0  # Hz (standard trot frequency for Go1-scale robot)
            phase = 2.0 * math.pi * freq * t
            self._cpg_phase = phase  # store for observation
            amp_hip = 0.12   # hip flexion amplitude (rad)
            amp_knee = 0.15  # knee flexion amplitude (rad)
            # Trot: diagonal pairs in anti-phase (FR+RL vs FL+RR)
            # Joint order: [FR_abd, FR_hip, FR_knee, FL_abd, FL_hip, FL_knee,
            #               RR_abd, RR_hip, RR_knee, RL_abd, RL_hip, RL_knee]
            sin_p = math.sin(phase)
            sin_p_pi = math.sin(phase + math.pi)
            cpg[1]  = amp_hip * sin_p       # FR hip
            cpg[2]  = amp_knee * sin_p       # FR knee
            cpg[4]  = amp_hip * sin_p_pi     # FL hip
            cpg[5]  = amp_knee * sin_p_pi    # FL knee
            cpg[7]  = amp_hip * sin_p_pi     # RR hip
            cpg[8]  = amp_knee * sin_p_pi    # RR knee
            cpg[10] = amp_hip * sin_p        # RL hip
            cpg[11] = amp_knee * sin_p       # RL knee

            # v13: Add lateral CPG component — abductor modulation for vy tracking.
            # Without this, lateral motion requires learning from scratch (hard).
            amp_abd = 0.08 * min(abs(float(self.command[1])) / 0.3, 1.0)  # scale with |vy_cmd|
            abd_sign = 1.0 if float(self.command[1]) > 0 else -1.0
            # All four abductors lean in the commanded lateral direction
            cpg[0]  = amp_abd * abd_sign   # FR abductor
            cpg[3]  = amp_abd * abd_sign   # FL abductor
            cpg[6]  = amp_abd * abd_sign   # RR abductor
            cpg[9]  = amp_abd * abd_sign   # RL abductor

            # v13c: Split CPG scaling — sagittal (fwd motion) primarily scales
            # with |vx|, but a base floor ensures leg cycling whenever ANY
            # command axis is active.  Without this floor the robot cannot
            # step sideways for pure lateral commands because the legs don't
            # cycle at all.
            fwd_activity = abs(float(self.command[0])) + abs(float(self.command[2])) * 0.15
            fwd_scale_raw = min(fwd_activity / 0.3, 1.0)
            # Base CPG floor: 30% amplitude when any velocity is commanded
            any_cmd = abs(float(self.command[0])) + abs(float(self.command[1])) + abs(float(self.command[2])) * 0.3
            base_floor = 0.3 if any_cmd > 0.05 else 0.0
            fwd_scale = max(base_floor, fwd_scale_raw)
            for i in [1, 2, 4, 5, 7, 8, 10, 11]:  # sagittal joints (hip/knee)
                cpg[i] *= fwd_scale
            # Abductors already scaled by |vy_cmd| above; also activate for wz
            wz_abd = 0.06 * min(abs(float(self.command[2])) / 0.5, 1.0)
            if abs(float(self.command[2])) > 0.05:
                # Differential drive: left/right lean differently for yaw
                yaw_sign = 1.0 if float(self.command[2]) > 0 else -1.0
                cpg[0]  += wz_abd * yaw_sign   # FR leans right for left turn
                cpg[3]  -= wz_abd * yaw_sign   # FL leans left for left turn
                cpg[6]  += wz_abd * yaw_sign   # RR
                cpg[9]  -= wz_abd * yaw_sign   # RL

        target_q = DEFAULT_STANCE + action_scaled + cpg

        mujoco = self._mj
        q = self.data.qpos[7:7 + NUM_JOINTS]
        qd = self.data.qvel[6:6 + NUM_JOINTS]
        tau = self.kp * (target_q - q) - self.kd * qd
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        # Motor strength randomization: scale torques per-joint (sim-to-real robustness)
        if self.randomize_domain and hasattr(self, '_motor_strength'):
            tau = tau * self._motor_strength

        self.data.ctrl[:NUM_JOINTS] = tau

        # Push perturbation: apply random force to base body periodically
        if self.randomize_domain and self.step_count > 0 and self.step_count % PUSH_INTERVAL == 0:
            rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random
            push_force = rng.uniform(-PUSH_FORCE_MAX, PUSH_FORCE_MAX, size=3)
            push_force[2] = 0.0  # no vertical push
            self.data.qvel[:3] += push_force

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
            # v10b: heavily favor locomotion modes (same weights as reset)
            mode_weights = [0.10, 0.40, 0.30, 0.05, 0.15]
            self.command_mode = str(rng.choice(SKILL_MODES, p=mode_weights))
            self.target_height = HEIGHT_TARGETS.get(self.command_mode, 0.27)
            self._randomize_command_for_mode(rng)
            self._last_mode_change_step = self.step_count

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self._get_info()

    def _randomize_command_for_mode(self, rng):
        """Set velocity commands appropriate for the current command_mode.

        Velocity ranges based on quadruped locomotion literature:
          - Walk: 0.3-0.8 m/s forward (Gait-heuristic 2024)
          - Run:  1.5-3.0 m/s forward (AMP 2021 Go1=3.0, Tencent 3.2 gallop)
          - Stand/Crouch: always zero (no random drift)
        v8: Widened vy and wz ranges to ensure lateral/yaw tracking training.
        """
        mode = self.command_mode
        if mode == "stand":
            vx, vy, wz = 0.0, 0.0, 0.0
        elif mode == "walk":
            # v13: Include zero/negative vx so policy learns to NOT always go forward.
            # Previous [0.2, 0.8] range meant vx_cmd was always positive → forward bias.
            vx = float(rng.uniform(-0.3, 0.8))
            vy = float(rng.uniform(-0.5, 0.5))
            wz = float(rng.uniform(-0.8, 0.8))
        elif mode == "run":
            vx = float(rng.uniform(1.5, 3.0))
            vy = float(rng.uniform(-0.5, 0.5))   # v8: ±0.3 → ±0.5
            wz = float(rng.uniform(-0.5, 0.5))   # v8: ±0.2 → ±0.5
        elif mode == "crouch":
            # Crouch is stationary — zero velocity like stand
            vx, vy, wz = 0.0, 0.0, 0.0
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

        # Base height (critical for crouch/jump/height tracking; legged_gym standard)
        base_height = np.array([float(d.qpos[2])], dtype=np.float32)

        # Foot contact binary (4 dims; standard in legged_gym for gait control)
        foot_contacts = np.zeros(4, dtype=np.float32)
        if d.ncon > 0:
            geom1 = d.contact.geom1[:d.ncon]
            geom2 = d.contact.geom2[:d.ncon]
            for j, fid in enumerate(self._foot_geom_ids):
                foot_contacts[j] = float(np.any((geom1 == fid) | (geom2 == fid)))

        # Skill one-hot encoding (5 dims)
        skill_onehot = np.zeros(SKILL_DIM, dtype=np.float32)
        skill_idx = SKILL_TO_IDX.get(self.command_mode, 0)
        skill_onehot[skill_idx] = 1.0

        # CPG phase signal (enables phase-dependent residual learning)
        # Active in walk/run modes; zero in stand/crouch/jump (no rhythm needed)
        if self.command_mode in ("walk", "run"):
            cpg_phase_signal = np.array([
                math.sin(self._cpg_phase),
                math.cos(self._cpg_phase),
            ], dtype=np.float32)
        else:
            cpg_phase_signal = np.zeros(2, dtype=np.float32)

        obs = np.concatenate([
            qpos, qvel, base_linvel, base_angvel,
            gravity_body, self.prev_action,
            np.append(self.command, self.target_height),
            base_height, foot_contacts,
            skill_onehot,
            cpg_phase_signal,
        ]).astype(np.float32)

        # Manual normalization: divide by fixed scales to get ~[-1, 1] range.
        # Replaces VecNormalize running statistics (which drifted during training
        # and caused catastrophic forgetting of learned locomotion gaits).
        obs = obs / OBS_SCALES

        if self.randomize_domain:
            # Add noise to sensor dims (0:54), NOT to skill encoding (54:59)
            noise = np.zeros(OBS_DIM, dtype=np.float32)
            noise[:54] = self.np_random.standard_normal(54).astype(np.float32) * 0.02
            obs += noise

        return obs

    # ── Reward ──────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray) -> float:
        """Reward v9: adaptive sigma, split vx/vy, stronger yaw, motion penalty."""
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

        # ── 1. Split velocity tracking (v10: command-proportional + adaptive sigma) ──
        # Two fixes combined:
        #   a) Adaptive sigma: prevents gradient vanishing for large commands
        #   b) Command-proportional scaling: prevents FREE reward for zero commands
        #      Without this, walk mode with cmd=[0.5,0,0] gives r_vel_y=3.0/step
        #      and r_yaw=3.0/step for standing still (trivial zero-command match).
        #      This made standing more rewarding than walking.
        vx_error = (base_linvel[0] - vx_cmd) ** 2
        vy_error = (base_linvel[1] - vy_cmd) ** 2
        sigma_vx = max(TRACKING_SIGMA, abs(vx_cmd) * 0.5)
        sigma_vy = max(LATERAL_SIGMA, abs(vy_cmd) * 0.5)
        r_vel_x = math.exp(-vx_error / sigma_vx)
        r_vel_y = math.exp(-vy_error / sigma_vy)

        # v13b: Keep command-proportional scaling for tracking rewards.
        # Without this, zero-cmd axes give free max reward for standing still.
        # The tracking PENALTY (r_vel_track_penalty) handles zero-axis motion.
        CMD_ACTIVE_THRESH = 0.1
        vx_cmd_scale = min(abs(vx_cmd) / CMD_ACTIVE_THRESH, 1.0)
        vy_cmd_scale = min(abs(vy_cmd) / CMD_ACTIVE_THRESH, 1.0)
        wz_cmd_scale = min(abs(wz_cmd) / CMD_ACTIVE_THRESH, 1.0)

        if mode in ("walk", "run"):
            r_vel_x *= vx_cmd_scale
            r_vel_y *= vy_cmd_scale

        # ── 2. Yaw rate tracking (adaptive sigma) ──
        ang_vel_error = (base_angvel[2] - wz_cmd) ** 2
        sigma_wz = max(TRACKING_SIGMA, abs(wz_cmd) * 0.5)
        r_yaw = math.exp(-ang_vel_error / sigma_wz)
        if mode in ("walk", "run"):
            r_yaw *= wz_cmd_scale

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

        # Air time sub-reward (legged_gym formulation)
        air_time_reward = float(np.sum(
            (self._feet_air_time - FEET_AIR_TIME_THRESHOLD) * first_contact
        ))
        if cmd_speed < 0.1:
            air_time_reward = 0.0
        self._feet_air_time[contact_filt] = 0.0

        # Trot symmetry: diagonal pairs should alternate (FR+RL vs FL+RR)
        diag1 = float(contact_filt[0] + contact_filt[3])  # FR + RL
        diag2 = float(contact_filt[1] + contact_filt[2])  # FL + RR
        trot_symmetry = math.sqrt(abs(diag1 - diag2) / 2.0) if cmd_speed > 0.1 else 0.0

        # Foot clearance: reward swing feet reaching target height (8cm)
        foot_heights = self.data.site_xpos[self._foot_site_ids, 2]
        swing_mask = ~contact_filt
        clearance_target = 0.08  # 8cm (legged_gym standard for Go1-scale)
        foot_clearance = 0.0
        if np.any(swing_mask) and cmd_speed > 0.1:
            swing_heights = foot_heights[swing_mask]
            foot_clearance = float(np.mean(
                np.clip(swing_heights, 0, clearance_target) / clearance_target
            ))

        # Stride frequency: reward ~2 touchdowns per step (ideal trot)
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

        # ── 5. Body height tracking (exp kernel, proper [0,1] reward) ──
        # v6 bug: had "-1.0" offset making this always ≤0 (penalty).
        # v7: pure exp kernel like velocity tracking — 1.0 at target, →0 away.
        r_body_height = math.exp(-(base_z - self.target_height) ** 2 / BODY_HEIGHT_SIGMA)
        self._height_history.append(base_z)

        # ── 6. Stillness reward (stand/crouch, rational kernel for gradient) ──
        # v6 bug: exp(-x/σ) saturated to ~0 for any motion (avg=0.045).
        # v7: 1/(1+x) gives usable gradient at all motion levels.
        r_stillness = 0.0
        if mode in ("stand", "crouch"):
            joint_motion = float(np.mean(joint_vel ** 2))     # per-joint average
            body_motion = float(np.sum(base_linvel[:2] ** 2))  # xy linear velocity
            ang_motion = float(np.sum(base_angvel[:2] ** 2))   # roll/pitch angular velocity
            r_stillness = 1.0 / (1.0 + joint_motion * 0.01 + body_motion * 2.0 + ang_motion * 0.5)

        # ── 6b. Motion penalty for stand/crouch (v8: explicit penalty for unwanted motion) ──
        # Complementary to stillness reward — this PENALIZES motion explicitly
        # rather than just rewarding stillness. The penalty raw value is always ≥0.
        r_motion_penalty = 0.0
        if mode in ("stand", "crouch"):
            body_speed_sq = float(np.sum(base_linvel[:2] ** 2))
            yaw_rate_sq = float(base_angvel[2] ** 2)
            r_motion_penalty = body_speed_sq + 0.5 * yaw_rate_sq

        # ── 6c. Linear forward velocity bonus (v10b: locomotion bootstrap) ──
        # The exp kernel gives near-zero reward when far from target (e.g., standing
        # still with cmd=2.0). This linear term provides gradient at ALL distances,
        # rewarding any motion in the commanded direction. Acts as exploration bonus
        # that becomes less important as the policy gets closer to the target.
        r_fwd_vel = 0.0
        if mode in ("walk", "run"):
            # Project actual velocity onto commanded direction
            if abs(vx_cmd) > 0.05:
                r_fwd_vel += float(base_linvel[0]) * (1.0 if vx_cmd > 0 else -1.0)
            if abs(vy_cmd) > 0.05:
                r_fwd_vel += float(base_linvel[1]) * (1.0 if vy_cmd > 0 else -1.0)
            # Clip to prevent reward from velocity overshoot
            r_fwd_vel = max(r_fwd_vel, 0.0)

        # ── 6d. Velocity tracking penalty for walk/run (v13: always active) ──
        # v13: Always penalize ALL velocity axes — no cmd_activity gate.
        # The gate previously meant pure-zero commands got no penalty at all.
        r_vel_track_penalty = 0.0
        if mode in ("walk", "run"):
            r_vel_track_penalty = vx_error + vy_error + ang_vel_error

        # ── 7. Jump FSM phase reward ──
        r_jump_phase = self._advance_jump_fsm(base_z, base_linvel, foot_contacts)

        # ── 8. Alive bonus (constant per step, RMA/legged_gym standard) ──
        r_alive = 1.0  # constant; scaled by REWARD_SCALES["r_alive"] = 0.5

        # ── 9. Orientation penalty (gravity projection) ──
        gravity_body = self._quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        r_orientation = float(np.sum(gravity_body[:2] ** 2))

        # ── 10. Angular velocity xy penalty (roll/pitch rate, legged_gym) ──
        # NEW in v7: prevents wobbling/oscillation. Standard in legged_gym.
        r_ang_vel_xy = float(base_angvel[0] ** 2 + base_angvel[1] ** 2)

        # ── 11. Torque penalty ──
        r_torque = float(np.sum(tau ** 2))

        # ── 12. Action smoothness penalty (L2, legged_gym consensus) ──
        # v6 used L1; all papers (legged_gym, RMA, Solo12) use L2 at -0.01.
        # L2 penalizes large jerks more than small cyclic changes.
        r_smooth = float(np.sum((action - self.prev_action) ** 2))

        # ── 13. Joint limit proximity penalty ──
        jnt_range = self.model.jnt_range[1:NUM_JOINTS + 1]
        margin = 0.1
        below = np.clip(jnt_range[:, 0] + margin - q, 0, None)
        above = np.clip(q - (jnt_range[:, 1] - margin), 0, None)
        r_joint_limit = float(np.sum(below ** 2 + above ** 2))

        # ── 14. Vertical bounce penalty ──
        r_lin_vel_z = float(base_linvel[2] ** 2)

        # ── 15. Joint velocity penalty (anti-shake) ──
        r_dof_vel = float(np.sum(joint_vel ** 2))

        # ── Update tracking state ──
        self._prev_base_linvel = base_linvel.copy()
        self._prev_foot_heights = foot_heights.copy()

        # ── Assemble with mode-dependent reward scales ──
        raw_components = {
            "r_vel_x":       r_vel_x,
            "r_vel_y":       r_vel_y,
            "r_yaw":         r_yaw,
            "r_gait":        r_gait,
            "r_posture":     r_posture,
            "r_body_height": r_body_height,
            "r_stillness":   r_stillness,
            "r_motion_penalty": r_motion_penalty,
            "r_vel_track_penalty": r_vel_track_penalty,
            "r_fwd_vel":     r_fwd_vel,
            "r_jump_phase":  r_jump_phase,
            "r_alive":       r_alive,
            "r_orientation": r_orientation,
            "r_ang_vel_xy":  r_ang_vel_xy,
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

        scaled_components["r_total"] = total
        self._last_reward_components = scaled_components

        return float(total)

    # ── Terminal ────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        # ── Grace periods: don't terminate while settling ──
        # After episode reset: give the untrained policy time to stabilise
        if self.step_count < TERMINATION_GRACE_STEPS:
            return False
        # After mid-episode mode switch: give time to transition
        # Critical for crouch→stand: robot at 0.18m needs time to stand up
        steps_since_mode_change = self.step_count - self._last_mode_change_step
        if steps_since_mode_change < MODE_TRANSITION_GRACE_STEPS:
            return False

        base_z = self.data.qpos[2]
        # v8: Only terminate on ground contact (height < 0.05), not tilt.
        # The orientation penalty (-2.0, mode-amplified) provides sufficient
        # gradient signal to learn upright posture. Tilt-based termination
        # killed 100% of early episodes at the grace boundary, preventing
        # the policy from ever learning balance beyond 100 steps.
        min_height = 0.03 if self.command_mode == "crouch" else 0.05
        return bool(base_z < min_height)

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
        """Advance jump FSM through crouch->launch->airborne->land phases.

        v8 changes: Behavior-based transitions with timer fallbacks.
        Crouch phase transitions when body is actually low enough, not just
        after N steps. Launch rewards scale with upward velocity squared.
        Airborne bonus for all-feet-off-ground state.

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
        n_contacts = int(np.sum(foot_contacts))

        if self._jump_phase == JUMP_PHASE_IDLE:
            # Enter crouch preparation
            self._jump_phase = JUMP_PHASE_CROUCH
            self._jump_step_counter = 0
            self._jump_max_height = base_z
            self.target_height = 0.18

        elif self._jump_phase == JUMP_PHASE_CROUCH:
            self.target_height = 0.18
            self._jump_step_counter += 1
            # Reward lowering body toward crouch height (shaped)
            crouch_depth = max(0.0, 0.27 - base_z)
            r = crouch_depth * 3.0
            # Extra reward for all 4 feet on ground (ready to push)
            if n_contacts >= 4:
                r += 0.3
            # Behavior-based transition: actually crouched, OR timer fallback
            crouched = base_z < 0.22
            if (crouched and self._jump_step_counter >= 5) or self._jump_step_counter >= JUMP_CROUCH_STEPS:
                self._jump_phase = JUMP_PHASE_LAUNCH
                self._jump_step_counter = 0

        elif self._jump_phase == JUMP_PHASE_LAUNCH:
            self.target_height = 0.45
            self._jump_step_counter += 1
            # Reward upward velocity (squared for stronger gradient at high vz)
            vz = float(base_linvel[2])
            r = max(0.0, vz) * 4.0 + max(0.0, vz) ** 2 * 2.0
            # Behavior-based transition: actually going up, OR timer
            if (vz > 0.5 and self._jump_step_counter >= 3) or self._jump_step_counter >= JUMP_LAUNCH_STEPS:
                self._jump_phase = JUMP_PHASE_AIRBORNE
                self._jump_step_counter = 0

        elif self._jump_phase == JUMP_PHASE_AIRBORNE:
            self.target_height = 0.40
            self._jump_step_counter += 1
            self._jump_max_height = max(self._jump_max_height, base_z)
            # Reward height achieved above standing
            r = max(0.0, base_z - 0.27) * 3.0
            # Bonus for all feet off ground (true airborne)
            if n_contacts == 0:
                r += 1.0
            # Transition to landing: descending + feet touching, or timeout
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
        new_mode = mode if mode in SKILL_MODES else "walk"
        # Grant mode-transition grace period when mode actually changes.
        # Without this, interactive control had no grace period on mode
        # switches (e.g. crouch→stand terminated instantly at height 0.18).
        if new_mode != self.command_mode:
            self._last_mode_change_step = self.step_count
        self.command_mode = new_mode
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

        # Motor strength randomization: per-joint scale factor [0.85, 1.15]
        # Simulates actuator degradation and manufacturing variation.
        self._motor_strength = rng.uniform(0.85, 1.15, size=NUM_JOINTS).astype(np.float32)

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
