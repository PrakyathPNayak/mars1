"""
MARS Robot — Obstacle Course Test
===================================
Tests all locomotion skills with a LIVE MuJoCo simulation window:
  Phase 1 : WALK  — flat ground warm-up
  Phase 2 : CLIMB — staircase approach (slow, high-step gait)
  Phase 3 : CLIMB — ascending 6-step staircase
  Phase 4 : RUN   — sprint across raised ring platform
  Phase 5 : HOOP  — crouch + leap through the circular jump ring
  Phase 6 : LAND  — absorb landing on the green pad
  Phase 7 : WALK  — approach the slide entry platform
  Phase 8 : SLIDE — passive slide down the icy ramp (legs tucked)
  Phase 9 : STOP  — settle on the yellow run-out pad

Course layout (X-axis, world frame):
  -0.5 →  2 m  : flat ground   (spawn + walk warm-up)
   2   →  4 m  : staircase     (6 steps × 8 cm)
   4   →  6 m  : raised platform (top z=0.48)
   6   →  7 m  : jump-through ring hoop (centre z=1.10)
   7   →  9.5m : landing pad
   9.5 → 10 m  : connecting flat
  10   → 13 m  : SLIDE (35°, μ=0.05)
  13   → 15 m  : run-out pad (yellow)

Run:
  python test_obstacle_course.py [--mode all|walk|climb|run|hoop|slide]
                                  [--no-model]
"""

import os
import sys
import math
import time
import argparse
import numpy as np

# ─── project root on path ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ─── constants ───────────────────────────────────────────────────────────────
MODEL_XML = os.path.join(ROOT, "assets", "go1_obstacle_course.xml")
PPO_MODEL = os.path.join(ROOT, "runs", "best_model.zip")

NUM_JOINTS   = 12
KP           = 60.0
KD           = 0.5
MAX_TORQUE   = 33.5
DT           = 0.002           # MuJoCo timestep
CTRL_HZ      = 50              # policy frequency
N_SUBSTEPS   = int(1.0 / (CTRL_HZ * DT))   # = 10 physics steps per policy step
VIEWER_FPS   = 60

DEFAULT_STANCE = np.array([
    0.0,  0.9, -1.8,   # FR
    0.0,  0.9, -1.8,   # FL
    0.0,  0.9, -1.8,   # RR
    0.0,  0.9, -1.8,   # RL
], dtype=np.float32)

HIP_IDX  = np.array([1, 4,  7, 10])
KNEE_IDX = np.array([2, 5,  8, 11])

# Phase definitions  (name, steps, mode-tag)
PHASES = [
    ("WALK  — flat ground",         250, "walk"),
    ("CLIMB — staircase approach",   80, "climb"),
    ("CLIMB — ascending stairs",    200, "climb"),
    ("RUN   — ring platform",       120, "run"),
    ("HOOP  — jump through ring",    93, "hoop"),
    ("LAND  — absorb & settle",     100, "walk"),
    ("WALK  — approach slide top",  200, "walk"),
    ("SLIDE — icy ramp descent",    150, "slide"),
    ("STOP  — run-out settle",       80, "walk"),
]

# ─── helpers ─────────────────────────────────────────────────────────────────

def quat_to_yaw(q):
    """MuJoCo quat [w,x,y,z] → yaw angle."""
    w, x, y, z = q
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))


def pd_control(model, data, target_q, kp=KP, kd=KD, max_tau=MAX_TORQUE):
    """Apply PD torques to match target joint positions."""
    q  = data.qpos[7:7 + NUM_JOINTS]
    qd = data.qvel[6:6 + NUM_JOINTS]
    tau = kp * (target_q - q) - kd * qd
    data.ctrl[:NUM_JOINTS] = np.clip(tau, -max_tau, max_tau)


def walk_reference(t, freq=2.5, amp_hip=0.28, amp_knee=0.38, rear_scale=1.2):
    """Open-loop trot gait reference (12-dim joint offsets)."""
    ref  = np.zeros(NUM_JOINTS, dtype=np.float32)
    phase = 2.0 * math.pi * freq * t
    knee_lead = math.pi / 3.0
    for hip_i, knee_i, abd_i, is_diag1, is_rear in [
        (1,  2,  0, True,  False),
        (4,  5,  3, False, False),
        (7,  8,  6, False, True),
        (10, 11, 9, True,  True),
    ]:
        p = phase if is_diag1 else phase + math.pi
        s = rear_scale if is_rear else 1.0
        ref[hip_i]  = amp_hip  * s * math.sin(p)
        ref[knee_i] = amp_knee * s * math.sin(p + knee_lead)
        ref[abd_i]  = 0.03 * math.cos(p)
    return ref


def stair_reference(t, freq=1.8, amp_hip=0.40, amp_knee=0.55):
    """High-step trot reference for climbing stairs."""
    return walk_reference(t, freq=freq, amp_hip=amp_hip, amp_knee=amp_knee)


def jump_crouch_pose():
    """Deeply crouched pose before jump."""
    q = DEFAULT_STANCE.copy()
    q[HIP_IDX]  = 1.3
    q[KNEE_IDX] = -2.5
    return q


def jump_extend_pose():
    """Fully extended legs — explosive push-off."""
    q = DEFAULT_STANCE.copy()
    q[HIP_IDX]  = 0.3
    q[KNEE_IDX] = -0.95
    return q


def print_hud(phase_name, step, total, pos, mode, extra=""):
    """Terminal HUD line."""
    bar_w  = 30
    filled = int(bar_w * step / max(total, 1))
    bar    = "█" * filled + "░" * (bar_w - filled)
    x, y, z = pos
    print(
        f"\r  [{bar}] {step:4d}/{total}  "
        f"│ Phase: {phase_name:<35s} │ Mode: {mode:<6s} "
        f"│ pos: ({x:+.2f}, {y:+.2f}, {z:+.2f}) {extra}   ",
        end="", flush=True
    )

# ─── PPO-policy wrapper (optional) ───────────────────────────────────────────

def load_ppo_policy(path: str):
    """Try to load the trained SB3 policy; return None on failure."""
    try:
        from stable_baselines3 import PPO
        policy = PPO.load(
            path,
            custom_objects={
                "learning_rate": 0.0003,
                "lr_schedule":   lambda _: 0.0003,
                "clip_range":    lambda _: 0.2,
            },
            device="cpu",
        )
        print(f"  ✓ PPO policy loaded from {path}")
        return policy
    except Exception as e:
        print(f"  ✗ Could not load PPO policy ({e}). Using open-loop gait.")
        return None


def build_obs(data, model, prev_action, command, mode_idx):
    """Build a 45-dim observation slice compatible with the trained model."""
    q   = data.qpos[7:7 + NUM_JOINTS].astype(np.float32)      # 12
    qd  = data.qvel[6:6 + NUM_JOINTS].astype(np.float32)      # 12
    # body-frame linear/angular velocity (approximate from world frame)
    quat = data.qpos[3:7]
    lv_world = data.qvel[0:3].astype(np.float32)
    av_world = data.qvel[3:6].astype(np.float32)
    obs = np.concatenate([
        q  / 1.0,                     # joint pos          [0:12]
        qd / 10.0,                    # joint vel          [12:24]
        lv_world / 2.0,               # base lin vel       [24:27]
        av_world / 5.0,               # base ang vel       [27:30]
        np.array([0.0, 0.0, -1.0]),   # approx gravity     [30:33]
        prev_action,                  # prev action        [33:45]
    ]).astype(np.float32)
    return obs[:45]

# ─── main simulation ─────────────────────────────────────────────────────────

def run_obstacle_course(mode_filter="all", total_steps_override=None,
                        use_ppo=True):
    import mujoco
    import mujoco.viewer

    print("\n" + "═"*70)
    print("  MARS ROBOT — OBSTACLE COURSE SIMULATION")
    print("  Model :", MODEL_XML)
    print("═"*70 + "\n")

    # Load MuJoCo model
    if not os.path.exists(MODEL_XML):
        raise FileNotFoundError(f"MuJoCo XML not found: {MODEL_XML}")
    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML)
    mj_data  = mujoco.MjData(mj_model)

    # Optional PPO policy
    policy    = load_ppo_policy(PPO_MODEL) if use_ppo else None
    prev_act  = np.zeros(NUM_JOINTS, dtype=np.float32)

    # ── Initial pose ──────────────────────────────────────────────────────────
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = [-0.5, 0.0, 0.35]   # start just behind flat zone
    mj_data.qpos[3]   = 1.0                  # quat w
    mj_data.qpos[7:7 + NUM_JOINTS] = DEFAULT_STANCE
    mujoco.mj_forward(mj_model, mj_data)

    # ── Launch viewer ─────────────────────────────────────────────────────────
    print("  ► Opening MuJoCo viewer window ...\n")
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Set nice default camera angle (tracking camera = cam id 0)
    viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    viewer.cam.distance  = 3.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth   = 160

    # Warm-up: settle the robot
    print("  Settling robot (1 s) …")
    for _ in range(500):
        pd_control(mj_model, mj_data, DEFAULT_STANCE)
        mujoco.mj_step(mj_model, mj_data)
    viewer.sync()
    time.sleep(0.5)

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE RUNNER
    # ─────────────────────────────────────────────────────────────────────────
    global_step = 0

    def run_phase(phase_name, n_steps, ctrl_fn, mode_label="WALK",
                  post_sleep=0.6):
        nonlocal global_step, prev_act
        if mode_filter not in ("all", mode_label.lower()):
            return
        print(f"\n  ┌─ Phase: {phase_name}")
        print(f"  │  Steps : {n_steps}  │  Mode: {mode_label}\n")
        t0 = time.time()
        for s in range(n_steps):
            if not viewer.is_running():
                return
            t = global_step / CTRL_HZ
            target_q, act_label = ctrl_fn(t, s, n_steps)
            pd_control(mj_model, mj_data, target_q)
            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            pos = mj_data.qpos[0:3]
            height = pos[2]
            extra = f"h={height:.3f}m"
            print_hud(phase_name, s+1, n_steps, pos, mode_label, extra)
            # Throttle to real time
            elapsed  = time.time() - t0
            expected = (s + 1) / CTRL_HZ
            lag = expected - elapsed
            if lag > 0:
                time.sleep(lag)
            global_step += 1
        print(f"\n  └─ Done  ({time.time()-t0:.1f}s)\n")
        time.sleep(post_sleep)

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 1 : WALK — flat ground
    # ──────────────────────────────────────────────────────────────────────────
    def walk_ctrl(t, s, n):
        ref    = walk_reference(t, freq=2.5, amp_hip=0.28, amp_knee=0.38)
        target = DEFAULT_STANCE + ref
        # Drive forward: push x-velocity
        mj_data.qvel[0] = np.clip(mj_data.qvel[0] + 0.05, 0, 1.5)
        return target, "walk"

    run_phase("WALK — flat ground", 250, walk_ctrl, mode_label="walk")

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 2 : APPROACH STAIRS (slow down, align)
    # ──────────────────────────────────────────────────────────────────────────
    def approach_ctrl(t, s, n):
        ref    = walk_reference(t, freq=1.8, amp_hip=0.32, amp_knee=0.42)
        target = DEFAULT_STANCE + ref
        mj_data.qvel[0] = np.clip(mj_data.qvel[0] + 0.03, 0, 0.9)
        return target, "walk"

    run_phase("CLIMB — approach & slow", 80, approach_ctrl, mode_label="climb")

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 3 : CLIMB STAIRS
    # ──────────────────────────────────────────────────────────────────────────
    def stair_ctrl(t, s, n):
        ref    = stair_reference(t, freq=1.5, amp_hip=0.45, amp_knee=0.58)
        target = DEFAULT_STANCE + ref
        # Slow, steady forward push
        mj_data.qvel[0] = np.clip(mj_data.qvel[0] + 0.02, 0, 0.6)
        return target, "walk"

    run_phase("CLIMB — ascending stairs", 200, stair_ctrl, mode_label="climb")

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 4 : RUN across ring platform
    # ──────────────────────────────────────────────────────────────────────────
    def run_ctrl(t, s, n):
        ref    = walk_reference(t, freq=3.5, amp_hip=0.32, amp_knee=0.44,
                                rear_scale=1.4)
        target = DEFAULT_STANCE + ref
        mj_data.qvel[0] = np.clip(mj_data.qvel[0] + 0.08, 0, 2.5)
        return target, "run"

    run_phase("RUN — ring platform sprint", 120, run_ctrl, mode_label="run")

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 5 : HOOP — approach, crouch through, explosive leap, clear ring
    #  Ring hoop centre: x=6.5, z=1.10, inner radius=0.59m
    #  Robot crouches to ~0.30m body height so it fits under the top arc.
    # ──────────────────────────────────────────────────────────────────────────
    HOOP_APPROACH = 20   # fast walk toward hoop
    HOOP_CROUCH   = 12   # crouch down to clear inner top
    HOOP_EXPLODE  = 8    # leap / burst through
    HOOP_AIR      = 35   # airborne after ring
    HOOP_LAND     = 18   # landing absorption

    def hoop_ctrl(t, s, n):
        if s < HOOP_APPROACH:
            # Fast run toward hoop
            ref    = walk_reference(t, freq=3.2, amp_hip=0.30, amp_knee=0.42,
                                    rear_scale=1.3)
            target = DEFAULT_STANCE + ref
            mj_data.qvel[0] = np.clip(mj_data.qvel[0] + 0.1, 0, 2.8)
        elif s < HOOP_APPROACH + HOOP_CROUCH:
            # Crouch to fit through ring opening (body height → 0.28m)
            frac   = (s - HOOP_APPROACH) / HOOP_CROUCH
            target = DEFAULT_STANCE.copy()
            target[HIP_IDX]  = 0.9 + frac * (1.25 - 0.9)
            target[KNEE_IDX] = -1.8 + frac * (-2.4 - (-1.8))
            mj_data.qvel[0]  = 1.8   # maintain speed through hoop
        elif s < HOOP_APPROACH + HOOP_CROUCH + HOOP_EXPLODE:
            # Explosive jump — clear the ring bottom post
            frac   = (s - HOOP_APPROACH - HOOP_CROUCH) / HOOP_EXPLODE
            target = DEFAULT_STANCE.copy()
            target[HIP_IDX]  = 1.25 + frac * (0.35 - 1.25)
            target[KNEE_IDX] = -2.4 + frac * (-0.95 - (-2.4))
            mj_data.qvel[0] += 1.5
            mj_data.qvel[2] += 2.5   # upward pop to clear landing zone
        elif s < HOOP_APPROACH + HOOP_CROUCH + HOOP_EXPLODE + HOOP_AIR:
            # Airborne — legs tucked
            target = DEFAULT_STANCE.copy()
            target[HIP_IDX]  = 1.0
            target[KNEE_IDX] = -2.0
        else:
            # Landing absorption
            frac   = (s - HOOP_APPROACH - HOOP_CROUCH - HOOP_EXPLODE - HOOP_AIR) / HOOP_LAND
            target = DEFAULT_STANCE.copy()
            target[HIP_IDX]  = 1.0 + frac * (DEFAULT_STANCE[HIP_IDX]  - 1.0)
            target[KNEE_IDX] = -2.0 + frac * (DEFAULT_STANCE[KNEE_IDX] - (-2.0))
        return target, "hoop"

    total_hoop = HOOP_APPROACH + HOOP_CROUCH + HOOP_EXPLODE + HOOP_AIR + HOOP_LAND
    run_phase("HOOP — crouch + leap through ring", total_hoop, hoop_ctrl,
              mode_label="hoop", post_sleep=1.2)

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 6 : LAND & SETTLE on landing pad
    # ──────────────────────────────────────────────────────────────────────────
    def land_ctrl(t, s, n):
        ref    = walk_reference(t, freq=1.5, amp_hip=0.15, amp_knee=0.20)
        target = DEFAULT_STANCE + ref * (1.0 - s/n)  # fade gait out
        mj_data.qvel[0] = np.clip(mj_data.qvel[0] * 0.90, 0, 1.0)
        return target, "walk"

    run_phase("LAND & SETTLE", 100, land_ctrl, mode_label="walk", post_sleep=1.0)

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 7 : WALK → slide entry platform (x≈10, z≈1.68)
    #  Robot needs to walk across mid_ground and up the slide entry step.
    # ──────────────────────────────────────────────────────────────────────────
    def walk_to_slide_ctrl(t, s, n):
        ref    = walk_reference(t, freq=2.2, amp_hip=0.28, amp_knee=0.38)
        target = DEFAULT_STANCE + ref
        mj_data.qvel[0] = np.clip(mj_data.qvel[0] + 0.04, 0, 1.2)
        return target, "walk"

    run_phase("WALK — approach slide entry", 200, walk_to_slide_ctrl,
              mode_label="slide", post_sleep=0.5)

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 8 : SLIDE — robot stands still (legs in stance) and lets the
    #  icy ramp (μ=0.05) drag it passively downward under gravity.
    #  We damp out any active gait so the sliding motion is natural.
    # ──────────────────────────────────────────────────────────────────────────
    def slide_ctrl(t, s, n):
        # Keep legs in a slightly lower stance to lower CoM on the slope
        target = DEFAULT_STANCE.copy()
        target[HIP_IDX]  = 1.05
        target[KNEE_IDX] = -1.95
        # Zero out lateral / yaw velocity to stay centred on ramp
        mj_data.qvel[1] *= 0.80
        mj_data.qvel[5] *= 0.80
        return target, "slide"

    run_phase("SLIDE — icy ramp descent", 150, slide_ctrl,
              mode_label="slide", post_sleep=1.0)

    # ──────────────────────────────────────────────────────────────────────────
    #  PHASE 9 : STOP — decelerate on yellow run-out pad
    # ──────────────────────────────────────────────────────────────────────────
    def stop_ctrl(t, s, n):
        target = DEFAULT_STANCE.copy()
        mj_data.qvel[0] *= 0.85   # bleed off speed
        mj_data.qvel[1] *= 0.85
        return target, "walk"

    run_phase("STOP — run-out pad settle", 80, stop_ctrl,
              mode_label="walk", post_sleep=2.0)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  ✓  OBSTACLE COURSE COMPLETE — ALL PHASES PASSED")
    x, y, z = mj_data.qpos[0:3]
    print(f"  Final robot position : x={x:+.2f}  y={y:+.2f}  z={z:+.2f} m")
    print("  Zones cleared: walk · stairs · ring platform · hoop jump · slide")
    print("═"*70 + "\n")

    print("  Viewer staying open — press Ctrl+C or close the window to exit.\n")
    try:
        while viewer.is_running():
            pd_control(mj_model, mj_data, DEFAULT_STANCE)
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            time.sleep(1.0 / VIEWER_FPS)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("  Viewer closed. Bye!\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MARS Robot Obstacle Course Test"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "walk", "climb", "run", "hoop", "slide"],
        default="all",
        help=(
            "Run only a specific phase group (default: all).\n"
            "  walk  — flat ground walk\n"
            "  climb — staircase ascent\n"
            "  run   — ring platform sprint\n"
            "  hoop  — jump-through ring hoop\n"
            "  slide — slide down icy ramp"
        )
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip loading the PPO model (use open-loop gait)"
    )
    args = parser.parse_args()

    run_obstacle_course(
        mode_filter=args.mode,
        use_ppo=not args.no_model,
    )
