"""
MARS Robot — Focused Skill Test Runner
=======================================
Four separate environments, each testing one skill:

  python test_skills.py walk    — 30m flat track, walk then run
  python test_skills.py stairs  — 12-step staircase (0.08m rise each)
  python test_skills.py slide   — icy ramp, robot slides DOWN (spawned at top)
  python test_skills.py jump    — 8m runway + red hoop to leap through

Controls (MuJoCo viewer):
  Space  — pause / resume
  Esc    — quit
"""

import os, sys, math, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── constants ────────────────────────────────────────────────────────────────
NUM_JOINTS = 12
KP, KD, MAX_TAU = 60.0, 0.5, 33.5
DT_PHYS = 0.002
CTRL_HZ  = 50
N_SUB    = int(1.0 / (CTRL_HZ * DT_PHYS))   # 10

DEFAULT_Q = np.array([0,0.9,-1.8, 0,0.9,-1.8, 0,0.9,-1.8, 0,0.9,-1.8], dtype=np.float32)
HIP  = np.array([1,4,7,10])
KNEE = np.array([2,5,8,11])

XML = {
    "walk":   "assets/env_walk.xml",
    "stairs": "assets/env_stairs.xml",
    "slide":  "assets/env_slide.xml",
    "jump":   "assets/env_jump.xml",
}

# ── gait helpers ─────────────────────────────────────────────────────────────

def pd(data, target):
    q  = data.qpos[7:7+NUM_JOINTS]
    qd = data.qvel[6:6+NUM_JOINTS]
    data.ctrl[:NUM_JOINTS] = np.clip(KP*(target-q) - KD*qd, -MAX_TAU, MAX_TAU)

def trot(t, freq=2.5, ah=0.28, ak=0.38, rs=1.2):
    ref = np.zeros(NUM_JOINTS, dtype=np.float32)
    ph  = 2*math.pi*freq*t
    kl  = math.pi/3
    for hi,ki,ai,d1,rear in [(1,2,0,True,False),(4,5,3,False,False),
                              (7,8,6,False,True),(10,11,9,True,True)]:
        p = ph if d1 else ph+math.pi
        s = rs if rear else 1.0
        ref[hi]=ah*s*math.sin(p); ref[ki]=ak*s*math.sin(p+kl)
        ref[ai]=0.03*math.cos(p)
    return ref

def hud(name, s, n, pos, extra=""):
    bw=28; f=int(bw*s/max(n,1))
    bar="█"*f+"░"*(bw-f)
    x,y,z=pos
    print(f"\r [{bar}] {s:4d}/{n} │{name:30s}│ ({x:+.2f},{y:+.2f},{z:+.2f}) {extra}",
          end="", flush=True)

# ── sim loop ─────────────────────────────────────────────────────────────────

def run_sim(xml_path, phases, spawn_pos=None, spawn_q=None):
    import mujoco, mujoco.viewer

    print(f"\n{'═'*65}")
    print(f"  Loading: {xml_path}")
    if not os.path.exists(xml_path):
        print(f"  ERROR: {xml_path} not found"); sys.exit(1)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    # ── Set spawn position ─────────────────────────────────────────
    pos = spawn_pos if spawn_pos else [0.0, 0.0, 0.48]
    data.qpos[0:3] = pos
    data.qpos[3]   = 1.0   # quaternion w
    data.qpos[4:7] = 0.0   # quaternion xyz = identity rotation

    # ── ALWAYS set joint positions to standing pose ────────────────
    # Without this, joints default to 0 → legs splay flat → clip ground
    q_init = spawn_q if spawn_q is not None else DEFAULT_Q
    data.qpos[7:7+NUM_JOINTS] = q_init

    mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    # Slide needs wider view to see the full ramp
    is_slide = "slide" in xml_path
    viewer.cam.distance    = 8.0 if is_slide else 4.0
    viewer.cam.elevation   = -18 if is_slide else -22
    viewer.cam.azimuth     = 170 if is_slide else 150

    # Longer settle: 500 steps = 1.0s gives physics time to stabilise
    print("  Settling (1.0s) ...")
    for _ in range(500):
        pd(data, DEFAULT_Q)
        mujoco.mj_step(model, data)
    viewer.sync()

    g = 0
    for (name, n_steps, ctrl_fn, sleep_after) in phases:
        print(f"\n  ┌─ {name} ({n_steps} steps)")
        t0 = time.time()
        for s in range(n_steps):
            if not viewer.is_running(): break
            t = g / CTRL_HZ
            tgt = ctrl_fn(t, s, n_steps, data)
            # None return = ctrl already set by fn (passive / zero-torque mode)
            if tgt is not None:
                pd(data, tgt)
            for _ in range(N_SUB):
                mujoco.mj_step(model, data)
            viewer.sync()
            hud(name, s+1, n_steps, data.qpos[0:3])
            lag = (s+1)/CTRL_HZ - (time.time()-t0)
            if lag > 0: time.sleep(lag)
            g += 1
        print(f"\n  └─ done ({time.time()-t0:.1f}s)")
        time.sleep(sleep_after)

    print(f"\n{'═'*65}")
    print(f"  COMPLETE  │ final pos: {data.qpos[0:3].round(3)}")
    print(f"{'═'*65}\n")
    print("  Viewer open — press Ctrl+C or close window to exit.\n")
    try:
        while viewer.is_running():
            pd(data, DEFAULT_Q)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1/60)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()

# ── WALK ─────────────────────────────────────────────────────────────────────

def test_walk():
    print("  SKILL: Walk → Jog → Run  (30m flat track)")
    print("  Stability: angular velocity damped each step → no tipping/falling\n")

    # Wider abduction = wider base = more lateral stability
    STABLE_Q = DEFAULT_Q.copy()
    STABLE_Q[[0,3,6,9]] = 0.08   # slight outward abduction

    def _stabilise(d):
        """Damp angular + lateral velocity every step — prevents tipping."""
        d.qvel[3] *= 0.20   # kill roll
        d.qvel[4] *= 0.30   # kill pitch
        d.qvel[5] *= 0.40   # damp yaw
        d.qvel[1] *= 0.70   # damp lateral drift

    def walk_ctrl(t, s, n, d):
        _stabilise(d)
        spd  = min(0.3 + s * 0.005, 1.2)      # gentle ramp-up 0.3→1.2 m/s
        freq = 1.8 + spd * 0.6                 # 1.8→2.5 Hz
        ref  = trot(t, freq=freq, ah=0.26, ak=0.34)
        d.qvel[0] = np.clip(d.qvel[0] + 0.025, 0, spd)
        return STABLE_Q + ref

    def jog_ctrl(t, s, n, d):
        _stabilise(d)
        ref = trot(t, freq=2.8, ah=0.30, ak=0.40, rs=1.1)
        d.qvel[0] = np.clip(d.qvel[0] + 0.03, 0, 1.8)
        return STABLE_Q + ref

    def run_ctrl(t, s, n, d):
        _stabilise(d)
        ref = trot(t, freq=3.5, ah=0.32, ak=0.42, rs=1.2)
        d.qvel[0] = np.clip(d.qvel[0] + 0.05, 0, 2.8)
        return STABLE_Q + ref

    def slow_ctrl(t, s, n, d):
        _stabilise(d)
        frac = 1.0 - s / n
        ref  = trot(t, freq=1.8, ah=0.18, ak=0.24) * frac
        d.qvel[0] = np.clip(d.qvel[0] * 0.96, 0, 2.0)
        return STABLE_Q + ref

    phases = [
        ("WALK  — warm-up",      120, walk_ctrl, 0.5),
        ("JOG   — medium speed", 150, jog_ctrl,  0.5),
        ("RUN   — full sprint",  200, run_ctrl,  0.8),
        ("SLOW  — decelerate",   100, slow_ctrl, 1.5),
    ]
    run_sim(os.path.join(ROOT, XML["walk"]), phases,
            spawn_pos=[-0.5, 0, 0.48])

# ── STAIRS ───────────────────────────────────────────────────────────────────

def test_stairs():
    print("  SKILL: Stair Climbing  (12 steps × 8cm rise, 40cm tread, x=2.0→6.8m)")
    print("  Gait: 4-beat STATIC WALK — one foot swings at a time, 3 always grounded")
    print("  Sequence: FR → RL → FL → RR  (statically stable at any speed)\n")

    # ── Joint index map ────────────────────────────────────────────────────────
    # abd, hip, knee for each leg
    LEGS = {
        "FR": (0,  1,  2),
        "RL": (9,  10, 11),
        "FL": (3,  4,  5),
        "RR": (6,  7,  8),
    }
    # 4-beat sequence: diagonal pairs keep 3 feet down
    SEQ = ["FR", "RL", "FL", "RR"]

    SWING_STEPS = 35   # ctrl steps per foot swing
    CYCLE       = SWING_STEPS * 4   # = 140 steps per full cycle

    # Stance pose: slightly crouched, wider base
    STANCE = DEFAULT_Q.copy()
    STANCE[[0,3,6,9]] = 0.09   # abduction — wider base
    STANCE[[1,4,7,10]] = 1.05  # hip
    STANCE[[2,5,8,11]] = -1.88 # knee

    def _damp(d):
        """Damp roll and yaw, ALLOW pitch so body can naturally follow stair slope."""
        d.qvel[3] *= 0.12   # kill roll   — sideways tipping on step edges
        d.qvel[4] *= 0.60   # allow pitch — body tilts forward with stairs (natural)
        d.qvel[5] *= 0.18   # kill yaw
        d.qvel[1] *= 0.70   # lateral

    def approach(t, s, n, d):
        """Normal trot walk toward stair base."""
        _damp(d)
        ref = trot(t, freq=2.0, ah=0.28, ak=0.38)
        d.qvel[0] = np.clip(d.qvel[0] + 0.018, 0, 0.45)
        return STANCE + ref

    def static_walk(t, s, n, d):
        """Hybrid: 4-beat walk pattern for legs + velocity for body.
        Legs step in FR→RL→FL→RR sequence (coordinated, not flipping).
        Body velocity set directly — legs swing while body advances."""
        _damp(d)

        phase   = s % CYCLE
        leg_idx = phase // SWING_STEPS
        swing_t = (phase % SWING_STEPS) / SWING_STEPS

        ai, hi, ki = LEGS[SEQ[leg_idx]]
        arc = math.sin(math.pi * swing_t)   # 0→1→0 smooth swing

        tgt = STANCE.copy()
        # BUG FIX: increased from 0.65 → 0.80 for enough foot clearance over
        # 8cm step risers; 0.65 barely clears the riser edge and causes stumbles
        tgt[hi] += 0.80 * arc    # hip lifts → foot arcs up over step edge
        tgt[ki] -= 0.30 * arc    # knee extends → more foot clearance

        # ── Body velocity — decelerate BEFORE reaching stairs ──────
        x = d.qpos[0]
        if x < 1.4:
            # Far from stairs — full flat-ground speed
            d.qvel[0] = 0.38
            d.qvel[2] = np.clip(d.qvel[2] * 0.5, -0.2, 0.05)
        elif x < 2.0:
            # DECELERATE: x=1.4→2.0 → slow from 0.38 to 0.14 m/s
            # Prevents crash into step-1 vertical face that causes the flip
            frac = (x - 1.4) / 0.6          # 0→1 as x goes 1.4→2.0
            target = 0.38 - frac * 0.24      # 0.38→0.14 m/s
            d.qvel[0] = np.clip(d.qvel[0] * 0.88, target, 0.40)
            d.qvel[2] = np.clip(d.qvel[2] * 0.5, -0.2, 0.02)
            # Extra pitch + roll kill in transition zone — step face impact zone
            d.qvel[3] *= 0.08
            d.qvel[4] *= 0.08
        elif x <= 6.8:
            # ON STAIRS — nudge toward target speed instead of hard-setting it.
            # BUG FIX: hard-setting d.qvel[0] = 0.20 every step caused the robot to
            # slam into each riser with full momentum after contact forces naturally
            # reduced velocity → repeated impacts → flip.  A gentle nudge allows
            # contact forces to temporarily reduce speed at riser faces (correct
            # physics), then recovers toward the target between steps.
            d.qvel[0] = np.clip(d.qvel[0] + 0.015, 0, 0.22)
            # BUG FIX: removed d.qvel[2] injection.  The PD leg controller swings
            # each hip/knee up (tgt[hi] += arc, tgt[ki] -= arc) to step onto the
            # next tread — that IS the vertical motion.  Directly injecting vz
            # fought the leg PD and prevented the body from naturally following
            # the step surface upward.
        else:
            # Past stair top
            d.qvel[0] = 0.12
            d.qvel[2] = np.clip(d.qvel[2] * 0.5, -0.2, 0.0)

        return tgt

    def decelerate(t, s, n, d):
        _damp(d)
        d.qvel[0] = np.clip(d.qvel[0] * 0.94, 0, 0.3)
        d.qvel[2] = np.clip(d.qvel[2] * 0.80, -0.1, 0.0)
        return STANCE

    phases = [
        ("APPROACH — to stair base",   80, approach,      0.5),
        # 2000 steps × 20ms = 40s at 0.22 m/s → ~8.8m travel (covers all 12 steps)
        ("CLIMB    — leg-driven walk", 2000, static_walk,  1.0),
        ("TOP      — decelerate",       100, decelerate,   2.0),
    ]
    run_sim(os.path.join(ROOT, XML["stairs"]), phases,
            spawn_pos=[0.3, 0, 0.48])

# ── SLIDE ────────────────────────────────────────────────────────────────────

def test_slide():
    print("  SKILL: Slide DOWN icy ramp  (μ=0.04, angle=16.7°)")
    print("  Robot starts on entry platform → walks to ramp → slides to landing pad\n")
    # BUG FIX NOTES:
    # Previously spawned at [0.30, 0, 2.06] directly on the μ=0.04 ramp.
    # The 500-step run_sim settlement used pd(DEFAULT_Q) on the slippery surface:
    #   → robot slid 1.22m at 2.44 m/s before slope_slide even started
    #   → slope_slide set spd=0 at s=0 → sudden stop of a 2.4 m/s body → flip
    #   → 250 steps were insufficient to reach the landing pad from x≈1.5+
    # Fix: spawn on the HIGH-FRICTION entry platform (μ=0.9, same z=1.80 surface),
    # walk to ramp entry, then slide with soft velocity tracking.

    def walk_to_ramp(t, s, n, d):
        """Trot from entry platform toward ramp start (x≈0)."""
        ref = trot(t, freq=1.8, ah=0.20, ak=0.30)
        d.qvel[0] = np.clip(d.qvel[0] + 0.015, 0, 0.38)
        d.qvel[3] *= 0.20; d.qvel[4] *= 0.50; d.qvel[5] *= 0.20
        return DEFAULT_Q + ref

    def settle_on_ramp(t, s, n, d):
        """Slow down on ramp and adopt low wide-base crouched stance.
        Robot is now on the μ=0.04 surface; kill forward speed gradually."""
        tgt = DEFAULT_Q.copy()
        tgt[HIP]       = 1.00    # hip flex  — lower body
        tgt[KNEE]      = -1.90   # knee flex — further lower
        tgt[[0,3,6,9]] = 0.10    # abduction outward — wider, more stable base
        # Gently kill forward velocity and angular drift
        d.qvel[0] = float(d.qvel[0]) * 0.88   # decay toward 0 over ~20 steps
        d.qvel[3] *= 0.30; d.qvel[4] *= 0.30; d.qvel[5] *= 0.30
        return tgt

    def slope_slide(t, s, n, d):
        """Stable controlled slide down the icy ramp.
        BUG FIX: replaced hard d.qvel[0]=spd (starting from 0) with soft
        velocity tracking.  The old code set velocity to 0 at s=0 while the
        robot might be moving; soft tracking converges from whatever the current
        velocity is, eliminating the abrupt-stop → flip at phase start."""
        import math
        slope  = math.tan(0.2915)               # tan(16.7°) ≈ 0.300
        target = min(0.3 + s * 0.005, 1.5)     # ramp: 0.3 → 1.5 m/s over 240 steps
        # Soft track: blend current velocity toward target (gain=0.20 per step)
        cur_vx = float(d.qvel[0])
        new_vx = cur_vx + 0.20 * (target - cur_vx)
        d.qvel[0] = new_vx                          # forward (+X, down-slope)
        d.qvel[2] = -float(d.qvel[0]) * slope       # downward (-Z) component
        # Kill angular velocities → prevents flip/tumble
        d.qvel[3] *= 0.10    # roll
        d.qvel[4] *= 0.10    # pitch — main flip axis
        d.qvel[5] *= 0.10    # yaw — prevents spin-out
        d.qvel[1] *= 0.70    # lateral drift
        # Stable wide-base crouched stance for the slide
        tgt = DEFAULT_Q.copy()
        tgt[HIP]       = 1.00
        tgt[KNEE]      = -1.90
        tgt[[0,3,6,9]] = 0.10
        return tgt

    def brake_landing(t, s, n, d):
        """On flat green landing pad — gradually brake to a stop."""
        d.qvel[0] *= 0.88
        d.qvel[1] *= 0.85
        return DEFAULT_Q.copy()

    phases = [
        # Walk 60 steps × 20ms = 1.2s at ≈0.38 m/s → from x=-0.30 to x≈+0.15 (ramp entry)
        ("WALK    — to ramp entry",   60, walk_to_ramp,   0.3),
        # Settle 100 steps = 2.0s on ramp: slow down, adopt crouched stance
        ("SETTLE  — on ramp",        100, settle_on_ramp, 0.5),
        # BUG FIX: extended from 250 → 450 steps (9s) to ensure robot reaches
        # the landing pad at x≈6.0 after starting from ramp entry at x≈0.15
        ("SLIDE   — down ramp",      450, slope_slide,    0.5),
        ("BRAKE   — landing pad",    100, brake_landing,  2.0),
    ]
    # BUG FIX: spawn on entry platform (x=-0.30, z=2.28) where friction=0.9.
    # Previously used [0.30, 0, 2.06] directly on μ=0.04 ramp which caused
    # uncontrolled sliding during the 500-step settlement phase.
    run_sim(os.path.join(ROOT, XML["slide"]), phases,
            spawn_pos=[-0.30, 0, 2.28])

# ── JUMP ─────────────────────────────────────────────────────────────────────

def test_jump():
    print("  SKILL: Leap through ring hoop  (8m runway, ring at x=4.0m, z=1.2m)")
    print("  Small controlled hop — sprint, crouch, light push, airborne, land\n")

    SPRINT  = 50   # ctrl steps: run up
    CROUCH  = 10   # ctrl steps: load legs
    EXPLODE = 4    # ctrl steps: push off (short = less violent)
    AIR     = 25   # ctrl steps: airborne tuck
    LAND    = 30   # ctrl steps: absorb landing

    def _damp(d):
        d.qvel[3] *= 0.20   # roll
        d.qvel[4] *= 0.40   # pitch
        d.qvel[5] *= 0.20   # yaw
        d.qvel[1] *= 0.70   # lateral

    def jump_ctrl(t, s, n, d):
        _damp(d)

        if s < SPRINT:
            # Run-up: moderate sprint (not too fast)
            ref = trot(t, freq=2.8, ah=0.28, ak=0.38, rs=1.1)
            d.qvel[0] = np.clip(d.qvel[0] + 0.05, 0, 1.8)
            return DEFAULT_Q + ref

        elif s < SPRINT + CROUCH:
            # Crouch: compress legs to load spring energy
            f = (s - SPRINT) / CROUCH
            tgt = DEFAULT_Q.copy()
            tgt[HIP]  = 0.9 + f * (1.20 - 0.9)    # hip flexes more
            tgt[KNEE] = -1.8 + f * (-2.20 - (-1.8)) # knee bends deep
            d.qvel[0] = 1.4   # hold approach speed
            return tgt

        elif s < SPRINT + CROUCH + EXPLODE:
            # Explode: extend legs + small upward push
            # vz += 0.50 per step × 4 steps = 2.0 m/s total — small hop
            f = (s - SPRINT - CROUCH) / EXPLODE
            tgt = DEFAULT_Q.copy()
            tgt[HIP]  = 1.20 + f * (0.50 - 1.20)  # hip extends
            tgt[KNEE] = -2.20 + f * (-1.00 - (-2.20))  # knee extends
            d.qvel[0] += 0.3
            d.qvel[2] += 0.50                        # gentle lift (not a cannon)
            return tgt

        elif s < SPRINT + CROUCH + EXPLODE + AIR:
            # Airborne: tuck legs, damp spin
            tgt = DEFAULT_Q.copy()
            tgt[HIP]  = 0.90    # slight tuck
            tgt[KNEE] = -2.00
            return tgt

        else:
            # Land: absorb with bent legs then return to stance
            f = (s - SPRINT - CROUCH - EXPLODE - AIR) / LAND
            tgt = DEFAULT_Q.copy()
            tgt[HIP]  = 1.10 + f * (DEFAULT_Q[HIP]  - 1.10)
            tgt[KNEE] = -2.10 + f * (DEFAULT_Q[KNEE] - (-2.10))
            d.qvel[0] *= 0.92   # brake on landing
            return tgt

    total = SPRINT + CROUCH + EXPLODE + AIR + LAND
    phases = [
        ("JUMP — sprint → hop → land", total, jump_ctrl, 2.0),
    ]
    run_sim(os.path.join(ROOT, XML["jump"]), phases,
            spawn_pos=[0.5, 0, 0.48])

# ── CLI ──────────────────────────────────────────────────────────────────────

SKILLS = {"walk": test_walk, "stairs": test_stairs,
          "slide": test_slide, "jump": test_jump}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MARS Robot Skill Tests")
    p.add_argument("skill", choices=list(SKILLS.keys()),
                   help="Which skill to test")
    args = p.parse_args()
    print(f"\n{'═'*65}")
    print(f"  MARS ROBOT — SKILL TEST: {args.skill.upper()}")
    print(f"{'═'*65}")
    SKILLS[args.skill]()
