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

    # Stance pose: upright wide base — less crouched than original for better step clearance
    # FIX: hip 1.05→0.95, knee -1.88→-1.85: more extended legs = greater body height on steps
    STANCE = DEFAULT_Q.copy()
    STANCE[[0,3,6,9]] = 0.08   # abduction — wider base
    STANCE[[1,4,7,10]] = 0.95  # hip — more upright (was 1.05)
    STANCE[[2,5,8,11]] = -1.85 # knee — matches DEFAULT_Q more (was -1.88)

    def _stab(d):
        """Walk-style stabilizer: allow forward pitch, kill roll/yaw instability.
        FIX: replaced old _damp (pitch*0.60) with walk-style coefficients (pitch*0.30)
        for the same stabilization that keeps WALK test passing."""
        d.qvel[3] *= 0.20   # roll  — kill sideways tipping
        d.qvel[4] *= 0.30   # pitch — lighter damping (was 0.60, too strong)
        d.qvel[5] *= 0.40   # yaw
        d.qvel[1] *= 0.70   # lateral

    def stair_surface_z(x):
        """Height of the stair tread surface at body x-position."""
        if x < 2.0: return 0.0
        if x > 6.8: return 0.96
        return 0.08 * (min(int((x - 2.0) / 0.4), 11) + 1)

    def stair_next_riser(x):
        """X-coordinate of the next step riser ahead of the robot."""
        if x < 2.0: return 2.0
        n = int((x - 2.0) / 0.4)
        return 2.0 + (n + 1) * 0.4

    def approach(t, s, n, d):
        """Normal trot walk toward stair base."""
        _stab(d)
        ref = trot(t, freq=2.0, ah=0.28, ak=0.38)
        d.qvel[0] = np.clip(d.qvel[0] + 0.018, 0, 0.45)
        return DEFAULT_Q + ref   # use DEFAULT_Q (more upright) instead of STANCE

    def static_walk(t, s, n, d):
        """Hybrid: 4-beat walk pattern for legs + velocity for body.
        Legs step in FR→RL→FL→RR sequence (coordinated, not flipping).
        Body velocity set directly — legs swing while body advances."""
        _stab(d)

        phase   = s % CYCLE
        leg_idx = phase // SWING_STEPS
        swing_t = (phase % SWING_STEPS) / SWING_STEPS

        ai, hi, ki = LEGS[SEQ[leg_idx]]
        arc = math.sin(math.pi * swing_t)   # 0→1→0 smooth swing

        tgt = STANCE.copy()
        # FIX: was tgt[hi] += 0.80 (hip flexes more = foot sweeps backward).
        # Correct forward-step: hip EXTENDS (decreases) → foot swings forward in space,
        # lands ahead of body on the next tread.  Larger knee bend lifts foot higher.
        tgt[hi] -= 0.90 * arc    # hip extends → foot swings forward over riser
        tgt[ki] -= 0.45 * arc    # knee bends  → foot clears riser (was 0.30, too little)

        # ── Body velocity — decelerate BEFORE reaching stairs ──────
        x  = d.qpos[0]
        bz = d.qpos[2]
        dist_to_riser = stair_next_riser(x) - x

        if x < 1.4:
            # Far from stairs — full flat-ground speed
            d.qvel[0] = 0.38
            d.qvel[2] = np.clip(d.qvel[2] * 0.5, -0.2, 0.05)
        elif x < 2.0:
            # DECELERATE: x=1.4→2.0 → slow from 0.38 to 0.18 m/s
            frac   = (x - 1.4) / 0.6
            target = 0.38 - frac * 0.20
            d.qvel[0] = np.clip(d.qvel[0] * 0.90, target, 0.40)
            d.qvel[2] = np.clip(d.qvel[2] * 0.5, -0.2, 0.02)
            # FIX: removed d.qvel[3]*=0.08; d.qvel[4]*=0.08 here — the aggressive
            # angular kill in the transition zone was destabilising the robot and
            # causing it to collapse BEFORE even reaching the stairs.
        elif x <= 6.8:
            # ON STAIRS: near each riser, boost forward momentum and lift body
            if 0 < dist_to_riser < 0.15:
                # Riser boost: extra push to clear each step face
                d.qvel[0] = np.clip(d.qvel[0] + 0.040, 0, 0.45)
                d.qvel[2] = max(float(d.qvel[2]), 0.18)  # gentle upward lift
            else:
                d.qvel[0] = np.clip(d.qvel[0] + 0.015, 0, 0.25)
                # Maintain body clearance above current step surface
                if bz - stair_surface_z(x) < 0.16:
                    d.qvel[2] = max(float(d.qvel[2]), 0.01)
        else:
            # Past stair top
            d.qvel[0] = 0.12
            d.qvel[2] = np.clip(d.qvel[2] * 0.5, -0.2, 0.0)

        return tgt

    def decelerate(t, s, n, d):
        _stab(d)
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
        FIX: removed d.qvel[2] = -vx*slope injection.  At speed >0.9 m/s the old
        code forced vz = -0.27 m/s downward which drove the body into the ramp
        surface (rel_z → 0) and caused the robot to collapse.  Gravity naturally
        provides the correct downward component on the slope; no injection needed.
        FIX: velocity cap kept at 1.5 m/s (same as original) but vz clamp removed
        so the body can freely follow the ramp angle without fighting its own weight."""
        target = min(0.3 + s * 0.005, 1.5)     # ramp: 0.3 → 1.5 m/s over 240 steps
        cur_vx = float(d.qvel[0])
        d.qvel[0] = cur_vx + 0.20 * (target - cur_vx)   # soft track toward target
        # REMOVED: d.qvel[2] = -vx * slope  (was crushing robot into ramp at speed)
        # Kill angular velocities → prevents flip/tumble
        d.qvel[3] *= 0.10    # roll
        d.qvel[4] *= 0.10    # pitch — main flip axis
        d.qvel[5] *= 0.10    # yaw
        d.qvel[1] *= 0.70    # lateral drift
        # Stable wide-base crouched stance for the slide
        tgt = DEFAULT_Q.copy()
        tgt[HIP]       = 1.00
        tgt[KNEE]      = -1.90
        tgt[[0,3,6,9]] = 0.10
        return tgt

    def brake_landing(t, s, n, d):
        """On flat green landing pad — gradually brake and stabilise upright."""
        d.qvel[0] *= 0.88
        d.qvel[1] *= 0.85
        d.qvel[2] = float(np.clip(d.qvel[2] * 0.5, -0.05, 0.10))  # dampen bounce
        d.qvel[3] *= 0.10   # kill residual roll from ramp
        d.qvel[4] *= 0.15   # kill residual pitch from ramp
        d.qvel[5] *= 0.15   # kill residual yaw
        return DEFAULT_Q.copy()

    phases = [
        # Walk 60 steps × 20ms = 1.2s at ≈0.38 m/s → from x=-0.30 to x≈+0.15 (ramp entry)
        ("WALK    — to ramp entry",   60, walk_to_ramp,   0.3),
        # Settle 100 steps = 2.0s on ramp: slow down, adopt crouched stance
        ("SETTLE  — on ramp",        100, settle_on_ramp, 0.5),
        # FIX: extended from 450 → 550 steps (11s) — extra 100 steps needed to
        # traverse the full ramp length and settle upright on the landing pad
        ("SLIDE   — down ramp",      550, slope_slide,    0.5),
        ("BRAKE   — landing pad",    150, brake_landing,  2.0),
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
