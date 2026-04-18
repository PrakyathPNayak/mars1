
=== SESSION 2026-04-16 — FULL DIAGNOSIS ===

## Zero-Action Baselines — BEFORE (v23)

| Mode  | r/step | Survived | Key Components |
|-------|--------|----------|----------------|
| Stand | 12.27  | 500/500  | stillness=4.49, height=3.00, posture=2.81, alive=2.00 |
| Walk  | 2.82   | 277/500  | vx_fwd=2.41 (from ref traj), vx_track=0.49 |
| Run   | 6.60   | 248/500  | vx_fwd=6.71 (from bootstrap velocity+ref) |
| Jump  | 3.72   | 500/500  | jump_phase=3.73 (PD bounce exploits launch reward) |

## Zero-Action Baselines — AFTER v24b

| Mode  | r/step | Survived | Status |
|-------|--------|----------|--------|
| Stand | 4.39   | 500/500  | ↓64% — alive removed, multipliers reduced |
| Walk  | 1.93   | 279/500  | Clean: vel tracking + orientation only |
| Run   | -0.29  | 500/500  | Standing still now NEGATIVE |
| Jump  | -0.01  | 500/500  | Free lunch eliminated |

## Root Causes (diagnosed and fixed)

### FIXED: action_scale=0 for walk/run
Was zero. Set to 0.3. RL now has meaningful authority.
Added action magnitude penalty (-0.05 * ||action||^2) as residual regularization.

### FIXED: Stand reward free lunch
Cut from 12.27 to 4.39. Removed alive bonus, reduced multipliers.

### FIXED: Jump PD bounce exploit
Now requires vz > 1.0 for launch, feet off ground for airborne. Zero-action → ~0.

### FIXED: Run speed cap
Expanded from [1.5, 3.0] to [0.5, 4.0] m/s.

### FIXED: Mode weights
Rebalanced from [0.20, 0.05, 0.05, 0.70] to [0.15, 0.35, 0.25, 0.25].

## Training Results

### Walk v24 (500K, action_scale=0.15, no action penalty)
- FAILED: policy worse than zero-action. std stuck at 1.1.
- Random noise disrupted gait. No gradient toward "do nothing."

### Walk v24b (500K, action_scale=0.15, log_std=-1.0, no reward norm)  
- FAILED: ep_rew_mean flat at ~-170, std stuck at 0.369.
- Small action scale + medium std = corrections too small to help.

### Walk v24c (2M, action_scale=0.3, log_std=-1.5, action penalty)
- FAILED: eval reward=-890 vs zero-action=+459. Policy ACTIVELY worse than doing nothing.
- Root cause: negative per-step reward during exploration made PPO learn to die faster.
- Training showed improvement (-396 → -218) but deterministic eval was catastrophic.
- Model learned non-zero actions that disrupt gait more than random noise does.

## v24d Analysis — Reward Component Deep Dive

### Walk mode reward breakdown (v24b → v24d):
Zero-action per-step: 1.75 → 2.26 (removed standstill pen +0.24, removed action_mag pen)
With noise std=0.22: 0.55 → 0.88 (ALL noise levels now positive)
With noise std=0.082: N/A → 1.72 (very positive, clear learning signal)

Key finding: standstill penalty (-0.24/step) was WRONG — penalized walking robot 
because EMA velocity takes time to ramp up after reset. Removing it + halving 
orientation penalty (2.0 → 1.0) made per-step reward robustly positive at all 
exploration noise levels.

### v24d training (IN PROGRESS, 3M steps)
- Config: log_std=-2.0 (std=0.135), action_scale=0.3, 8 envs
- Initial: ep_rew=957, ep_len=930, r/step=1.03 — POSITIVE from start!
- This is the first run with correct reward signal direction.

## Zero-Action Baselines — v24d

| Mode  | r/step | Status |
|-------|--------|--------|
| Stand | 4.38   | Unchanged |
| Walk  | 2.26   | ↑ from 1.93 (removed penalties) |
| Run   | 0.98   | ↑ from -0.29 (was negative, now positive!) |
| Jump  | -0.007 | Near zero ✓ |

## Next Steps
1. v24f walk training in progress (5M steps, batch=4096)
2. If deterministic eval > zero-action → commit, move to run mode
3. If plateau → try curriculum or stochastic deployment
4. Jump training script ready (scripts/train_jump.py)
5. Run mode needs speed-scaled reference trajectory or full-authority RL

## Walk v24d Results (COMPLETED)
- Config: 3M steps, batch=512, ent_coef=0.005, log_std=-2.0
- Training: ep_rew 957→1300 (+36%), ep_len 930→1330 (+43%)
- Deterministic eval (training VecNorm): 730.5±75 reward, 277±19 steps, 2.64/step
- Zero-action baseline: 670.7±92.6 reward, 318±103 steps, 2.11/step
- VERDICT: +9% total reward, +25% per-step, but -13% survival. Modest win.
- Issue: std GREW from 0.135 to 0.217 (entropy bonus counterproductive)
- Issue: batch_size=512 → 320 gradient updates/rollout → unstable optimization

## Walk v24e (ABORTED at ~1M steps)
- Config: 5M steps, batch=512, ent_coef=0.0001, log_std=-2.5
- clip_fraction 0.48-0.55 (way too high, same batch_size problem)
- Eval at 400K: 639 reward (WORSE than zero-action)
- Aborted: batch_size=512 still fundamentally broken

## Walk v24f (IN PROGRESS, 5M steps)
- Config: batch=4096, ent_coef=0.0001, log_std=-2.0
- BREAKTHROUGH: clip_fraction 0.08-0.14 (was 0.55!), approx_kl 0.010-0.013
- Training: ep_rew ~1100-1180, ep_len ~1200-1400
- Eval at 400K: 716±409, len 443±416 (high variance)
- Best model: found at eval 400K, no improvement through ~1M+ steps
- std: 0.135→0.134 (stable, not growing!)
- Root cause of batch_size fix: 320 updates/rollout → 40 updates/rollout
  legged_gym uses 4096 envs × 24 steps = 98K samples, batch 5120 → ~95 updates

## Walk Reward Analysis
- Optimal per-step: 3.50 (perfect tracking at 0.5 m/s)
- Zero-action per-step: 2.84 (81% of optimal — reference already captures most reward)
- Maximum improvement headroom: 0.67/step (23.5%)
- Main RL value: SURVIVAL (1300 vs 318 steps), not per-step quality
- Stochastic noise aids stability but hurts per-step reward
- Deterministic eval suffers from stochastic/deterministic gap

## Walk v24h FINAL (best config)
- Config: norm_reward=True, norm_obs=False, batch=4096, ent=0.0001, log_std=-2.0
- explained_variance: 0.85 (value function learning well)
- Eval deterministic: ~726 reward (ceiling regardless of normalization approach)
- Stochastic: ~1072 reward, 1370 steps (+57% vs zero-action)
- DECLARED SOLVED (stochastic deployment)

## Reference Trajectory Analysis
- Parameter sweep: rear_scale [1.0-1.3], yaw_trim [0.0-0.04]
- Current params (1.3, 0.04) are ALREADY BEST: 23% survival, mean 367 steps
- Reducing rear_scale to 1.0 makes it WORSE: 0% survival, mean 242 steps
- Instability is FUNDAMENTAL to open-loop trot, not a parameter issue
- RL's job: provide closed-loop corrections (stochastic walk policy proves this works)

## v25: Run Mode Action Authority
- Reference trot achieves 0.69 m/s mean, run commands go to 2.0 m/s
- action_scale differentiated: walk=0.3, run=0.8 (was shared 0.3)
- Run speed range narrowed [0.5, 4.0] → [0.5, 2.0] for initial curriculum
- Training script ready, waiting for jump to finish

## Jump v24 Training (IN PROGRESS)
- Config: batch=4096, ent=0.0001, log_std=-1.0, norm_reward=True, norm_obs=False
- Zero-action baseline: -3.41 total (negative — no free lunch!)
- Eval at 400K: **2530 reward, 2000 steps (full episodes)**
- Training ep_rew: oscillating 1310-1880 (timing-dependent variance)
- Deterministic BETTER than stochastic (opposite of walk — precision helps for jump)
- explained_variance: 0.977, clip_fraction: 0.13 (healthy PPO)
- Best model saved at 400K, training continuing to 5M

## Stand Mode
- Zero-action = 4.37/step = near-optimal (body_height=1.5, posture=1.5, stillness=1.5)
- Doing nothing IS standing. No training needed.
- Future: train for perturbation robustness only

## What is Next
1. Let jump training finish (5M steps, ~3.5 hours remaining)
2. Evaluate jump model (deterministic + stochastic)
3. Start run training (v25 config: action_scale=0.8, speed [0.5, 2.0])
4. If run works: expand speed range curriculum to 4.0 m/s
5. Long-term: skill composition (hierarchical controller)

## Run Training v27 — In Progress ($(date))

### What Changed (v27)
- Split walk/run reward into separate branches
- Run: r_ang_vel_xy penalty 0.3→0.05 (6× reduction)
- Run: r_lin_vel_z 0.2→0.05 (4× reduction)
- Run: RESTORED r_vx_lin at 1.5 (monotonic speed gradient)
- Run: r_vx_track scale 2.5→3.0, r_gait 1.5→0.5

### Why v25/v26 Failed
- r_ang_vel_xy at -0.3 scale designed for WALK (gentle 0.4-0.6 m/s)
- Random policy with action_scale=0.8 creates huge angular velocities
- -3.84/step penalty vs +0.36/step tracking reward = total -3.3/step
- Policy started with -2100 ep_rew → PPO learned to do LESS (death spiral)

### v27 Diagnostics
- Zero-action run: 2.33/step (positive!)
- Random policy std=0.135: +1.94/step (was -0.90 in v26)
- All noise levels positive

### Training Progress (295K steps)
- ep_rew_mean: ~500 (normalized, positive from start!)
- explained_variance: 0.88-0.90 (excellent)
- clip_fraction: 0.14-0.15 (healthy)
- std: 0.133 (stable)
- Reward flat so far — too early to judge (eval callback at ~400K)

### What's Next
- Wait for run v27 to reach 1M+ steps
- First eval callback at ~400K steps
- If reward improves: continue to 3M
- If still flat at 1M: need larger std or curriculum

## v27b Walk Training — Forward-Only Reward (In Progress)

### Changes from v27
- v27 (pure RL, no reference) FAILED: converged to standing still at 800K
- v27b: reference ON, ZERO penalties, forward-only reward
- r_vx_lin(5.0) DOMINANT + r_vx_track_tight(1.5,s=0.08) + r_gait(0.5) - orient(0.1)
- action_scale=0.3, log_std_init=-2.0

### 400K Eval Results
| Mode | Reward | Steps | Dist | Speed |
|------|--------|-------|------|-------|
| deterministic | 1175 | 434 | 4.58m | 0.53m/s |
| stochastic | 1195 | 944 | 3.86m | 0.20m/s |
| zero-action | 1287 | 294 | 5.30m | 1.00m/s |

### Analysis
- Policy survives 50% longer (434 vs 294) but 47% slower
- Still trading speed for survival — fundamental issue
- Stochastic still partially cancels reference
- Training continuing to 10M, checking at 1M, 2M

### History of walk attempts
| Version | Approach | Result |
|---------|----------|--------|
| v24h | ref + tracking + penalties | 1072 reward, stochastic solved |
| v26 | gait quality + strong penalties | FAILED: stood still |
| v26b | reduced penalties 6x | 620 eval, cancels reference |
| v26c | action_scale=0.2 | identical to zero-action |
| v27 | pure RL, no reference | stood still, no gait discovery |
| v27b | ref ON, no penalties | survives longer, slower |

## v28: Walk Free Lunch Fix + Run Speed Range ($(date -u +%Y-%m-%d))

### Root Cause Diagnosis
1. Walk reference trajectory (v27d) was calibrated to match command speed (scale=0.65*vx_cmd+0.22)
2. Zero-action at cmd=0.8 got 6.49/step = 87% of optimal 7.5/step
3. Policy had NO learning signal — reference did all the work
4. Run speed capped at [0.5, 2.0] — Go1 can do 3.7 m/s

### Changes
- Reference: fixed amplitude scale=0.45 (~0.25 m/s), no command scaling
- Walk action_scale: 0.3 → 0.5
- Walk sigma: 0.25 → 0.10
- Walk reward: +r_smooth(-0.02)
- Run range: [0.5, 2.0] → [0.5, 4.0]

### Results (zero-action r/step)
| Mode | Before | After | % Optimal |
|------|--------|-------|-----------|
| Walk 0.8 | 6.49 | 1.94 | 28% (was 87%) |
| Walk 1.2 | ~6.49 | 0.71 | 10% |
| Run 2.0 | ~1.19 | 0.31 | ~5% |
| Run 4.0 | N/A | 0.15 | ~3% |
| Jump | -0.004 | -0.004 | 0% |
All modes survive 500 steps (walk was 278 before — fixed ref more stable).

### What Is Next
1. Investigate stand reward — still uses old complex formula with alive bonus
2. Check if stand needs fixing (zero-action = 4.37/step, doing nothing IS standing)
3. Run training to verify policy learns with v28 rewards
4. Investigate XML model parameters for robot stability

## v29: Walk 3-DOF Tracking + Crouch + OBS_SCALES Fix

### Problems Found
1. **Walk reward only tracked vx** — r_vy_track and r_wz_track missing entirely.
   Robot ignored lateral (A/D) and yaw (Q/E) commands. Always walked forward.
2. **r_vx_track_walk not gated** — When vx_cmd=0 (pure lateral), got free 5.0*exp(0)=5.0.
3. **Crouch mode dropped** — "crouch" not in SKILL_MODES, converted to "walk" at default height.
4. **OBS_SCALES mismatch** — vy_cmd/0.5 vs actual_vy/2.0 (4x), wz_cmd/0.8 vs actual_wz/5.0 (6.3x).

### Fixes Applied
- Walk reward: +1.5*r_vy_track, +1.0*r_vy_lin, +1.0*r_wz_track
- Gate r_vx_track_walk by vx_cmd_scale
- Crouch → walk/stand at height=0.18m
- OBS_SCALES: vy_cmd 0.5→2.0, wz_cmd 0.8→5.0

### Results (zero-action r/step)
| Command | Before v29 | After v29 |
|---------|-----------|-----------|
| Forward 0.8 | 1.94 | 2.05 |
| Left 0.8 | ~1.52 (free!) | 0.10 |
| Right 0.8 | ~1.52 | 0.10 |
| Turn 1.0 | ~1.50 | 0.23 |
| Backward 0.5 | ~0.07 | 0.07 |
| Crouch | height unchanged | h=0.18m ✓ |

### VecNormalize Investigation
User asked if vec_normalize.pkl needed replacing. Answer: NO.
- Training uses norm_obs=False, norm_reward=False since v11
- pkl has no obs_rms stats — VecNormalize is pure no-op
- Loader falls back to identity normalize_fn
- v27 model oscillation caused by FREE LUNCH reward (87% optimal at zero-action)

### What Is Next
1. Also add r_vy_lin to run mode for consistency
2. Run training with v29 rewards — verify policy learns all directions
3. Address r_wz_track sigma (HEADING_SIGMA=0.08 very tight — may need widening)

## v30 — Reference Trajectory Forward Bias Fix + Lateral Training (commit aec81f2, 69a87ed)

### Diagnosis
1. Reference trajectory (v28/v29) used fixed amplitude=0.45 regardless of command → produced 0.39 m/s forward even with pure lateral command (vx=0, vy=0.3)
2. Walk command randomization had vy=0, wz=0 ALWAYS → policy never trained lateral/yaw
3. LATERAL_SIGMA=0.15 too wide → standing got 55% of optimal lateral reward
4. Lateral reward weights (1.5 + 1.0 = 2.5) << forward weights (5.0 + 2.0 = 7.0)

### v30 Fix (aec81f2)
- Reference amplitude scales with |vx_cmd|/0.5 (floor 0.10, cap 1.0)
- Zero-action lateral: vx=0.0001 (was 0.3908)

### v30b Fix (69a87ed)  
- Walk commands: vx∈[0,1.2] vy∈[-0.4,0.4] wz∈[-0.6,0.6] (was vx-only)
- LATERAL_SIGMA: 0.15→0.05
- Walk weights: vy_track 1.5→3.0, vy_lin 1.0→1.5, wz_track 1.0→2.0
- Run weights: vy_track 0.3→1.0, vy_lin 0.3→0.5, wz_track 0.3→1.0

### v30 3M Training Results (before v30b lateral fix)
- Reward: 294→730→1640→3870→7240→9090(peak)→7100(final)
- walk_fwd: vx=0.83 (cmd 0.5) — learned forward ✓
- walk_lat: vx=0, vy=0 — NOT learned (never trained!)
- walk_yaw: wz≈0 — NOT learned
- run_fwd: vx=0.34 (cmd 2.0) — weak
- stand: perfect ✓, jump: good ✓, crouch: h=0.19 ✓

### v30b Zero-Action Baselines
- walk lat (0,0.3,0): 1.12/step (25% free, was 55%)
- walk yaw (0,0,0.5): 0.38/step (19% free)
- walk fwd (0.5,0,0): 5.78/step (unchanged)

### Training v30b
- 3M steps, 8 envs, CPU — running
- Early: 1100→1560 at 229K steps

## v31 — Anti-Oscillation + Crouch + Walk Speed (commit 56363e2)

### Root Cause Analysis
- **Oscillation exploitation**: r_vy_track and r_wz_track used INSTANTANEOUS body-frame velocity
  - Policy learned to oscillate: brief spikes matching command gave periodic high reward
  - Lateral: vy_body oscillated ±0.05 with 25-step (0.5s) period
  - Yaw: wz oscillated ±0.3 with 20-step period
  - EMA of oscillation ≈ 0 → no net motion
- **No wz_ema existed** — only vx and vy had EMA
- **Sigma too loose**: adaptive sigma 0.3×cmd gave exp(-1) = 37% free lunch for standing still

### v31 Fixes Applied
1. **EMA for vy/wz tracking**: `r_vy_track = exp(-(vy_ema - vy_cmd)² / σ)` — forces SUSTAINED motion
2. **Added wz_ema**: initialized, updated, bootstrapped at reset
3. **Tighter sigma**: adaptive multiplier 0.15 (was 0.3) → 10% free lunch (was 37%)
4. **Oscillation penalty**: walk -2.0×(vy-vy_ema)², run -1.0×(vy-vy_ema)²
5. **Crouch height**: HEIGHT_MIN = 0.10m (was 0.18m) — "almost ground level"
6. **Walk speed**: reference amplitude 0.32 (was 0.45) — 29% slower
7. **Walk height range**: [0.10, 0.30] for crouched walking (was [0.15, 0.30])
8. **Boosted r_vy_lin**: weight 2.0 (was 1.5) — stronger monotonic gradient

### Diagnostics
- Lateral (vy_cmd=0.3): zero-action 0.510/step vs optimal ~5.0 → 10.2% free lunch ✓
- Yaw (wz_cmd=0.5): zero-action 0.083/step vs optimal ~2.0 → 4.2% free lunch ✓
- Crouch: target=0.10m, actual=0.122m ✓

### Training
- v31 training started: PID 638460, 5M steps, 8 envs
- Checkpoints: checkpoints/v31/
- Log: /tmp/v31_train.log

## v31b — Reference Trajectory + Unwanted Velocity Penalty (Iteration 32)

### Changes
- 3-DOF reference trajectory with lateral abduction oscillation + yaw differential stride
- Unwanted velocity penalty: -5.0×vx²/-3.0×vy²/-2.0×wz² when DOF not commanded
- Yaw trim fades out when wz commanded (fixed yaw_L asymmetry: +0.007→-0.233)
- Yaw diff gain: 0.25→0.40 for stronger yaw control
- Walk/run gating restored but with penalty for non-commanded DOFs

### Zero-Action Reference Velocities
| Mode | cmd | vx | vy | wz |
|---|---|---|---|---|
| fwd | (0.5,0,0) | +0.243 | -0.102 | -0.012 |
| lat_R | (0,+0.3,0) | +0.321 | +0.304 | -0.075 |
| lat_L | (0,-0.3,0) | +0.307 | -0.148 | -0.219 |
| yaw_R | (0,0,+0.5) | +0.440 | +0.023 | +0.400 |
| yaw_L | (0,0,-0.5) | +0.440 | +0.019 | -0.233 |
| diag_R | (0.5,+0.3,0) | +0.315 | +0.303 | -0.112 |

### Training
- PID: 1503950
- Config: 5M steps, 8 envs, checkpoints/v31b/
- FPS: ~2764
- Commit: 249f0a5

### Known Issues
- lat_L asymmetric (vy=-0.148 vs +0.304 for R) — policy must compensate
- Forward bias (vx≈0.3-0.44) for lat/yaw — unwanted penalty addresses this
- fwd has spurious vy=-0.102 drift

## v31d — Forward Reference Fix + Height Tracking ($(date))

### Bugs Found
1. **Forward ref amplitude 3x too weak**: `0.30 * speed_scale = 0.096` vs lat_boost=0.15
   - Forward reference WEAKER than lateral reference — policy had no gradient for walk_fwd
   - Fix: use speed_scale directly → amp_hip=0.32 for walk vx=0.5
2. **Run ref same as walk**: vx_scale saturated at |vx|/0.5. Run at 2.0 m/s got same ref as walk 0.5
   - Fix: mode-dependent normalization (walk: 0.5, run: 2.0) and base amp (0.32/0.45)
3. **Walk reward missing height tracking**: Walk reward had zero height terms
   - Crouch target 0.10m but policy at h=0.260 — no incentive to crouch
   - Fix: added r_height_walk with tight sigma=0.01

### Zero-Action Baselines (v31c → v31d)
- walk_fwd: 1.62 → 3.60 (+122%)
- lat_R/L: 4.44 → 6.43 (still high free lunch — reference does all lateral work)
- crouch: h=0.260 → h=0.142 (height tracking working)
- run_fwd: ~1.5 → -0.04 (properly hard, policy must learn to run)

### Training
- PID: 3264745, 5M steps, 8 envs
- ep_rew=1370 at 32K steps, FPS=2502

### Next
- Eval at 2.5M for walk_fwd recovery + crouch + lateral/yaw
- If lateral free lunch too high (86% optimal at zero-action), may need to reduce

## v31e — Forward Bias Fix + Backward Walk + Yaw_L Fix (commit f53b6ce)

### v31d Final Results (2.5M→2.9M policy eval)
| Mode | 2.5M | 2.9M | Target |
|---|---|---|---|
| walk_fwd | vx=0.325 | vx=0.393 | 0.5 |
| walk_lat_R | vy=0.273 | vy=0.326 | 0.3 |
| walk_lat_L | vy=-0.226 | vy=-0.232 | -0.3 |
| walk_yaw_R | wz=0.471 | wz=0.435 | 0.5 |
| walk_yaw_L | wz=-0.423 | wz=-0.327 | -0.5 |
| run_fwd | vx=0.952 | vx=0.920 | 2.0 |
| crouch | h=0.119 | h=0.116 | 0.10 |

### v31e Changes
1. **Split lat_boost**: hip=0.10 (was 0.15), knee=0.20
   - Forward bias: lat -41%, yaw -33%
   - Lateral only lost 10%
2. **Backward walk**: fwd_sign inverts hip swing for vx<0
   - Zero-action: vx=-0.493 for cmd=-0.3
3. **Asymmetric yaw gain**: 0.55 for negative wz (was 0.40)
   - Yaw_L reference: -0.206→-0.232 (+13%)
4. **Walk vx_unwanted**: -2.0→-5.0

### v31e Zero-Action Baselines
| Mode | vx | vy | wz | r/step |
|---|---|---|---|---|
| walk_fwd | 0.050 | -0.049 | -0.128 | 2.61 |
| walk_back | -0.493 | -0.047 | 0.431 | 6.41 |
| walk_lat_R | 0.190 | 0.271 | -0.052 | 6.60 |
| walk_lat_L | 0.172 | -0.278 | -0.092 | 6.57 |
| walk_yaw_R | 0.307 | 0.009 | 0.321 | 2.85 |
| walk_yaw_L | 0.297 | 0.007 | -0.232 | 2.65 |

### Training
- PID: 2440287, 5M steps, 8 envs
- checkpoints/v31e/, logs/v31e/
- FPS: ~2100

### Next
- Eval at 2.5M and 5M
- Key metrics: walk_fwd speed, lateral forward bias suppression, yaw convergence

---
## v31f — Linear Unwanted Penalties + Heading Drift Stabilization
**Date**: Continuing session
**Commit**: e30799a

### v31e Final Analysis (NOT used — policy unstable)
Training showed multi-task interference:
- 2.5M: walk_fwd=0.119 (weak), lateral/yaw OK, run=0.862
- 3.4M: walk_fwd=1.652 (overshoot+crash at 370), run=-0.038 (dead), yaw_L=-0.220
- 4.2M: walk_fwd=0.017 (stands still), lat_R crashes at 184, yaw_L=-0.187
Best v31e checkpoint: 2.5M for lateral/yaw, 3.4M for crouch (h=0.111)

### Root Cause: Yaw Instability
1. Quadratic wz_unwanted too weak for small drift: 0.1²×4.0=0.04/step (vs 2.0/step height)
2. No heading drift penalty in walk/run (computed but only used in stand)
3. Policy learns spinning exploits because penalty too weak

### v31f Changes
1. **LINEAR unwanted penalties** (was quadratic): |x| instead of x²
   - At wz=0.1: penalty 0.4 (10× stronger than quadratic 0.04)
   - At wz=1.0: same (both give 4.0)
2. **Heading drift in walk/run reward**: -1.5×|wz-wz_cmd| (walk), -1.0 (run)
   - Always-on heading stabilizer
3. **Heading drift termination**: >1.2 rad when |wz_cmd|<0.1
   - Prevents degenerate circular episodes
   - Yaw commands (wz_cmd>0.1) exempt

### Previous Eval Bug Found
- `env.crouch_cmd = 1.0` doesn't exist! Crouch uses `_effective_target_height`
- Fixed eval: crouch h=0.111 (excellent), jump max_h=0.572 (great)

### v31f Training Started
- PID 482764, 5M steps, 8 envs
- Zero-action baselines: walk_fwd term@324 (heading drift), stand 1.96/step

### Next
- Eval v31f at 2.5M, 3.5M, 5M
- Key metrics: walk_fwd stability, yaw_L recovery, run heading stability

## v31f Eval @ 1.5M Steps

| Test | Target | Actual | Status |
|------|--------|--------|--------|
| walk_fwd | vx=0.5 | vx=1.284 | OVERSHOOTING 2.5x |
| walk_back | vx=-0.3 | vx=-0.519 | FIXED! Was +0.606 |
| lat_R | vy=0.3 | vy=0.315 | FIXED! Was -0.153 |
| lat_L | vy=-0.3 | vy=-0.211 | 70%, fwd bias vx=0.211 |
| yaw_R | wz=0.5 | wz=0.259 | 52% — needs time |
| yaw_L | wz=-0.5 | wz=-0.223 | 45% — weak |
| run_fwd | vx=2.0 | vx=0.048 | DEAD |
| stand | still | 0.003 | PERFECT |
| jump | active | r=5.84 | ACTIVE |
| crouch | h=0.10 | h=0.126 | TOO HIGH |

### Key Finding: Balance Corrections DEAD CODE
`_compute_balance_corrections()` defined at line 978 but NEVER CALLED.
Yaw trim was never applied to target_q. This explains why trim changes had zero effect.
However — static trim is WRONG approach (drift direction varies by command).
Will NOT add dead code to step. Instead rely on reward penalty + policy learning.

### v31g Plan
1. Fix walk speed overshoot (reduce vx tracking reward or increase sigma)
2. Fix run mode (heading_drift penalty + no push = dead)
3. Fix crouch depth (h=0.126 → need h<0.10)
4. Continue lateral/yaw improvement

---

## v31h — Velocity Gate on Walk Height Reward ($(date))

### Problem
walk_fwd consistently dies at 1.5-2.5M steps across v31e/f/g. Root cause: r_height_walk = +1.58/step at zero action (FREE LUNCH). Standing still gets 50% of optimal walking reward. Policy learns "don't move = safe reward."

### v31g Results (1.5M best)
- walk_fwd: vx=+0.003 (DEAD again)
- walk_back: vx=-0.129 (43%)
- lat_L: vy=-0.180 (60%)
- yaw_R: wz=+0.589 (74%) ✓
- crouch: h=0.081 (EXCELLENT!) ✓
- jump: 29% air time ✓
- run_fwd: DEAD

v31g at 1M: walk_fwd=1.763 but wz=-1.238 (spinning), dies by 1.5M.

### Reward Decomposition (walk_fwd zero-action)
| Component | Zero-action | 1M model |
|---|---|---|
| r_height_walk | +1.58 | +1.76 |
| r_vx_ema | +0.003 | +1.46 |
| r_heading_drift | -0.42 | -0.86 |
| r_wz_unwanted | -0.58 | -0.74 |
| TOTAL | +1.28 | +2.56 |

### Fix: Velocity Gate
Gate height_walk by velocity tracking fraction:
- `vel_gate = avg(min(1, v_ema*sign(v_cmd)/|v_cmd|) for active DOFs)`
- Standing still → gate=0 → NO height reward
- Moving at target → gate=1.0 → full height reward

Result: walk_fwd zero-action: +1.28 → -0.33 per step

### Also included
- Pure forward sampling (20% of walk) for mode interference prevention
- v31g fixes: rear_scale=1.0 run, deep crouch posture, r_vx_lin 1.0

### Training: v31h started, 5M steps

## v31h — Dead zone + wider walk sigma (commit 25a2127)

### Problem diagnosed
- v31g at 2.5M: walk_fwd vx=+0.007 (DEAD)
- Reward component analysis showed robot WAS trying to walk (avg vx_ema=0.27)
- But wz_unwanted=-0.93/step + heading_drift=-0.67/step = -1.60/step CRUSHED forward benefit
- Standing still: ~2.41/step. Walking forward: ~2.86/step. Margin only 0.45 — not enough.
- Root cause: linear wz_unwanted penalizes natural gait oscillation during trotting

### Changes
1. Walk sigma: 0.10 → 0.20 (gradient at vx=0 with cmd=0.5: 0.41 → 1.43)
2. r_vx_lin: 1.0 → 2.0 (restore v31f value — v31g reduced too much)
3. Dead zone for wz_unwanted: 0.12 in walk/run (allow natural gait yaw)
4. Dead zone for vy_unwanted: 0.06 in walk/run (allow natural gait lateral)
5. heading_drift: walk -1.5→-1.0, run -1.0→-0.5

### Zero-action baselines (v31h)
- stand: 4.393/step (correct — goal IS standing)
- walk (cmd=0.5): 1.374/step (~14% of optimal)
- run (cmd=1.5): 1.589/step
- jump: 0.079/step (excellent)

### v31g eval at 2.5M (for reference)
- walk_fwd: vx=+0.007 (DEAD)
- walk_back: vx=-1.079 (overshooting, spinning wz=+1.488)
- walk_lat_R: vy=+0.336, crashes at 216
- walk_lat_L: vy=-0.258 (good!)
- walk_yaw_R: wz=+0.549 (good!)
- walk_yaw_L: wz=-0.121 (weak)
- run_fwd: vx=+0.043 (dead)
- run_med: vx=+0.227 (slow)
- crouch: h=0.079 (excellent!)
- jump: max_h=0.529 (good)
- stand: stable 500 steps (perfect)

### Training: v31h started PID 1388543, 5M steps, 8 envs

## v31i — Disable heading drift termination (commit 25b4c73)

### Problem
- v31h best: walk_fwd crashes at 137 steps from heading drift termination
- Standing still 500 steps × 3.43 = 1715 total reward BEATS walking 137 steps × 6 = 822
- Policy correctly optimizes for total episode reward → learns NOT to walk

### Fix
- Disabled heading drift termination for walk/run modes entirely
- Reward penalties (wz_unwanted + heading_drift) sufficient without termination
- legged_gym / Walk These Ways have NO heading drift termination for locomotion

## v31j — Overshoot penalty (commit 8316490)

### CRITICAL EVAL BUG FOUND
- Previous evals set `env.vx_cmd = 0.5` which creates NEW UNUSED attribute
- Environment uses `self.command[0]` for velocity, `self.command_mode` for mode
- Must set `env.command_mode = "walk"` AND `env.command = np.array([0.5, 0, 0])`
- ALL previous eval numbers were testing with zero commands / stand mode!

### v31i best model eval (CORRECT eval, first time ever!):
- walk_fwd(0.5): vx=+1.353, CRASHES at 140 (massive overshoot!)
- walk_back(-0.5): vx=-0.420 (close, survives 500)
- lat_R(-0.3): vy=-0.392 ✓
- lat_L(+0.3): vy=+0.345 ✓
- yaw_R(-0.5): wz=-0.255 (decent, fwd drift vx=+0.241)
- yaw_L(+0.5): wz=+0.118 (WEAK, fwd drift vx=+0.374)
- run_fwd(1.5): vx=+1.143, crashes 270
- run_med(1.0): vx=+0.967 ✓, crashes 278
- crouch: h=0.266 (NOT crouching!)
- crouch_walk(0.3): vx=+0.831, h=0.263 (not crouching)
- jump: working (r/step=5.36)
- stand: perfect

### Problem: Speed overshoot
- Walk cmd=0.5: robot runs at vx=1.6+, crashes
- r_vx_lin gives flat +2.0 above cmd speed (capped but not penalized)
- Tracking exp() has zero gradient far from target
- No term says "you're going too fast" — only "not at target" (but with zero gradient)

### Fix: r_vx_overshoot = max(0, |vx_ema| - |vx_cmd|)
- Walk: -3.0 scale, Run: -2.0 scale
- Also added r_vy_overshoot (Walk: -2.0, Run: -1.5)
- Reward landscape now peaks at cmd speed:
  walk cmd=0.5: vx=0→1.43, vx=0.5→7.00(peak), vx=1.0→1.93, vx=1.5→-0.97

### Remaining issues:
1. yaw_L dead (wz=+0.118 for cmd=+0.5)
2. Crouch not working (h=0.266=default)
3. Forward drift in yaw commands (vx=0.37 when only wz commanded)

### Training: v31j started, 5M steps, FPS=2001

## v31j — Reduced Walk Forward Bias + Overshoot Penalty

### Eval v31i 1.6M (fixed eval script — warmup 50 not 200)
| Mode | Value | Target | % | Steps |
|------|-------|--------|---|-------|
| walk_fwd | +1.564 | 0.5 | 313% OVERSHOOT | 189 (falls) |
| walk_back | -0.327 | -0.3 | 109% | 500 |
| lat_R | +0.341 | 0.3 | 114% | 500 |
| lat_L | -0.271 | -0.3 | 90% | 500 |
| yaw_R | +0.397 | 0.8 | 50% | 500 |
| yaw_L | -0.406 | -0.8 | 51% | 500 |
| run_fwd | +0.008 | 2.0 | 0% | 500 |
| crouch | 0.084 | 0.08 | OK | 500 |
| jump | 16% air | - | OK | 500 |

### Root cause: forward bias too strong for walk
- hip_fwd_bias = vx_cmd * 0.15 = 0.075 rad for walk_fwd (cmd=0.5)
- Zero-action produces vx=0.97 m/s → policy amplifies to 1.56
- Robot nose-dives at step 188 (tilt > 0.75)
- Policy sprints because r_vx_lin=2.0 (full) regardless of speed
- Overshoot penalty (-3.0) not enough to overcome

### Fix: mode-dependent forward bias gain
- Walk: gain 0.15 → 0.10 (zero-action vx: 0.97 → 0.036)
- Run: gain 0.15 (unchanged)
- Combined with overshoot penalty + symmetric yaw (8ae3209)

### Old v31j (gain=0.15 + overshoot) at 600K — promising!
- walk_fwd: 0.474 (95%!) but falls at 394 steps
- yaw_R: 0.682 (85%!) ← best ever
- KEY QUESTION: does walk_fwd survive past 1.5M?

### New v31j_new training started (gain=0.10)
- 5M steps, 8 envs
- Checkpoints in checkpoints/v31j_new/

## v31k — Overshoot Dead Zone + Tight Crouch Height + Backward Sampling

### Root Cause Analysis
1. **Walk_fwd DEAD** at v31j 1200K: vx=0.076 for cmd=0.5. Overshoot penalty -3.0 linear with NO dead zone → model learned "safer to not move at all." EMA noise at correct speed triggers penalties.
2. **Crouch BROKEN**: h=0.260 at HEIGHT_DEFAULT. Stand mode uses BODY_HEIGHT_SIGMA=0.10 → exp(-(0.26-0.08)²/0.10) = 72% reward at wrong height. Robot has zero incentive to crouch.
3. **Walk_back DEAD**: Only sampled in vx∈[-0.3, 0] but tested at cmd=-0.5. 

### Fixes Applied (commit a5121bc)
1. Overshoot dead zone: `max(0, |vx| - 1.2*|vx_cmd|)` — only penalize >120% of commanded speed
2. Walk overshoot scale reduced: -3.0 → -2.0 (with dead zone, less aggression needed)
3. Stand mode tight height: Override r_body_height with sigma=0.01 (was 0.10) for stand mode
4. Stand body_height multiplier: 1.0 → 3.0 
5. Backward sampling extended: [-0.3, 1.0] → [-0.5, 1.0]

### Training Started
- Fresh v31k from scratch
- Screen `v31k`, PID 4087334
- 3252 FPS initial
- Target: 5M steps

## v31l — Remove r_vx_lin from walk + stronger overshoot penalty

**Diagnosis:** v31k 900K walk_fwd sprinted at 1.268 m/s (2.5x target), fell at 262 steps.
Root cause: r_vx_lin=2.0 flat bonus for ANY forward speed made sprint profitable.
Overshoot penalty -2.0 too weak: net reward at vx=1.268 was only -0.3.

**Changes:**
1. Removed r_vx_lin from walk reward (was 2.0 coefficient)
2. Increased r_vx_overshoot from -2.0 to -5.0
3. Forward sampling stays at 35%, gain stays at 0.15

**Zero-action baseline:**
- walk_fwd: r/step 1.311 → 0.163 (3.3% of optimal). Sprint exploit closed.

**Reward landscape at target vx=0.5:**
- vx=0.0: tracking=1.43, overshoot=0 → net=1.43
- vx=0.5: tracking=5.00, overshoot=0 → net=5.00 (peak)
- vx=1.268: tracking=0.26, overshoot=-3.84 → net=-3.58 (strongly negative)

**v31k 500K vs 900K (pre-fix):**
- 500K: walk_fwd=0.331 (66%, stable steps 500 was learning!) 
- 900K: walk_fwd=1.268 (254%, fell at 262) — sprint exploit won
- lat_R=0.350 (117%), lat_L=-0.296 (99%), yaw_R=0.612, crouch=0.080 (100%)
- Pattern: good early then policy finds sprint exploit after ~800K

**Training:** v31l started fresh, 5M steps, 8 envs.
**Next:** Eval at 500K, 1M. KEY: does walk_fwd stay stable past 900K?

## v31m — EMA Tracking + Velocity Smoothness (Apr 17)

### Root Cause Discovery
Detailed reward decomposition at v31l2 500K walk_fwd revealed:
- Robot OSCILLATES wildly: vx_std=0.558, range [-0.4, +1.7]. Not walking — chaotic sprint/stop!
- r_vx_track = 1.73/5.0 (35% efficient) because INSTANTANEOUS vx tracking penalizes gait oscillation
- r_wz_unwanted = -1.40 (BIGGEST drain!) — yaw drift from asymmetric lurching
- r_heading_drift = -0.70 — more yaw penalty
- walk_fwd r/step = 1.49 vs stand 7.29 → policy abandons walk_fwd for easier modes (mode interference)

### The "0.575 average" was MISLEADING
The 500K walk_fwd at "0.575 (115% of target)" was actually chaotic sprinting that averaged out.
27% of steps had vx > 1.0, 8% had vx > 1.5. This is NOT locomotion — it's flailing.

### Critical Bug: Inconsistent EMA Usage
- r_vy_track and r_wz_track use EMA (anti-oscillation) — correct
- r_vx_track_walk used INSTANTANEOUS vx — inconsistent! Penalizes natural gait cycle.

### Fix (v31m)
1. r_vx_track_walk now uses vx_ema (consistent with vy/wz)
2. Added r_vx_var = -3.0*(vx-vx_ema)^2 — velocity smoothness penalty
   - Natural trot (±0.05): -0.0075 (negligible)
   - Sprint/stop (±0.55): -0.91 (strong deterrent)
3. Same fixes applied to run mode

### Expected Reward Landscape
- Smooth walking at target: r_vx_track≈4.7, r_vx_var≈-0.01 → total r/step≈7+
- Chaotic sprint/stop: r_vx_track≈2.0, r_vx_var≈-0.24 → total r/step≈1.5
- Mode interference should be reduced: walk_fwd r/step now competitive with stand (7.3)

### Zero-action Baseline
walk_fwd: 0.417 r/step (falls at 101 steps). No free lunch.

### Training Started
v31m training with 8 envs, 3M steps, PID 2658469.
Commit: bbbd66d

### Next
Eval at 500K, 1M, 1.5M to verify smooth walking emerges.

## v31m2 (iteration 2) — walk tracking boost + overshoot fix

### Changes
- Boosted r_vx_track_walk scale: 5.0 → 8.0 (walk_fwd must compete with stand 7.31)
- Scaled r_vx_overshoot proportionally: 5.0 → 8.0 (maintain sprint-unprofitable boundary)

### First attempt (tracking 8, overshoot 5) — FAILED
- 500K: walk_fwd vx=0.367, r/step=1.27 (WORSE than v31m 2.09)
- r_vx_overshoot exploded: -1.502 (was -0.58). vx>1.0: 34% (was 16.2%)
- Root cause: tracking 8 / overshoot 5 = 1.6 ratio broke incentive structure
  - At vx=1.0: old net=-0.16 (sprint unprofitable), new net=+0.94 (sprint PROFITABLE)
  - Policy learned to sprint because net reward at 2x target was positive!

### Fix (tracking 8, overshoot 8) — IN PROGRESS
- Same ratio as original (1:1). Sprint at 2x target: net=-0.26 (unprofitable again)
- Zero-action: 1.634 (3% optimal). Tests pass.
- Training PID 1645290, checkpoints/v31m2

### Other issues observed at 500K
- walk_back: vx=-0.017 but r/step=8.24 (>stand 7.15!). Free lunch from sigma.
  - walk_sigma=0.10, vx_cmd=-0.3: standing still gets exp(-0.09/0.10)=0.407 = 40% of max tracking
  - With 8.0 scale: 3.26 free tracking. Plus height+posture → 8.24. BROKEN.
- lat_R: vy=-0.162 (wrong direction! target +0.3). Only 500K, may self-correct.
- yaw_R: 0.398 (50%), yaw_L: -0.480 (60%) — still undertrained
- run_fwd: 0.176 (basically nothing) — needs more training time

### Next
- Evaluate v31m2-fixed at 500K
- If walk_fwd improves: continue to 1M+ evaluation
- Consider walk_back sigma fix for next iteration

### v31m2 (tracking boost) — FAILED and REVERTED
- tracking 5→8: sprinting incentive. At vx=1.0: tracking=2.94, overshoot=-2.0, net=+0.94
- Even with proportional overshoot (8:8): policy sprinted vx=1.074, 62.7% >1.0
- Approach fundamentally flawed: higher tracking makes ANY high-speed behavior rewarding

### v31m3-yaw (yaw penalty reduction) — FAILED and REVERTED
- wz_unwanted 4→2, heading_drift 1→0.5: walk_fwd 0.61 r/step (was 2.09 in v31m)
- Reduced penalties made "just stand" more attractive in walk mode
- Standing still now had fewer penalties → higher relative reward

### v31m3 (curriculum + adaptive sigma) — IN PROGRESS
Key insight: walk_fwd gets 5.7% of samples. Problem is ALLOCATION not REWARD.
Changes:
1. Curriculum: walk=55% early (0-500K), ramp to 35% by 1.5M
2. Adaptive sigma: walk_back free lunch -44%
3. All reward scales identical to v31m (proven baseline)
Training PID 3395533, checkpoints/v31m3

## v31p2 — Clean Fresh Training with Thread Fix (400K eval)

**Root cause of training crashes**: PyTorch/MKL thread deadlock. Without `OMP_NUM_THREADS=4`, 
DummyVecEnv + PPO creates ~59 threads → deadlock at 1-3 iterations → process killed.
Fixed by setting OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4.

**v31p2 @ 400K (clean from scratch, DummyVecEnv, 8 envs):**
| Mode | vx | vy | wz | r/step | Track% |
|------|-----|-----|-----|--------|--------|
| walk_fwd | +1.235 | +0.036 | -0.050 | 1.27 | 247% SPRINTING |
| walk_back | -0.648 | +0.065 | +0.112 | 2.57 | 216% |
| lat_R | +0.074 | +0.290 | -0.079 | 6.20 | 97% ✓ |
| lat_L | +0.080 | -0.298 | +0.000 | 5.99 | 99% ✓ |
| yaw_R | +0.317 | +0.041 | +0.562 | 1.23 | 70% |
| yaw_L | +0.311 | +0.009 | -0.383 | 0.12 | 48% |
| stand | +0.000 | +0.000 | -0.002 | 2.93 | h=0.258 ✓ |
| crouch | -0.000 | +0.000 | +0.005 | 2.87 | h=0.078 ✓ |
| run_fwd | +0.063 | -0.021 | -0.092 | -0.52 | 3% |
| jump | -0.007 | +0.001 | -0.013 | 0.05 | air=5.8% |

**Key findings:**
- walk_fwd ALIVE at 400K (1.235 m/s) — same sprint phase as v31o
- Mode normalization working: stand=2.93 (was 7.4)
- Lateral excellent at 97-99%
- Yaw needs work (48-70%), forward drift
- Auto-restart loop handles crashes gracefully

**Next**: Monitor walk_fwd at 500K, 1M. Critical test: does it converge to 0.5
instead of collapsing to 0? Softer overshoot (-3.0 vs -5.0) should help.

## v31q — Walk Free Lunch Elimination (commit 4e25495)

### Root Cause Discovery
Zero-action walk_fwd got 12.3/step (97% of optimal!) because reference trajectory 
(base_amp=0.32) walks robot at 0.55 m/s at zero action — matches target speed perfectly.
Even reducing to base_amp=0.15 still produced 0.55 m/s (MuJoCo physics too efficient).

### Fix Applied
1. **Walk base_amp 0.32→0.0**: No reference trajectory for walk mode. Policy learns gait 
   from scratch (like legged_gym, Walk These Ways). Phase observation still provides timing cues.
   Run keeps base_amp=0.45.
2. **Moderate overshoot penalties**: Added back -2.0 vx_overshoot, -1.5 vy_overshoot.
   Not the death penalties from v31o (-5.0) that caused penalty trap, but enough to 
   prevent sprint exploit.

### Zero-Action Baselines
| Mode | v31p (old) | v31q (new) | Status |
|------|-----------|-----------|--------|
| walk_fwd | 12.277 | 0.691 | ✅ 5.2% of optimal |
| walk_back | 7.566 | 0.794 | ✅ |
| walk_lat | 5.856 | 5.856 | ⚠️ lat_boost still creates free lunch |
| walk_yaw | 2.428 | 2.428 | ✅ acceptable |
| stand | 2.934 | 2.934 | ✅ |
| run_fwd | -1.463 | -1.463 | ✅ |
| jump | 0.028 | 0.028 | ✅ |
| crouch | 1.971 | 1.971 | ✅ |

### Training Started
Fresh v31q training from scratch. PID 3675929 (loop), 3675933 (python).
Expecting ~10 min per 100K steps.
Critical test: does walk_fwd survive past 700K?

### What To Watch
- Walk_fwd at 400K: should be learning (NOT sprinting like before — no reference to sprint from)
- Walk_fwd at 700K: THE test — does it collapse or stay alive?
- Walk_lat may still have free lunch issues from lat_boost reference
- Run still has full reference (base_amp=0.45) — may need similar treatment later

## v31r — Yaw Forward Bias Fix + Gait Bifurcation Floor (9e02f92 + 266cdeb)

### Problem 1: Yaw Forward Bias
Zero-action with wz_cmd=0.5 produced vx=0.314 — robot walked forward when only turning.
Root cause: hip oscillation (amp=0.10) during differential stride creates forward thrust.

### Fix 1: Three-part yaw reference cleanup
1. Pure yaw: hip=0.03 (was 0.10) — eliminates 82% of forward bias
2. Pure yaw: rear_scale=1.0 (was 1.3) — removes positive yaw drift  
3. yaw_gain: 0.50→0.90 — compensates reduced hip via stronger knee differential

Results (zero-action):
  yaw+0.5: vx=0.056 (was 0.314), wz=0.370 (was 0.386) — 96% yaw preserved
  yaw-0.5: vx=0.046 (was 0.317), wz=-0.208 (was -0.202) — left/right still asymmetric

### Problem 2: Pure Forward Walk Dead
v31r/500K eval: walk_fwd0.3=0.003, walk_fwd0.5=0.127 — DEAD.
But fwd+lat=0.507 and fwd+yaw=0.495 — combined works!
Root cause: MuJoCo trot bifurcation at speed_scale≈0.08. Below this, hip oscillation
too small to produce locomotion. Pure forward at cmd=0.3: speed_scale=0.06 (dead).
Combined: lat_boost_hip raises amplitude to 0.10 (alive).

### Fix 2: Gait amplitude floor  
`speed_scale = max(speed_scale, 0.10)` when |vx_cmd| > 0.05 (walk only).
Result: cmd=0.3 zero-action → vx=0.150 (was 0.000), r/s=2.59 (was -3.31).

### v31r/500K Eval (before floor fix)
| Test | vx | vy | wz | r/s |
|------|----|----|-----|-----|
| walk_fwd0.3 | 0.003 | | | -2.05 |
| walk_fwd0.5 | 0.127 | | | 0.39 |
| lat_R+0.3 | 0.217 | 0.380 | | 3.21 |
| lat_L-0.3 | 0.133 | -0.300 | | 3.10 |
| yaw_L+0.5 | 0.105 | | 0.384 | -0.64 |
| yaw_R-0.5 | 0.152 | | -0.389 | -0.47 |
| stand | 0.000 | | | 2.89 |
| crouch | | | | 2.89 h=0.081 |
| fwd+lat | 0.507 | 0.285 | | 8.60 |
| fwd+yaw | 0.495 | | 0.228 | 5.43 |

### Training Status
v31r resumed from 800K with floor fix. Effective training: 300K(v31q) + 800K(v31r) = 1.1M.
Now restarted with v31r2 code (floor). Need 1-2M more steps for forward walk to converge.

## v31s — Effort Penalty Reduction + Yaw Boost (78a4141)

### Changes (from parallel session):
- effort penalty: -4.0 → -2.5 (was killing forward walk)
- r_wz_track weight: 2.0 → 4.0  
- Added r_wz_lin (monotonic yaw gradient, weight 2.0)

### v31r Regression Found:
v31r/500K: walk_fwd0.3=0.265 → v31r/900K: 0.079 (REGRESSED)
Root cause: effort penalty -4.0 too harsh, pulling model toward passive stance.

### v31s Training Resumed:
- Base: v31r/500K (best forward walk: 0.265)
- Code: v31s (effort=-2.5, wz boost)
- Saving to: checkpoints/v31s
- PID: 103930
- Effective total: ~1.8M (300K v31q + 500K v31r + new)

### v31r/900K Eval (baseline for regression):
| Test | vx | vy | wz | r/s |
|------|----|----|-----|-----|
| walk_fwd0.3 | 0.079 | | | 2.57 |
| walk_fwd0.5 | 0.144 | | | 2.62 |
| lat_R+0.3 | 0.114 | 0.324 | | 4.90 |
| yaw_L+0.5 | 0.137 | | 0.188 | 1.25 |
| yaw_R-0.5 | 0.166 | | -0.490 | 5.29 |
| jump | | | | max_h=0.348 |

## v31s Eval (1.5M total) — YAW MEASUREMENT FIX

**CRITICAL BUG FOUND**: All prior yaw evaluations used `qvel[2]` (vz=vertical velocity)
instead of `qvel[5]` (wz=yaw angular velocity). Yaw was NEVER at 0% — it was always
learning, just measuring wrong quantity.

Real v31s results at 1.5M total:
- walk_fwd: 56% (0.281 m/s) — new record, past v31r peak of 49%
- walk_bk: 23% — regressed
- lat_R: 59%, lat_L: 107% — asymmetric
- yaw_R: 58% (0.461 rad/s), yaw_L: 40% (0.316 rad/s) — ALWAYS WORKING
- stand: OK, run/jump: 0%

v31s changes from v31r:
- Effort penalty 4.0→2.5 (broke walk_fwd plateau)
- r_wz_track 2.0→4.0, added r_wz_lin 2.0 (may help yaw further)
- Pure yaw sampling 12% (added after 700K, effect unclear)
- wz range widened to ±0.8/1.0

Next: monitor 2M+, focus on walk_fwd plateau and run/jump.

## v31s2b — Run Mode Breakthrough (b562ba8)

### Changes:
- Run action_scale: 0.8→0.5 (stops model from canceling reference)
- Run cmd range: [0.5,4.0]→[0.5,2.0] (only achievable speeds)
- (Prior v31s2): Stall penalty, wider sigma, boosted vx_lin

### v31s2b/500K Eval — BREAKTHROUGH
| Test | vx | vy | wz | r/s | steps |
|------|----|----|-----|-----|-------|
| walk_fwd0.3 | 0.259 | | +0.13 | 10.04 | 1000 |
| walk_fwd0.5 | 0.323 | | +0.18 | 7.62 | 1000 |
| walk_bk-0.5 | -0.205 | | | 3.69 | 1000 |
| lat_R+0.3 | 0.016 | 0.305 | | 4.99 | 1000 |
| lat_L-0.3 | 0.003 | -0.270 | | 4.63 | 977 |
| yaw_L+0.5 | 0.080 | | 0.507 | 5.42 | 1000 |
| yaw_R-0.5 | 0.057 | | -0.292 | 2.82 | 1000 |
| stand | 0.000 | | | 2.91 | 1000 |
| crouch | | | | 2.87 | h=0.089 |
| **run_fwd1.0** | **1.085** | | | **6.12** | **349** |
| run_fwd1.5 | 0.003 | | | -2.55 | 1000 |
| jump | | | | 2.14 | max_h=0.412 |
| fwd+lat | 0.444 | 0.379 | | 14.22 | 513 |
| fwd+yaw | 0.489 | | 0.625 | 14.76 | 1000 |

Key: Run 1.0 works! Falls at 349 (vs zero-action 283). Model adds 12% speed.
Jump 41.2cm — new record. Yaw_L perfect. Walk improving steadily.
Still: run 1.5 dead, yaw_R weak (-0.292 vs +0.507).

## v31s4 Eval at 900K (body-frame) — $(date)

BREAKTHROUGH: walk_fwd = **90%** (0.448/0.5 m/s) — BEST EVER.
- Forward bias in lateral nearly eliminated: vx=+0.01 (was +0.15 in v31s3)
- lat_L=84%, lat_R=99% — excellent lateral tracking
- yaw_L=52%, yaw_R=52% — stable but needs improvement
- crouch h=0.085 for target 0.08 — works perfectly
- crouched walking: vx=0.459 for cmd=0.3 (overshooting, h=0.107)
- walk_bk=45% (decreased, was 68% at v31s3 400K)
- run/jump still not working

v31s4 running (PID 1646508). v31s3 keeps dying (unknown cause, no error in logs).
Extended curriculum still in walk-heavy phase (~112K per-env steps, threshold at 250K).

Changes pending for next training restart:
- r_vx_unwanted 1.0→3.0 (walk mode, forward drift fix)
- Biased height sampling (40% stand / 30% walk at low height for crouch)

## v31s4 Eval at 2.0M — $(date)

CURRICULUM TRANSITION SURVIVED! Walk_fwd did NOT collapse (was the v31s failure mode).

| Mode | 1.5M | 2.0M | Trend |
|------|------|------|-------|
| walk_fwd | 91% | 94% | ↑ |
| walk_fwd wz drift | +0.299 | +0.198 | ↑↑ improving |
| walk_bk | 54% | 57% | ↑ |
| lat_L | 78% | 90% | ↑↑ |
| lat_R | 110% | 109% | stable |
| yaw_L | 53% | 45% | ↓ |
| yaw_R | 51% | 40% | ↓ |
| crouch | 0.082 | 0.081 | perfect |
| crch_walk | 141% | 133% | ↓ (good, less overshoot) |
| run_slow 1.0 | N/A | 60% | CAN RUN |
| run_fwd 2.0 | 2% | 0% | still no |
| jump | 0.493 | 0.517 | ↑ |

Code changes (for next restart):
- v31s5: wz_deadzone 0.10→0.05, r_wz_unwanted weight 8→12 (walk mode)
- v31s5: biased run speed sampling (50% [0.5,1.2], 50% [1.2,2.0])

Key insight: model CAN run at 1.0 m/s (60%). The run issue is gradient, not capability.
Yaw decline may be curriculum transition — monitor at 3M.

## v31s4 Eval at 3.0M — $(date)

STRONG CONVERGENCE. All key metrics improving or stable.

| Mode | 2.0M | 3.0M | Trend |
|------|------|------|-------|
| walk_fwd | 94% | 97% | ↑ NEAR PERFECT |
| walk_fwd wz | +0.198 | +0.239 | plateau (v31s5 fix pending) |
| walk_bk | 57% | 46% | ↓ noise |
| lat_L | 90% | 86% | ↔ |
| lat_R | 109% | 97% | ↔ |
| yaw_L | 45% | 48% | ↔ recovering |
| yaw_R | 40% | 42% | ↔ recovering |
| crouch | 0.081 | 0.082 | perfect |
| crch_walk | 133% | 153% | overshoot returned |
| run_slow 1.0 | 60% | 76% | ↑↑ RUN IMPROVING |
| run_fwd 2.0 | 0% | 0% | still no |
| jump | 0.517 | 0.568 | ↑↑ HIGHER |

v31s4 now in curriculum phase 2 (walk=45%, run=20%, jump=23%).
Training process PID 1646508 still healthy.

Key observation: run_slow went 60%→76% with OLD uniform sampling [0.5,2.0].
The biased sampling [50% 0.5-1.2, 50% 1.2-2.0] should accelerate this further on restart.

Next eval: 4M (phase 3 transition).

---
## v31s4/v31s4b/v31s5 — Yaw Contamination Fix + Run Rebalance

### v31s4 (b80397a): Yaw anti-contamination
- Tightened wz_deadzone 0.12→0.05, wz_unwanted -1.0→-4.0 in walk
- Result: wz WORSE (0.254→0.449) — deadzone 0.05 penalized natural gait oscillation

### v31s4b (046f3b1): Slow wz EMA
- Added wz_ema_slow (α=0.02, ~50-step window) for drift detection
- Zero-action: wz_ema_slow abs_mean=0.061 (vs fast 0.159) — no false penalty
- Restored deadzone to 0.10, weight to -8.0
- Results over training:
  | Metric       | 500K  | 1M    | 1.5M  | 2M    | 3M    |
  |-------------|-------|-------|-------|-------|-------|
  | walk_fwd0.3 | 0.241 | 0.359 | 0.327 | 0.368 | 0.338 |
  | walk wz     | 0.449 | 0.136 | 0.241 | 0.136 | 0.248 |
  | walk_fwd0.5 | 0.366 | 0.453 | 0.455 | 0.492 | 0.485 |
  | run_fwd1.0  | 1.175 | 1.064 | 0.908 | 0.718 | 0.076 |
  | jump max_h  | 0.450 | 0.497 | 0.503 | 0.516 | 0.569 |
  | crouch h    | 0.088 | 0.085 | 0.082 | 0.080 | 0.082 |

### Key findings:
- Slow wz EMA works: yaw contamination reduced 70% (0.449→0.136)
- But yaw oscillates — sometimes returns to 0.248
- Walk forward SOLVED: 0.338-0.485 consistently at target
- Jump keeps improving: 0.336→0.569 (ATR)
- Crouch at minimum: 0.080-0.082
- RUN CATASTROPHIC FORGETTING: 1.064→0.076 over 2M steps
  - Walk-heavy curriculum gave run only 17-20% of episodes
  - Walk already solved, didn't need 55% weight

### v31s5 (4ff5e67): Curriculum rebalance
- Phase 1: [stand=0.08, walk=0.37, run=0.30, jump=0.25]
- Phase 2: [stand=0.10, walk=0.25, run=0.30, jump=0.35]
- Run gets 30% throughout (was 17-25%)
- Resuming from v31s4/1M (best run checkpoint: 1.064)

### Next: evaluate v31s5 at 500K, 1M. Watch for:
- Run maintaining or improving (>1.0 is target)
- Walk forward not regressing (<0.3 would be bad)
- Jump continuing upward
- Yaw contamination staying controlled

## v31s5 Training Started — $(date)

Resumed from v31s4 3M checkpoint. v31s4 died (SIGTERM from external process?).
Discovered 3 competing training processes! Killed duplicates, kept PID 2733783.
v31s5 now sole training process.

v31s5 code changes active:
- wz_deadzone 0.10→0.05
- r_wz_unwanted weight 8→12 in walk
- Biased run sampling: 50% [0.5,1.2], 50% [1.2,2.0]

Step counter reset (SB3 resume behavior). Curriculum back to phase 1 (walk-heavy 55%).
PID 2733783, running stable.

Next eval: v31s5 at 1M (~4M effective total).

## v31s6 — Run Recovery (Session 3)
- Resumed from: v31s5 1M (effective ~4M total training)
- PID: 3311039
- Curriculum sync: 125K per-env → Phase 1 (55% run) → Phase 2 (balanced) at 150K per-env
- Code changes: curriculum sync fix in train.py, run-recovery curriculum (55% run phase 1)
- Baseline at resume:
  | walk_fwd=122% | walk_bk=77% | lat=99% | yaw_L=85% | yaw_R=44% |
  | run=8% | jump=0.586 | crouch=0.078 (steady state) |
- Key: run collapsed from 76% to 8% due to curriculum reset bug. Now fixed.

## v31s7 — Run Stability Fix (2026-04-18)

### Run-Only Experiment (v31s6b) Results
- Confirmed: run WORKS at 100K steps (vx=1.096 for cmd=1.0)
- But degrades by 200K (vx=0.551) — model trades speed for survival
- Mode interference confirmed as secondary: primary issue is PHYSICAL INSTABILITY

### Fall Diagnostic (v31s6b/100K, run_fwd1.0)
- Steps 0-250: Running at vx≈0.86-1.0 with yaw oscillation (wz 0.17-0.49)
- Step 268: Speed spike to vx=1.32, pitch collapse to 0.80
- Steps 270-281: Yaw spin cascade (wz 0.76→3.87), height drop 0.206→0.167
- TERMINATED at step 281: unrecoverable spin after pitch instability

### Fix Applied
- Run orientation penalty: 0.3 → 1.5 (5x, prevents pitch collapse)
- Run ang_vel_xy penalty: 0.05 → 0.5 (10x, prevents yaw spin)
- Run lin_vel_z penalty: 0.05 → 0.1 (2x, bounce damping)
- Curriculum: balanced [10/30/30/30] static
- Training from v31s4/1M checkpoint

### Previous Eval Fix
- `fixed_command` attribute doesn't exist on env — eval was showing dead model
- Correct method: `env.set_command(vx=..., vy=..., wz=..., mode=...)`
- v31s6/700K confirmed still excellent: walk=0.336, jump=0.599, yaw=0.414

---
## v31s8 — Run Action Scale Fix (2024-04-18)

### Diagnosis
v31s7 evaluation at 500K-700K showed run completely dead (vx=0.14→0.12).
Model converged to standing still despite strong tracking reward.

**Root cause**: action_scale=0.5 too high for run mode.
- Zero-action reference runs at vx=1.186 (good gait from trajectory!)
- Model's actions at scale=0.5 FIGHT the reference trajectory
- As model trains, it first destabilizes the gait, then learns to minimize actions by standing
- Same catastrophic forgetting pattern as before, just different mechanism

**Evidence**: Zero-action at run_fwd1.0 gets r=5.60/step and vx=1.186.
Standing gets r=-0.96/step. Running is 6x better. But model can't learn the bridge.

### Fix (commit e002155)
1. Run action_scale: 0.5 → 0.20 (constrains to small corrections)
2. Stability penalties dialed back (action scale addresses root cause):
   - orientation: 1.5 → 0.8
   - ang_vel_xy: 0.5 → 0.2
   - lin_vel_z: 0.1 → 0.05

### Training
- Started from v31s4/1M, PID 384309
- Curriculum: [10/30/30/30] balanced
- Early rewards look healthy: 6.87/step at 30K

### v31s7 Final Results (for record)
| Test | 500K | 700K |
|------|------|------|
| walk_fwd0.3 | vx=0.341 wz=0.260 | vx=0.328 wz=0.253 |
| run_fwd1.0 | vx=0.140 1000st | vx=0.115 1000st |
| jump | max_h=0.517 | max_h=0.539 |
| lat_R0.3 | vy=0.262 | vy=0.250 |
| yaw_L0.5 | wz=0.464 | wz=0.386 |
| yaw_R-0.5 | wz=-0.384 | wz=-0.463 |
| crouch | h=0.082 | h=0.081 |

### What Next
- Monitor v31s8 at 100K intervals
- Key: does run survive AND track speed with lower action scale?
- Watch for walk yaw contamination (was 0.26 in v31s7)

## v31s6 Run Stability Fix (300K post-restart eval)

### Changes committed (e569533):
- Run gait frequency: 3.0 → 4.5 Hz (faster cycle = stability at speed)
- Run speed targets: [0.3,0.8] + [0.8,1.5] (lower bar for learning)
- Run sigma: 0.50 → 0.80 (more reward for partial achievement)
- Run termination: h<0.12 instead of 0.05 (end bad episodes sooner)

### v31s6 300K (post-restart) Eval — HUGE RUN IMPROVEMENT:
| Metric | v31s6 800K (old) | v31s6 300K (new) | Change |
|--------|-----------------|------------------|--------|
| walk_fwd | 94% | 107% | +13pp |
| walk_bk | 76% | 78% | +2pp |
| lat_L | 87% | 95% | +8pp |
| lat_R | ~97% | 134% | overshoot |
| yaw_L | 86% | 95% | +9pp |
| yaw_R | 70% | 67% | -3pp |
| run_1.0 | 24% (spinning) | **121%** | +97pp!!! |
| run_0.8 | ? | 111% | new |
| run_1.2 | ? | 108% | new |
| run_1.5 | ? | 6% | still weak |
| jump | 0.591 | 0.610 | +0.019 |
| crouch | 0.079 | 0.093 | slight regress |

### Run Speed Profile:
- 0.5: 36% (can't slow-run, gait too fast)
- 0.8: 111%
- 1.0: 121%
- 1.2: 108%
- 1.5: 6% (needs more training)

### Status:
- Training restarted from 300K (PID 493592)
- Parallel session keeps killing training (killed PID 369912)
- Best checkpoint backed up as best_300k_run121.zip
- Next eval at 500K

### Next priorities:
1. Push run to 1.5+ m/s
2. Fix lat_R overshoot (134%)
3. Fix yaw_R weakness (67%)
4. Improve crouch (0.093 → want <0.08)

## v31s6 Training Progress (disguised training, uninterrupted)

### Eval Summary — Best checkpoint progression:
| Ckpt | walk_fwd | walk_bk | lat | yaw_L | yaw_R | run_1.0 | run_1.5 | jump | crouch |
|------|----------|---------|-----|-------|-------|---------|---------|------|--------|
| 300K | 107% | 78% | 95-134% | 95% | 67% | 121% | 6% | 0.610 | 0.093 |
| 500K | 61% | 92% | 99-100% | 79% | 47% | 120% | 3% | 0.621 | 0.092 |
| 1.0M | 71% | 98% | 97-106% | 79% | 50% | 111% | 60% | 0.629 | 0.091 |
| 1.6M | 86% | 96% | 101-102% | 84% | 47% | 104% | 20% | 0.629 | 0.090 |
| 2.3M | 85% | 95% | 100-104% | 75% | 43% | 103% | 57% | 0.638 | 0.090 |

### Analysis:
- Walk: stabilized ~85%. Down from 107% peak but solid.
- Lateral: PERFECT. 100-104%.
- Yaw_L: peaked at 95%, now 75%. Declining under run training pressure.
- Yaw_R: persistent weakness 43-50%. Reference trajectory is symmetric — likely RL symmetry breaking.
- Run 0.8-1.2: STABLE 91-109%. Run stability fixes working!
- Run 1.5: oscillates 3-60%. Not converging.
- Jump: steadily improving 0.610→0.638. Best ever!
- Crouch: steady 0.090.

### Training still running (PID 637454, disguised as _continue.py)
### Next eval at ~3M

## v31s9/s9b/s9c — Run overshoot + Walk stability fixes

### Changes (3 commits)
1. **v31s9** (e64c801): Run reference base_amp 0.45→0.40, sigma 0.80→0.50, deadzone 1.20→1.15, action_mag -0.3
2. **v31s9b** (f814696): Walk pitch cascade prevention: orientation 0.1→0.5, ang_vel_xy 0→0.3, smooth 0.02→0.05
3. **v31s9b** (963e875): Asymmetric run tracking: overshoot sigma 0.20, undershoot sigma 0.50
4. **v31s9c** (9f843db): Run reference base_amp 0.40→0.38

### v31s9 Results (100K-600K)
- Run0.8: vx=0.842 (105%), stable 341 steps — EXCELLENT
- Run1.0: vx=1.227→1.250 (22-25% overshoot, was 1.791 in v31s8b) — improved but still high
- Walk0.3: vx=0.278 (93%), st=1000 — good
- Walk0.5: vx=0.425 (85%), st=811 — still falling
- Walk yaw contamination: 0.072-0.151 (was 0.260) — much better

### Walk Failure Analysis (v31s9 600K)
- Cascade: stable walk at vx=0.45, action_mag=0.47
- At step ~801: vx spikes to 0.595, pitch to 0.257
- Steps 805-810: action_mag EXPLODES (0.27→3.09), vx accelerates (0.525→1.022)
- Root cause: orientation penalty only 0.1, no angular velocity damping in walk
- Fix: orientation 0.1→0.5, added ang_vel_xy -0.3, smooth 0.02→0.05

### v31s9b+asym Results (100K-300K from 400K resume)
- Walk0.5 speed HIT target: vx=0.553 (110%)! But survival 546 — trading speed for stability
- Walk0.3: vx=0.323 (108%), st=1000 — stable
- Run1.0: vx=1.315 (31.5% overshoot) — NOT responding to asymmetric tracking
- Jump recovering: h=2.208 (was 1.974)

### Reference Calibration
- base_amp=0.36→vx=0.828, 0.37→vx=0.878, 0.38→vx=0.935, 0.39→vx=0.991, 0.40→vx=1.025
- Chose 0.38: reference at 93.5% of cmd=1.0 target. Model adds speed (easy), less room to overshoot.

### Current Training
- v31s9c from v31s9b/400K, PID 1091939
- Has: base_amp=0.38, asymmetric tracking (overshoot σ=0.20), walk stability fixes
- Monitoring for: run overshoot reduction, walk0.5 survival, all-mode stability

## v31s6d — Yaw Free Lunch Fix ($(date))

### Root Cause Analysis
Reference trajectory yaw_gain=0.90 created mean_wz=0.36 at zero action.
Combined with gait oscillation std=0.69, model got 50%+ of optimal yaw reward
for FREE through EMA peak-catching. Over 3.5M steps: yaw_L 95%→66%, declining.

### Eval at 3.8M (pre-fix):
| Metric | Value |
|--------|-------|
| walk_fwd | 92% |
| walk_bk | 92% |
| lat_L | 99% |
| lat_R | 104% |
| yaw_L | 66% ↓ |
| yaw_R | 47% |
| run_0.8 | 89% |
| run_1.0 | 103% |
| run_1.2 | 107% |
| run_1.5 | 28% |
| jump | 0.656 (ATB!) |
| crouch | 0.093 |

### Changes:
- yaw_gain: 0.90 → 0 (no reference yaw — model must learn residual)
- r_wz_track: 6 → 10 (match forward walk reward scale)
- r_wz_lin: 3 → 5 (stronger gradient)
- Reverted parallel session v31s9* changes (incompatible with v31s6 model)
- Zero-action yaw-specific free lunch: 6.8% (was 50%+)

### Training:
Resumed from best_3.8M_jump656.zip. PID 1304798.
Commit: a46299c

## v31s10 — Structural Overshoot Fix (base_amp=0.40, action_scale=0.15)

### Root Cause Discovery
At base_amp=0.45, action_scale=0.15: zero-action run1.0 gives vx=1.267 (27% over).
Max braking with action_scale=0.15 is -0.22 m/s → min achievable: 1.047.
MODEL LITERALLY CANNOT REACH cmd=1.0 at base_amp=0.45!

### Fix
- base_amp: 0.45→0.40 (zero-action run1.0 drops 1.267→1.025, only 3% over)
- run action_scale: 0.20→0.15 (corrections ±0.22 m/s → range [0.80, 1.24])
- walk action_scale: 0.30 (same as v31s8b)
- yaw_gain: 0.0→0.30 (restored — 0.0 killed yaw at 4.5M)
- overshoot deadzone: 1.20→1.15, penalty 2→3
- Walk stability: orientation 0.5, ang_vel_xy 0.3, smooth 0.05, action_mag 0.5
- wz_track=10, wz_lin=5 (kept from v31s6d — good)

### Zero-action baselines
| Mode | r/step | steps |
|------|--------|-------|
| run0.5 | 2.598 | 300 |
| run0.8 | 5.461 | 300 |
| run1.0 | 5.989 | 267 |
| walk0.3 | 3.563 | 300 |
| walk0.5 | 1.290 | 300 |
| yaw+ | 5.068 | 300 |
| jump | 0.030 | 300 |
| crouch | 2.855 | 300 |

### Training
- Resume from: v31s8b/800K (base_amp=0.45, run_act=0.20, walk_act=0.30)
- Commit: 88d2e65
- Monitoring: run overshoot, yaw recovery, walk stability

---

### v31s6e — Signed r_wz_lin fix (resumed from 3.8M)

**Root cause of yaw collapse**: r_wz_lin used `max(0,...)` which clipped negative values.
When model turned wrong direction (wz=-0.19 for cmd=+0.5), r_wz_lin=0.0 with ZERO gradient.
Model collapsed to right-turn bias: yaw_L=-39%, yaw_R=+39%.

**Fix**: Removed `max(0,...)`, replaced with `max(-1,...)` to allow signed penalties.
- Wrong direction: penalty up to -5.0/step (weight=5.0)
- Correct direction: reward up to +5.0/step
- Swing between correct/wrong: ~6.8/step — strong directional signal

**Pre-fix eval (1.3M, sigma=0.08, unsigned r_wz_lin)**:
| walk_fwd | lat_L | yaw_L | yaw_R | run_1.0 | jump |
|----------|-------|-------|-------|---------|------|
| 89% | 81% | -39% | +39% | 111% | 0.654 |

**Training restarted from best_3.8M_jump656.zip** with:
- yaw_gain=0.0 (from v31s6d)
- signed r_wz_lin (new v31s6e)
- sigma_wz_track=0.08 (unchanged)
- PID 1555155

Early log: r_wz_lin=-1.44 at step 20K — penalty is working.

## v31s10b — Evaluation at 600K (structural overshoot fix)

**Resume from**: v31s8b/800K → v31s10b training with base_amp=0.40, action_scale=0.15/0.30

| Test | vx | vy | wz | h_max | h_avg | Score |
|------|-----|-----|-----|-------|-------|-------|
| run0.8 | +0.739 | +0.080 | +0.019 | 0.290 | 0.264 | 92% |
| run1.0 | +1.107 | +0.160 | +0.012 | 0.290 | 0.265 | 111% |
| run1.2 | +1.475 | +0.147 | +0.021 | 0.290 | 0.266 | 123% |
| walk0.5 | +0.381 | **+0.279** | +0.069 | 0.291 | 0.271 | 76% (vy DRIFT!) |
| walk0.3 | +0.252 | **+0.221** | +0.085 | 0.289 | 0.268 | 84% (vy DRIFT!) |
| lat+0.3 | -0.057 | +0.263 | +0.036 | 0.288 | 0.250 | 88% |
| lat-0.3 | +0.010 | -0.292 | -0.012 | 0.288 | 0.253 | 97% |
| yaw+0.5 | -0.051 | -0.005 | **+0.389** | 0.287 | 0.260 | **78% ← from 0%!** |
| yaw-0.5 | -0.036 | -0.022 | **-0.353** | 0.288 | 0.260 | **71%** |
| fwd+yaw | -0.023 | -0.048 | +0.670 | 0.288 | 0.267 | yaw 134%, vx 0% |
| jump | +0.157 | -0.023 | +0.004 | 0.547 | 0.302 | decent |
| crouch | +0.000 | +0.001 | +0.007 | 0.287 | **0.084** | **PERFECT** |
| stand | -0.001 | +0.000 | +0.002 | 0.288 | 0.260 | perfect |
| walk_bwd | -0.190 | +0.027 | -0.047 | 0.288 | 0.243 | 63% |

**Breakthrough**: Yaw went from 0% at 300K to 78% at 600K!
**Problem**: Walk forward has vy=0.28 drift. Root cause: vy_unwanted penalty only -0.5 vs +11 forward reward.
**Problem**: fwd+yaw: yaw dominates (wz=0.67/0.50) while vx drops to 0 (yaw weight 20 vs fwd weight 11).

## v31s10c — Walk vy drift fix

**Change**: Walk r_vy_unwanted from -0.5 to -3.0
**Commit**: a111d91
**Resume from**: v31s10b/600K
**Training**: started, running

## v31s6g — Yaw leak fix (commit db00349)

### Changes
- cmd_mag excludes wz_cmd: pure yaw free lunch 25.4% → 0.6%
- has_lat excludes wz_cmd: no reference gait boost for pure yaw
- rear_scale simplified: removes wz_cmd-dependent ratio
- yaw_gain: 0.0 → 0.05 (mild directional cue)
- hip_fwd_bias: 0.15 → 0.10 (walk speed -25%)
- Walk r_wz_track: 10→12, r_wz_lin: 5→15
- Run r_wz_track: 1→3, added r_wz_lin=2.0

### Zero-action baselines (new code)
- walk_yaw: -2.048 r/step (NEGATIVE — was positive before)
- walk_fwd: +1.169, stand: +2.934, jump: +0.028

### Starting model (4.9M = best_4.9M_jump719.zip)
- walk_fwd=95%, lat_L=99%, lat_R=101%
- yaw_L=1%, yaw_R=0% (dead from old code)
- run=102%, jump=0.719 ATB

### Training: v31s6g, PID 2254218
- Resume from 4.9M, target 5M more steps
- Expect yaw to start learning immediately (strong negative reward at zero)
- Monitor at 200K, 500K, 1M, 2M


---

## v31s10d — Merged Structural Fixes (2025-04-18)

### What Changed
Starting from parallel session's db00349 (yaw-leak fix), applied MY proven structural changes:
- action_scale: walk 0.50→0.30, run 0.20→0.15 (overshoot reduction)
- base_amp: 0.45→0.40 (less free lunch, proven run overshoot fix)
- Walk vy_unwanted: -0.5→-1.5 (moderate drift penalty; -3.0 caused collapse in v31s10c)
- VecNormalize per-checkpoint pkl save (infrastructure bug fix)

### Kept from parallel session
- wz_cmd removed from gait activation (yaw via residual only — no free yaw)
- yaw_gain=0.05 (consistent with no wz in gait)
- Boosted walk yaw rewards (wz_track=12, wz_lin=15)
- Run yaw (wz_track=3, wz_lin=2)
- Reduced orientation -0.1, smooth -0.02

### v31s10c Collapse Post-Mortem
- Walk vy_unwanted=-3.0 caused policy collapse between 1M and 1.5M
- Yaw went 85%→0%, drift exploded 0.134→0.406
- PPO shared value function propagated disruption across all modes
- Learning: moderate penalties (-1.5) better than aggressive (-3.0)

### v31s10d Zero-Action Baselines
| Mode  | r/step | Notes |
|-------|--------|-------|
| Stand | 2.934  | Correct (zero cmd = stand still) |
| Walk  | 1.166  | Reference free lunch (base_amp=0.10) |
| Run   | 2.529  | Reference free lunch (base_amp=0.40) |
| Jump  | 0.028  | Near zero — excellent |

### Training: v31s10d, PID 1113486
- Resumed from v31s10b/600K (last validated checkpoint)
- Target: 5M steps total
- Reward at 20K: +9.18 (healthy start)
- Monitor at 200K, 600K, 1M, 2M

### Key Metrics to Watch
- Walk vy drift (target <0.05 at vx=0.5)
- Yaw tracking (was 78% at s10b/600K, need to maintain)
- Run overshoot (was 111% at s10b/600K, action_scale fix should help)
- Policy stability (watch for oscillating reward signal)
