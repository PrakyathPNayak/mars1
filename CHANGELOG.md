# Changelog

## [Unreleased]

### Cycle 4 — Reward v6: Gait & Training Optimization (2025-XX-XX)

**Diagnosis**: 10M-step MLP PPO training showed weak gait (r_gait=0.35-0.53),
smoothness penalty hurting locomotion (r_smooth=-0.15 to -0.19/step), frozen
exploration std (0.637 throughout), LR decaying to 1.4e-6, and only 40 gradient
steps per rollout. Research into quadruped gait literature (SLIP models, legged_gym,
Walk These Ways, DreamWaQ) informed the following changes.

- **fix(reward)**: r_smooth penalty: L2 at -0.05 → L1 at -0.01 (5× reduction).
  L2 quadratically penalized the big leg swings needed for proper trotting gait.
  L1 is linear and tolerates cyclic motions (legged_gym convention: -0.01).
- **fix(reward)**: r_gait scale: 2.0 → 3.5. Added stride frequency sub-reward
  (exp kernel on touchdown count, peaks at trot=2). Gait now has 4 sub-components:
  air_time(0.3) + trot_symmetry(0.25) + foot_clearance(0.25) + stride_freq(0.20).
- **fix(reward)**: Trot symmetry: abs(diff)/2 → sqrt(abs(diff)/2) for smoother gradient.
- **fix(reward)**: Foot clearance target: 5cm → 8cm (5cm was too low for natural gait).
- **fix(reward)**: r_linvel scale: 4.0 → 5.0. Walk multiplier: 1.0 → 1.2. Run: 1.5 → 2.0.
- **fix(reward)**: r_dof_vel scale: -2e-4 → -1e-4 (was constant -0.315/step, halved).
- **fix(reward)**: Run mode r_smooth multiplier: -0.03 → 0.5× base (halves penalty for fast motion).
- **fix(reward)**: Walk mode r_gait multiplier: 1.5 → 2.0 (proper gait is key for walking).
- **fix(training)**: LR schedule: linear decay to 0 → linear decay to min 1e-5
  (prevents clip_fraction→0 and std freezing at end of training).
- **fix(training)**: log_std_init: -0.5 → -1.0 (initial std≈0.37, was frozen at 0.637).
- **fix(training)**: n_epochs: 5 → 10, batch_size: 4096 → 2048
  (160 gradient steps per rollout, was 40 — 4× more learning per trajectory).
- **docs**: Updated architecture.md with v6 reward tables and training parameters.
- **test**: All 39 tests pass (unit, integration, performance, regression).

### Cycle 3 — Reward v5: Multi-Skill Training Overhaul (2025-XX-XX)

**Diagnosis**: Hierarchical Transformer+MoE PPO training failed at ~15M steps.
Root causes: action std=1.49 (never learned), LR decayed to ~1e-9, no skill encoding
in observations, 27 noisy reward terms, ONLY_POSITIVE_REWARDS clipping gradients.

- **fix(env)**: Observation space expanded from 49 → 54 dims. Added 5-dim skill one-hot
  encoding (stand/walk/run/crouch/jump) as the last 5 observation elements.
- **fix(reward)**: Redesigned reward function (v5): 14 terms (down from 27), mode-dependent
  multipliers (Walk These Ways style). Each skill mode has its own reward emphasis.
- **fix(reward)**: Disabled `ONLY_POSITIVE_REWARDS` — was hiding gradient information
  by clipping negative rewards to zero.
- **fix(env)**: Removed "trot" and "explore" modes. Consolidated to 5 canonical skills:
  stand, walk, run, crouch, jump.
- **fix(env)**: Added jump finite state machine (FSM) with 5 phases:
  idle → crouch → launch → airborne → land. Phase-specific rewards and height targets.
- **fix(training)**: Learning rate schedule now has `min_lr=1e-5` floor (was decaying to 0).
- **fix(training)**: `log_std_init` changed from -0.5 to -1.0 (std: 0.61 → 0.37).
  Prevents chaotic exploration that prevented any learning.
- **feat(env)**: Mode-dependent reward multipliers: stand/crouch emphasize posture+stillness;
  walk/run emphasize velocity tracking+gait; jump uses FSM phase rewards.
- **feat(env)**: Adaptive termination: crouch mode uses lower min_height threshold (0.08m).
- **refactor**: Updated all downstream code for OBS_DIM=54: advanced_policy.py, sb3_integration.py,
  train_hierarchical.py, train_advanced.py, policy_loader.py, curriculum.py, keyboard_controller.py.
- **test**: All 39 tests pass (unit, integration, performance, regression).

### Cycle 2 — Body-Frame Velocity Tracking (2025-XX-XX)
- **fix(reward)**: Velocity tracking reward now computed in body frame (Isaac Gym convention).
  Before: world-frame velocity vs world-frame command caused confusing reward signal when robot yawed.
  After: body-frame velocity vs body-frame command, consistent with observation space.
- **perf(env)**: `_update_foot_contacts()` uses cached geom-ID-to-index dict (O(1) int lookup)
  instead of per-contact `mj_id2name()` string comparison.

### Cycle 1 — Reward Function Overhaul (2025-XX-XX)
- **fix(reward)**: Removed catastrophic `r_time` penalty that produced -450/step for the first
  500 steps, dominating all other reward terms by 2–3 orders of magnitude.
- **fix(reward)**: Added `REWARD_SCALES` dictionary with 14 independently weighted terms,
  each scaled so its contribution at typical operating conditions is O(0.1–1.0) per step.
- **fix(env)**: Tightened `_check_done()` termination from 66° tilt to 45° tilt (gravity_body[2] > -0.7);
  base height threshold raised from 0.10m to 0.15m.
- **fix(env)**: Base velocities in `_get_obs()` now rotated to body frame via `_quat_rotate_inv`
  for sim-to-real transfer consistency (DreamWaQ, RMA convention).
- **fix(env)**: Push perturbation changed from `qvel[:3] += force` (velocity hack) to
  `xfrc_applied` (proper MuJoCo external force integrated by physics engine).
- **feat(reward)**: Added 4 new penalty terms: `joint_acc` (high-frequency jitter),
  `joint_limit` (proximity to mechanical limits), `collision` (trunk/thigh ground contact),
  `stumble` (foot lateral hits on non-floor obstacles).
- **feat(env)**: Extended domain randomization: joint damping ±20%, armature ±20%,
  PD gains kp/kd ±15%, motor strength ±10%, foot friction 0.8–2.0.
- **fix(env)**: Floor friction range widened from 0.5–1.5 to 0.3–1.5 (CHRL consensus).
- **fix(env)**: PD gain kd raised from 1.0 to 2.0 (legged_gym convention, reduces oscillation).
- **fix(env)**: Episode length reduced from 5000 to 2000 steps (40s episodes at 50 Hz).
- **fix(env)**: Push magnitude changed from 0.5 m/s velocity to 50N force (5.2 m/s² for 9.57 kg robot).
- **fix(env)**: Stumble detection excludes floor contacts (heightfield normals are naturally shallow).
- **feat(config)**: Created `training_config.json` with hardware-tuned parameters for
  RTX 4090 + i9-14900 + 128 GB RAM (64 parallel envs, batch 8192, cosine LR decay).
