# Changelog

## [Unreleased]

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
