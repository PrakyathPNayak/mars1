# Reward v6.1 — Grace Period & Eval Fix

## Status: READY FOR TRAINING

## Changes from v6

### Termination Grace Periods (cheetah_env.py)
- TERMINATION_GRACE_STEPS = 50 (1s after reset: no early termination)
- MODE_TRANSITION_GRACE_STEPS = 25 (0.5s after mode switch)
- Tracks _last_mode_change_step, updated on reset and mode resample
- Root cause: untrained policy died in ~29 steps (0.58s), no time to learn

### Eval Best-Model Fix (train.py)
- DelayedEvalCallback: skips eval until 200K training steps
- Root cause: deterministic do-nothing policy at 50K steps scored 21K reward
  (stood still for 2001 steps with survival_mult) → saved as "best" forever
- Increased n_eval_episodes: 5 → 10 for more reliable mode coverage

## Previous v6 Changes (preserved)
- r_smooth: -0.05 L2 → -0.01 L1
- r_gait: 2.0 → 3.5 (4 sub-components including stride_freq)
- r_linvel: 4.0 → 5.0
- r_dof_vel: -2e-4 → -1e-4
- Foot clearance: 5cm → 8cm
- LR floor: 1e-5, log_std_init: -1.0
- n_epochs: 10, batch_size: 2048

## Validation
- 39/39 tests passing
- Devil's advocate: verified grace period doesn't mask genuine falls
- Devil's advocate: verified DelayedEvalCallback correctly skips early evals

## Files Modified
- src/env/cheetah_env.py (grace periods)
- src/training/train.py (DelayedEvalCallback)
