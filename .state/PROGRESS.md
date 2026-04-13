# Progress Log
[2026-03-30 05:02:20] Phase 0 (Bootstrap) complete
[2026-03-30 05:03:01] Phase 1 (Research) complete
[Phase 2] Generated Mini Cheetah MJCF model (18 bodies, 13 joints, 12 actuators)
[Phase 3] Created cheetah_env.py, keyboard_controller.py, exploration_policy.py
[Phase 4] Created train.py (PPO) and curriculum.py
[Phase 5] Created viewer.py and stats_dashboard.py
[Phase 6] Created test_env.py (10 tests) and evaluate.py
[Phase 7] Created run.py and Makefile
[Phase 8] Created README.md and docs/architecture.md
[Phase 9] All 10/10 tests pass. Training smoke test passed (4096 steps, 1543 fps).
[Phase 9] Fixed default stance: knee angles must be negative to match MJCF joint limits.
=== AUTONOMOUS SESSION COMPLETE ===
[05:19:07] Step   10,000 | timesteps=80,000
[05:20:27] Step   20,000 | timesteps=160,000
[05:21:43] Step   30,000 | timesteps=240,000
[05:23:46] Step   10,000 | timesteps=80,000
[05:25:15] Step   20,000 | timesteps=160,000
[05:26:35] Step   30,000 | timesteps=240,000
[05:27:58] Step   40,000 | timesteps=320,000
[05:29:22] Step   50,000 | timesteps=400,000
[05:30:46] Step   60,000 | timesteps=480,000
[05:32:04] Step   70,000 | timesteps=560,000
[05:33:23] Step   80,000 | timesteps=640,000
[05:34:37] Step   90,000 | timesteps=720,000
[05:36:04] Step  100,000 | timesteps=800,000
[05:37:27] Step  110,000 | timesteps=880,000
[05:38:55] Step  120,000 | timesteps=960,000

=== TRAINING SESSION 2 - REWARD FIX & FULL RUN ===

=== REWARD REDESIGN v3 — Stand/Shake/Crouch Fix ===
[setup] Progress tracking active, .state/ in .gitignore
[Bug Fix] Removed r_joint_acc penalty (was -12 to -57/step, dominating all other rewards ~2-3/step total).
[Training] PPO 1M steps, 8 envs, ~17 min, 978 fps (CPU).
[Result] Training complete: ep_len=195, ep_rew=426 at 1M steps.
[Best checkpoint] checkpoints/best/best_model.zip (saved ~200K steps, eval reward 2630 during training).

=== EVALUATION RESULTS (20 episodes, randomize_domain=False) ===
Best model (best_model.zip):
  mean_reward:   3112.45
  std_reward:     454.89
  max_reward:    3841.02
  min_reward:    2394.98
  mean_length:   1001.0
  survival_rate:    1.0 (100%)

Final model (cheetah_final.zip, 1M steps):
  mean_reward:   2167.90
  std_reward:     585.31
  max_reward:    2975.98
  min_reward:     237.87
  mean_length:    929.7
  survival_rate:    0.9 (90%)

[Conclusion] Best model significantly outperforms final model. Policy degraded after ~200K steps likely due to domain randomization noise.
[Video] logs/best_model_rollout.mp4 (1001 frames, reward 3268.95)
[15:45:48] Step   10,000 | timesteps=10,000
[15:46:05] Step   20,000 | timesteps=20,000
[15:46:29] Step   30,000 | timesteps=30,000
[15:46:50] Step   40,000 | timesteps=40,000
[15:47:14] Step   50,000 | timesteps=50,000
[15:47:36] Step   60,000 | timesteps=60,000
[15:47:59] Step   70,000 | timesteps=70,000
[15:48:20] Step   80,000 | timesteps=80,000
[15:48:42] Step   90,000 | timesteps=90,000
[15:49:04] Step  100,000 | timesteps=100,000
[15:49:25] Step  110,000 | timesteps=110,000

=== REWARD V3 REDESIGN — ANTI-SHAKE & CROUCH FIX ===
## Diagnosed issues (via 200-step simulation)
- STAND: r_total=+8.33/step for zero action, r_stand_still only 1.7% of total
- CROUCH: robot stays at 0.264m (target 0.18), r_body_height=+0.48 at WRONG height
- Missing r_dof_vel (joint velocity penalty — key anti-shake from legged_gym)
- r_posture exp kernel too narrow (exp(-6.8)=0.001 at crouch deviation)
- r_stand_still fires in crouch mode (opposes crouch target)
- command_mode never randomized during training

## Research sources
- legged_gym (Rudin 2022): _reward_dof_vel, _reward_stand_still, tracking sigma
- Humanoid-Gym (Gu 2024): dof_vel=-5e-4, tracking_sigma=5, action_smoothness
- Go1 URDF: trunk 5.2kg, 12.9kg total, 0.27m standing height, ±33.5Nm

## v3 Changes applied
- Added r_dof_vel=-0.1 (sum joint_vel²) — key anti-shake
- r_stand_still: -0.5→-3.0, gated by mode!=crouch/jump  
- r_smooth: -0.05→-0.3, r_body_acc: -0.0005→-0.005, r_action_jerk: -0.005→-0.02
- r_body_height: exp(-)→exp(-)-1.0 (centered at 0)
- r_posture: 0.3×exp(-)→-1.5×error (negative penalty, gradient everywhere)
- BODY_HEIGHT_SIGMA: 0.005→0.02
- Randomized command_mode in reset() and step() resample
- MLP default steps: 5M→3M, BC epochs: 50→100

## Validation results
- Stand+zero_action: +5.30/step (r_stand_still now 16% of total)
- Stand+oscillation: 0.0/step (r_dof_vel=-98/step completely crushes shaking)
- Crouch+wrong_posture: +0.83/step (was +6.49!)
- Crouch+correct: +5.58/step (4.75/step incentive to crouch)
- Walking: r_dof_vel only -0.06/step (not over-penalized)

## Meta-learning assessment
- MAML: not suited (single robot, narrow task distribution)
- RMA: partially covered by Transformer 16-step history (implicit adaptation)
- Domain randomization: already serves as implicit meta-training

## All 10/10 tests pass

=== REWARD V5 — MULTI-SKILL TRAINING OVERHAUL ===

## Diagnosis: Hierarchical Transformer+MoE PPO training failure at ~15M steps
Root causes identified:
1. Action std=1.49 (constant, never decreased) → random actions → immediate falls
2. Cosine LR decayed to ~1e-9 → no learning capacity
3. No skill mode encoding in observation → policy can't distinguish commands
4. No jump mode in training
5. 27 noisy reward terms → weak gradient signal
6. ONLY_POSITIVE_REWARDS=True hiding gradient information (clipping negatives to 0)

## Changes applied (Reward v5)
- OBS_DIM: 49 → 54 (added 5-dim skill one-hot: stand/walk/run/crouch/jump)
- Removed "trot" and "explore" modes → 5 canonical skills
- REWARD_SCALES: 14 terms (from 27), stronger alive bonus (2.0)
- MODE_REWARD_MULTIPLIERS: per-mode emphasis (Walk These Ways style)
- ONLY_POSITIVE_REWARDS: True → False
- log_std_init: -0.5 → -1.0 (std: 0.61 → 0.37)
- LR schedule: added min_lr=1e-5 floor
- Jump FSM: 5-phase (idle→crouch→launch→airborne→land) with phase-specific rewards
- Adaptive termination: crouch min_height=0.08m (vs 0.18m for other modes)

## Files modified
- src/env/cheetah_env.py (major overhaul: obs space, reward, modes, FSM)
- src/training/advanced_policy.py (OBS_DIM=54, skill_cmd sensory group)
- src/training/sb3_integration.py (log_std_init=-1.0)
- src/training/train_hierarchical.py (obs_dim=54, LR floor)
- src/training/train_advanced.py (obs_dim=54)
- src/utils/policy_loader.py (_BASE_OBS_DIM=54)
- src/training/curriculum.py (removed "trot" from skill sequence)
- src/control/keyboard_controller.py (default mode: walk)
- tests/test_env.py, tests/test_suite.py, tests/test_final_validation.py (obs_dim updates)
- training_config.json (obs_dim=54, modes, only_positive_rewards=false)
- run.py (explore→walk)

## Validation
- All 39 tests pass (unit, integration, performance, regression)
- Env produces correct obs shape (54,) with valid skill one-hot encoding
- Transformer policy forward pass OK with obs_dim=54
- Reward components returned correctly for all 5 modes

[08:04:57] Step   10,000 | timesteps=80,000
[08:27:58] Step   10,000 | timesteps=240,000
[08:42:10] Step   20,000 | timesteps=480,000
[08:57:11] Step   30,000 | timesteps=720,000
[09:11:04] Step   40,000 | timesteps=960,000
[09:26:43] Step   50,000 | timesteps=1,200,000
[09:51:46] Step   10,000 | timesteps=240,000
