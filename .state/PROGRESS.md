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
[10:05:53] Step   20,000 | timesteps=480,000
[10:21:26] Step   30,000 | timesteps=720,000
[10:34:35] Step   40,000 | timesteps=960,000
[11:03:18] Step   10,000 | timesteps=240,000
[11:03:41] Step   10,000 | timesteps=240,000
[11:17:05] Step   20,000 | timesteps=480,000
[11:17:49] Step   20,000 | timesteps=480,000
[11:34:59] Step   30,000 | timesteps=720,000
[11:35:44] Step   30,000 | timesteps=720,000
[11:47:05] Step   40,000 | timesteps=960,000
[11:47:41] Step   40,000 | timesteps=960,000
[11:56:45] Step   50,000 | timesteps=1,200,000
[12:03:23] Step   50,000 | timesteps=1,200,000
[12:18:41] Step   10,000 | timesteps=240,000
[12:33:48] Step   20,000 | timesteps=480,000
[12:50:05] Step   30,000 | timesteps=720,000
[13:02:35] Step   40,000 | timesteps=960,000
[13:14:59] Step   50,000 | timesteps=1,200,000
[13:24:05] Step   60,000 | timesteps=1,440,000
[13:35:30] Step   70,000 | timesteps=1,680,000
[13:49:15] Step   80,000 | timesteps=1,920,000
[14:03:13] Step   90,000 | timesteps=2,160,000
[14:17:15] Step  100,000 | timesteps=2,400,000
[14:29:36] Step  110,000 | timesteps=2,640,000
[14:42:57] Step  120,000 | timesteps=2,880,000
[14:54:20] Step  130,000 | timesteps=3,120,000
[15:06:21] Step  140,000 | timesteps=3,360,000
[15:17:12] Step  150,000 | timesteps=3,600,000
[15:31:03] Step  160,000 | timesteps=3,840,000
[15:44:47] Step  170,000 | timesteps=4,080,000
[15:55:42] Step  180,000 | timesteps=4,320,000
[16:06:47] Step  190,000 | timesteps=4,560,000
[16:18:19] Step  200,000 | timesteps=4,800,000
[16:31:14] Step  210,000 | timesteps=5,040,000
[16:42:39] Step  220,000 | timesteps=5,280,000
[16:54:20] Step  230,000 | timesteps=5,520,000
[17:06:11] Step  240,000 | timesteps=5,760,000
[17:17:57] Step  250,000 | timesteps=6,000,000
[17:28:45] Step  260,000 | timesteps=6,240,000
[17:40:45] Step  270,000 | timesteps=6,480,000
[17:51:26] Step  280,000 | timesteps=6,720,000
[18:02:11] Step  290,000 | timesteps=6,960,000
[18:14:27] Step  300,000 | timesteps=7,200,000
[18:36:45] Step   10,000 | timesteps=240,000
[18:51:12] Step   20,000 | timesteps=480,000
[19:03:45] Step   30,000 | timesteps=720,000
[19:16:46] Step   40,000 | timesteps=960,000
[19:35:08] Step   10,000 | timesteps=240,000
[19:52:22] Step   20,000 | timesteps=480,000
[20:08:14] Step   30,000 | timesteps=720,000
[20:23:54] Step   40,000 | timesteps=960,000
[20:40:22] Step   50,000 | timesteps=1,200,000
[20:51:47] Step   60,000 | timesteps=1,440,000
[21:06:31] Step   70,000 | timesteps=1,680,000
[21:22:35] Step   80,000 | timesteps=1,920,000
[21:35:10] Step   90,000 | timesteps=2,160,000
[22:03:19] Step   10,000 | timesteps=240,000
[22:19:45] Step   20,000 | timesteps=480,000
[22:36:39] Step   30,000 | timesteps=720,000
[23:00:25] Step   10,000 | timesteps=240,000
[23:15:03] Step   20,000 | timesteps=480,000
[23:32:22] Step   30,000 | timesteps=720,000
[23:47:14] Step   40,000 | timesteps=960,000
[00:03:17] Step   50,000 | timesteps=1,200,000
[00:16:59] Step   60,000 | timesteps=1,440,000
[00:31:18] Step   70,000 | timesteps=1,680,000
[00:51:14] Step   10,000 | timesteps=240,000
[01:07:35] Step   10,000 | timesteps=240,000
[01:14:29] Step   20,000 | timesteps=480,000
[01:21:40] Step   30,000 | timesteps=720,000
[01:28:30] Step   40,000 | timesteps=960,000
[01:35:29] Step   50,000 | timesteps=1,200,000
[01:41:58] Step   60,000 | timesteps=1,440,000
[01:54:08] Step   10,000 | timesteps=480,000
[02:05:02] Step   20,000 | timesteps=960,000
[02:15:21] Step   30,000 | timesteps=1,440,000
[02:25:50] Step   40,000 | timesteps=1,920,000
[02:36:16] Step   50,000 | timesteps=2,400,000
[02:46:44] Step   60,000 | timesteps=2,880,000
[02:57:14] Step   70,000 | timesteps=3,360,000
[03:07:40] Step   80,000 | timesteps=3,840,000
[03:18:19] Step   90,000 | timesteps=4,320,000
[03:28:38] Step  100,000 | timesteps=4,800,000
[03:38:34] Step  110,000 | timesteps=5,280,000
[03:49:08] Step  120,000 | timesteps=5,760,000
[03:58:38] Step  130,000 | timesteps=6,240,000
[04:08:57] Step  140,000 | timesteps=6,720,000
[04:18:34] Step  150,000 | timesteps=7,200,000
[04:27:26] Step  160,000 | timesteps=7,680,000
[04:37:40] Step  170,000 | timesteps=8,160,000
[04:47:30] Step  180,000 | timesteps=8,640,000
[04:57:39] Step  190,000 | timesteps=9,120,000
[05:07:29] Step  200,000 | timesteps=9,600,000
[05:53:19] Step   10,000 | timesteps=480,000
[06:04:11] Step   20,000 | timesteps=960,000
[06:14:16] Step   30,000 | timesteps=1,440,000
[06:24:33] Step   40,000 | timesteps=1,920,000
[06:35:18] Step   50,000 | timesteps=2,400,000
[06:45:18] Step   60,000 | timesteps=2,880,000
[06:55:56] Step   70,000 | timesteps=3,360,000
[07:05:58] Step   80,000 | timesteps=3,840,000
[07:16:23] Step   90,000 | timesteps=4,320,000
[07:27:05] Step  100,000 | timesteps=4,800,000
[07:37:43] Step  110,000 | timesteps=5,280,000
[07:48:45] Step  120,000 | timesteps=5,760,000
[07:59:00] Step  130,000 | timesteps=6,240,000
[08:08:58] Step  140,000 | timesteps=6,720,000
[08:18:31] Step  150,000 | timesteps=7,200,000
[08:28:10] Step  160,000 | timesteps=7,680,000
[08:38:08] Step  170,000 | timesteps=8,160,000
[08:48:09] Step  180,000 | timesteps=8,640,000
[08:58:29] Step  190,000 | timesteps=9,120,000
[09:08:21] Step  200,000 | timesteps=9,600,000
[12:11:47] Step   10,000 | timesteps=480,000
[12:21:38] Step   20,000 | timesteps=960,000
[12:31:05] Step   30,000 | timesteps=1,440,000
[12:41:43] Step   40,000 | timesteps=1,920,000
[12:51:38] Step   50,000 | timesteps=2,400,000
[13:01:28] Step   10,000 | timesteps=480,000
[13:11:33] Step   20,000 | timesteps=960,000
[13:20:26] Step   30,000 | timesteps=1,440,000
[13:29:15] Step   40,000 | timesteps=1,920,000
[13:39:01] Step   50,000 | timesteps=2,400,000
[13:47:36] Step   60,000 | timesteps=2,880,000
[13:58:19] Step   70,000 | timesteps=3,360,000
[14:16:55] Step   10,000 | timesteps=480,000
[14:28:58] Step   20,000 | timesteps=960,000
[15:09:03] Step   10,000 | timesteps=480,000
[15:23:16] Step   20,000 | timesteps=960,000
[15:44:02] Step   10,000 | timesteps=480,000
[15:52:06] Step   20,000 | timesteps=960,000
[15:59:22] Step   30,000 | timesteps=1,440,000
[16:25:07] Step   10,000 | timesteps=480,000
[16:38:18] Step   20,000 | timesteps=960,000
[16:50:54] Step   30,000 | timesteps=1,440,000
[17:04:09] Step   40,000 | timesteps=1,920,000
[17:18:39] Step   50,000 | timesteps=2,400,000
[17:32:42] Step   60,000 | timesteps=2,880,000
[17:49:24] Step   10,000 | timesteps=480,000
[18:13:12] Step   10,000 | timesteps=480,000
[18:27:19] Step   20,000 | timesteps=960,000
[18:49:54] Step   10,000 | timesteps=480,000
[19:00:12] Step   20,000 | timesteps=960,000
[19:08:07] Step   30,000 | timesteps=1,440,000
[19:19:50] Step   10,000 | timesteps=240,000
[19:31:54] Step   10,000 | timesteps=480,000
[19:43:52] Step   20,000 | timesteps=960,000
[19:54:49] Step   30,000 | timesteps=1,440,000
[20:06:59] Step   40,000 | timesteps=1,920,000
[20:29:22] Step   10,000 | timesteps=480,000
[20:41:14] Step   20,000 | timesteps=960,000
[20:52:33] Step   30,000 | timesteps=1,440,000
[21:05:33] Step   40,000 | timesteps=1,920,000
[21:20:48] Step   50,000 | timesteps=2,400,000
[21:32:45] Step   60,000 | timesteps=2,880,000
[21:45:42] Step   70,000 | timesteps=3,360,000
[21:57:38] Step   80,000 | timesteps=3,840,000
[22:10:37] Step   90,000 | timesteps=4,320,000
[22:24:10] Step  100,000 | timesteps=4,800,000
[22:36:24] Step  110,000 | timesteps=5,280,000
[22:49:33] Step  120,000 | timesteps=5,760,000
[23:01:02] Step  130,000 | timesteps=6,240,000
[23:13:13] Step  140,000 | timesteps=6,720,000
[23:25:17] Step  150,000 | timesteps=7,200,000
[23:36:23] Step  160,000 | timesteps=7,680,000
[23:48:26] Step  170,000 | timesteps=8,160,000
[23:59:24] Step  180,000 | timesteps=8,640,000
[00:11:33] Step  190,000 | timesteps=9,120,000
[00:23:34] Step  200,000 | timesteps=9,600,000
[00:34:52] Step  210,000 | timesteps=10,080,000
[00:46:55] Step  220,000 | timesteps=10,560,000
[00:58:13] Step  230,000 | timesteps=11,040,000
[01:09:47] Step  240,000 | timesteps=11,520,000
[01:21:48] Step  250,000 | timesteps=12,000,000
[01:33:19] Step  260,000 | timesteps=12,480,000
[01:45:43] Step  270,000 | timesteps=12,960,000
[02:03:22] Step   10,000 | timesteps=480,000
[02:19:59] Step   10,000 | timesteps=480,000
[02:31:40] Step   20,000 | timesteps=960,000
[02:53:40] Step   10,000 | timesteps=480,000
[03:05:09] Step   20,000 | timesteps=960,000
[03:16:41] Step   30,000 | timesteps=1,440,000
[03:28:11] Step   40,000 | timesteps=1,920,000
[03:39:32] Step   50,000 | timesteps=2,400,000
[03:51:11] Step   60,000 | timesteps=2,880,000
[04:03:22] Step   70,000 | timesteps=3,360,000
[04:17:45] Step   80,000 | timesteps=3,840,000
[04:41:08] Step   10,000 | timesteps=480,000
[04:54:02] Step   20,000 | timesteps=960,000
[05:51:04] Step   30,000 | timesteps=1,440,000
[06:03:37] Step   40,000 | timesteps=1,920,000
[06:14:21] Step   10,000 | timesteps=480,000
[06:22:43] Step   20,000 | timesteps=960,000
[06:31:16] Step   30,000 | timesteps=1,440,000
[06:40:50] Step   40,000 | timesteps=1,920,000
[06:50:27] Step   50,000 | timesteps=2,400,000
[06:57:11] Step   60,000 | timesteps=2,880,000
[07:05:58] Step   10,000 | timesteps=480,000
[07:11:24] Step   20,000 | timesteps=960,000
[07:17:55] Step   30,000 | timesteps=1,440,000
[07:23:54] Step   40,000 | timesteps=1,920,000
[07:35:37] Step   10,000 | timesteps=480,000
[07:40:35] Step   20,000 | timesteps=960,000
[07:45:31] Step   30,000 | timesteps=1,440,000
[07:50:17] Step   40,000 | timesteps=1,920,000
[07:54:58] Step   50,000 | timesteps=2,400,000
[08:04:41] Step   10,000 | timesteps=480,000
[08:10:00] Step   20,000 | timesteps=960,000
