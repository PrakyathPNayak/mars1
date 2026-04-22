# Unitree Go1 Locomotion: Complete Training and Architecture Guide

This document explains how the full system works end-to-end in this repository: environment design, reward shaping, baseline PPO training, hierarchical BC + Transformer+MoE training, SB3 integration details, artifacts, and practical run/eval workflows.

It is intentionally detailed and code-aligned, so each section maps directly to current implementation behavior.

## 1. Big Picture

The repository supports two major policy development paths:

1. Standard MLP PPO (Stage 1 expert).
2. Hierarchical Transformer+MoE policy trained with BC warm-start and PPO fine-tuning (Stage 2).

In practice, the intended pipeline is:

1. Train MLP expert.
2. Collect expert rollouts.
3. Behavior clone a Transformer encoder from those rollouts.
4. Inject BC weights into SB3 Transformer policy.
5. Continue RL with PPO on the full architecture.

Orchestration is done by `scripts/pipeline.py`.

## 2. Repository Flow and Ownership of Logic

Primary control and ownership points:

- `run.py`: top-level CLI commands (`train`, `eval`, `demo`, `test`, etc.).
- `scripts/pipeline.py`: two-stage orchestration and output consolidation.
- `src/env/cheetah_env.py`: task definition (dynamics interface, observation, reward, done).
- `src/training/train.py`: Stage 1 MLP PPO.
- `src/training/train_hierarchical.py`: Stage 2 BC + hierarchical PPO.
- `src/training/advanced_policy.py`: architectural modules (sensory grouping, symmetry, temporal transformer, MoE, terrain/phase/auxiliary heads).
- `src/training/sb3_integration.py`: wrappers, SB3 policy bridge, and training callbacks.

Practical rule: if behavior is ambiguous between docs and code, treat these files as authoritative.

## 3. Environment Deep Dive (`MiniCheetahEnv`)

### 3.1 Core timing and actuation

Implemented in `src/env/cheetah_env.py`.

- Control timestep `dt`: 0.02 s (50 Hz policy frequency).
- Physics timestep `physics_dt`: 0.002 s (500 Hz MuJoCo stepping).
- Substeps per control step: `dt / physics_dt = 10`.
- Action space: 12-dim Box, clipped to `[-0.5, 0.5]`.
- Action semantics: delta joint target offsets from nominal stance.

### 3.2 Skill/mode system

Modes in current env:

- `stand`
- `walk`
- `run`
- `crouch`
- `jump`

These modes affect:

- Command randomization ranges.
- Reward multipliers (`MODE_REWARD_MULTIPLIERS`).
- Height targets (`HEIGHT_TARGETS`).
- Special jump FSM behavior.

### 3.3 Observation space (current env implementation)

Current env reports `OBS_DIM = 54` and builds observation as:

- `[0:12]`: joint positions.
- `[12:24]`: joint velocities.
- `[24:27]`: base linear velocity in body frame.
- `[27:30]`: base angular velocity in body frame.
- `[30:33]`: projected gravity.
- `[33:45]`: previous action.
- `[45:49]`: command `(vx, vy, wz, target_height)`.
- `[49:54]`: skill one-hot (5 modes).

If `randomize_domain=True`, Gaussian noise (`std=0.02`) is added only to first 49 dims, not to skill one-hot dims.

### 3.4 Action and command interfaces

- Policy action is applied every control step.
- `set_command(vx, vy, wz, mode)` allows external command injection.
- Mode transitions set a grace timestamp to avoid immediate termination after abrupt mode switch.
- `set_exploration_heading(heading_rad, speed)` maps heading to `(vx, vy)` command.

### 3.5 Reward function (v7, componentized)

Reward is computed in `_compute_reward` from explicit subcomponents then scaled.

Raw components (before scale and mode multiplier):

- Positive-style terms:
  - `r_linvel`: exponential xy velocity tracking.
  - `r_yaw`: exponential yaw-rate tracking.
  - `r_gait`: weighted gait quality aggregate.
  - `r_posture`: exponential joint target matching.
  - `r_body_height`: exponential target-height matching.
  - `r_stillness`: rational kernel in stand/crouch.
  - `r_jump_phase`: jump FSM phase reward.
  - `r_alive`: constant alive term.
- Penalty-style terms:
  - `r_orientation`: tilt penalty from gravity projection.
  - `r_ang_vel_xy`: roll/pitch angular velocity penalty.
  - `r_torque`: squared torque penalty.
  - `r_smooth`: squared action delta penalty (L2).
  - `r_joint_limit`: proximity to joint limit margins.
  - `r_lin_vel_z`: vertical velocity penalty.
  - `r_dof_vel`: squared joint velocity penalty.

Scaled reward computation is:

`total = sum_k raw[k] * REWARD_SCALES[k] * MODE_REWARD_MULTIPLIERS[mode].get(k, 1.0)`

The environment stores the final per-component scaled values in `info["reward_components"]`.

### 3.6 Gait substructure used in `r_gait`

`r_gait` combines:

- Air-time term based on first contact after swing.
- Diagonal-pair symmetry proxy.
- Swing foot clearance toward an 8 cm target.
- Stride frequency preference (touchdown count shaped around 2).

This creates a multi-factor gait objective rather than only command tracking.

### 3.7 Jump finite state machine

Jump reward/target handling uses an internal FSM:

1. `IDLE`
2. `CROUCH`
3. `LAUNCH`
4. `AIRBORNE`
5. `LANDING`

The FSM updates `target_height` dynamically and provides phase-specific rewards (crouch depth, upward velocity, airtime height, landing stability, peak-height bonus).

### 3.8 Termination and grace periods

Done logic (`_check_done`) includes:

- Initial grace period after reset.
- Mode-transition grace period after mode switch.
- Fall checks:
  - base height below threshold (mode-dependent minimum), or
  - excessive tilt (gravity z component threshold).

This avoids over-penalizing early transients and improves training stability.

### 3.9 Domain randomization

`_apply_domain_randomization` randomizes:

- Per-body masses (typically 0.85x to 1.15x scale).
- Floor friction scale.

This is intended as sim robustness training rather than exact real-robot identification.

## 4. Stage 1: MLP PPO Expert (`src/training/train.py`)

### 4.1 Why this stage exists

Stage 1 creates a reliable expert policy that can be imitated by a richer temporal policy in Stage 2.

### 4.2 Environment vectorization and normalization

Training setup:

- Vectorized envs: `SubprocVecEnv` preferred, fallback `DummyVecEnv`.
- Monitoring: `VecMonitor`.
- Observation normalization: `VecNormalize(norm_obs=True, norm_reward=False, clip_obs=100)`.
- Eval env: separate `VecNormalize` instance, synchronized during eval callback.

Normalization is crucial for PPO stability in locomotion with mixed-scale signals.

### 4.3 Policy architecture and PPO config

Policy:

- `MlpPolicy` with separate actor/critic networks.
- Net arch: `pi=[2048,1024,512]`, `vf=[2048,1024,512]`.
- Activation: ELU.
- Orthogonal initialization.
- `log_std_init=-1.0`.

PPO defaults in this stage:

- `learning_rate`: linear decay with floor (`3e-4 -> >=1e-5`).
- `n_steps=4096`.
- `batch_size=4096`.
- `n_epochs=10`.
- `gamma=0.99`, `gae_lambda=0.95`.
- `clip_range=0.2`.
- `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=1.0`.

### 4.4 Callback stack

- `CheckpointCallback`: periodic saves.
- `DelayedEvalCallback`: blocks early "best" selection before enough timesteps.
- `ProgressLogger`: periodic rollout/episode stats in console.
- `RewardComponentCallback`: logs reward breakdown for diagnosis.

### 4.5 Stage outputs

Typical output artifacts:

- Final model (SB3 zip).
- Best eval model (SB3 zip).
- `vec_normalize.pkl`.
- Training config JSON.
- TensorBoard and monitor logs.

`vec_normalize.pkl` must be retained with the model for consistent inference/evaluation.

## 5. Stage 2: Hierarchical BC + PPO (`src/training/train_hierarchical.py`)

Stage 2 is explicitly three-phase.

### 5.1 Phase 1: expert rollout collection

Function: `collect_expert_data(...)`.

Behavior:

- Loads expert PPO model.
- Uses flat env (no history wrapper) for rollout.
- Maintains a sliding history buffer per env to build temporal BC input.
- Records `(history_flat, action)` pairs.

Output:

- `history_data`: shape `(N, history_len * obs_dim)`.
- `act_data`: shape `(N, act_dim)`.

Notes:

- Supports parallel collection envs (`n_collect_envs`).
- Can load expert VecNormalize stats if provided.

### 5.2 Phase 2: behavioral cloning on Transformer extractor

Function: `behavioral_cloning_transformer(...)`.

What is trained:

- `TransformerExtractor` (same SB3 feature extractor class used in PPO).
- Lightweight BC action head on top.

Training setup:

- Loss: MSE between predicted and expert action.
- Optimizer: `AdamW`.
- LR schedule: cosine annealing over BC epochs.
- Gradient clipping enabled.
- GPU auto-selection if available.

Output:

- Extractor `state_dict` on CPU for portable weight injection.

### 5.3 Phase 3: PPO with full Transformer+MoE policy

Function: `train_ppo_hierarchical(args, bc_extractor_state)`.

Pipeline:

1. Create history-wrapped training envs.
2. Build PPO with `TransformerActorCriticPolicy`.
3. Inject BC-trained extractor weights into policy extractor.
4. Train with PPO and auxiliary callbacks.

PPO defaults in hierarchical stage:

- `n_steps=2048`.
- `batch_size=256`.
- `n_epochs` from args (default 10 in script).
- Warmup + cosine-like LR with floor (`peak ~3e-4`, floor `1e-5`).
- `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`.
- `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=0.5`.

Callback stack includes:

- `CheckpointCallback`.
- `EvalCallback`.
- `WorldModelCallback`.
- `CurriculumCallback`.
- `PhaseOscillatorResetCallback`.
- `ProgressLogger`.
- `RewardComponentCallback`.

## 6. Architecture Internals (`advanced_policy.py` + `sb3_integration.py`)

### 6.1 Important compatibility note about observation dimensionality

`advanced_policy.py` currently defines constants and sensory groups for a 196-dim observation layout (`OBS_DIM=196`, including heightmap groups), while the current `MiniCheetahEnv` implementation reports 54 dims.

Integration code in `train_hierarchical.py` and `sb3_integration.py` attempts to pass env-derived obs dimension where needed. If you modify observation schemas, verify all of these are synchronized:

1. Environment `OBS_DIM` and actual concatenation in `_get_obs`.
2. Any hardcoded group ranges in `SENSORY_GROUPS`.
3. `HistoryWrapper` and extractor reshape assumptions.
4. Model loading checkpoints trained under older layouts.

Treat this as a high-risk area when changing architecture or env observation format.

### 6.2 Core modules

Implemented modules include:

- `RunningNormalizer`: online mean/variance normalization.
- `SinusoidalPositionalEncoding`: sequence position encoding.
- `SensoryGroupEncoder`: per-sensor-group projection + cross-attention fusion.
- `SymmetryAugmenter`: bilateral reflection-based feature augmentation.
- `TemporalTransformerBlock`: causal self-attention + FFN.
- `MixtureOfExperts`: gated expert ensemble with load-balancing loss.
- `WorldModelHead`: next-observation prediction auxiliary head.
- `GaitPhaseOscillator`: phase signal from step count and command.
- `TerrainEstimator`: temporal conv estimator from proprioceptive history.
- `ContrastiveTemporalHead`: BYOL-style temporal coherence auxiliary loss.
- `PrivilegedEncoder`: teacher-side privileged feature encoder for distillation workflows.

### 6.3 High-level computation graph

For a history input tensor:

1. Normalize observations.
2. Encode each timestep through sensory group encoder.
3. Symmetry-augment the latest timestep embedding.
4. Add positional encoding.
5. Run causal temporal transformer layers.
6. Extract last-token temporal latent.
7. Derive terrain and phase features.
8. Fuse features.
9. Produce policy output through MoE.
10. Produce value through critic branch.
11. Optionally produce world model and contrastive auxiliary outputs.

### 6.4 MoE behavior

`MixtureOfExperts`:

- Learns gate probabilities over expert networks.
- Produces weighted expert output (`einsum` combination).
- Adds a load-balancing term to reduce expert collapse.

In debugging, monitor gate distributions over time to ensure multiple experts are used.

## 7. SB3 Bridge Details (`sb3_integration.py`)

### 7.1 Wrappers

- `ActionSmoothingWrapper`: EMA smoothing for actions.
- `HistoryWrapper`: stacks history into flattened vector (`history_len * obs_dim`).

### 7.2 Feature extractor (`TransformerExtractor`)

Responsibilities:

- Unflatten history.
- Update/apply running normalization.
- Encode timesteps and run transformer.
- Add symmetry, terrain, and phase features.
- Return fused latent vector for actor/critic heads.

### 7.3 SB3 policy class

`TransformerActorCriticPolicy`:

- Extends `ActorCriticPolicy`.
- Replaces standard MLP extractor with transformer extractor.
- Uses `_MoEExtractor` to provide actor latent and critic latent.
- Keeps SB3-compatible Gaussian action distribution.

### 7.4 Auxiliary callbacks

- `WorldModelCallback`: trains separate world-model head using rollout buffer, with detached policy features to avoid optimizer conflict.
- `PhaseOscillatorResetCallback`: resets step counter at rollout starts for phase coherence.
- `CurriculumCallback`: tracks episodic reward progression and adjusts difficulty via `AdaptiveCurriculum`.

## 8. Pipeline Orchestration (`scripts/pipeline.py`)

### 8.1 What pipeline.py does

`run_pipeline(args)` handles:

1. Optional Stage 1 execution (or skip with external expert).
2. Stage 2 execution using Stage 1 best model as expert by default.
3. Consolidation of outputs in one run folder.
4. Summary JSON generation with metadata and model paths.

### 8.2 Standard run folder structure

`runs/{run_id}/` typically contains:

- `mlp_final.zip`
- `mlp_best.zip`
- `mlp_vec_normalize.pkl`
- `hierarchical_final.zip`
- `hierarchical_best.zip`
- `training_summary.json`
- Stage-specific logs/checkpoint subfolders

### 8.3 Skip-MLP mode

Use skip mode when you already have an expert:

- Provide `--skip-mlp --expert <path>`.
- Optionally provide `--vec-normalize <path>`.
- Pipeline can auto-detect normalize stats next to expert in common layouts.

## 9. CLI and Task Runner Workflows

### 9.1 Quick task recipes (`justfile`)

Common recipes:

- `just install`
- `just test`
- `just train`
- `just train-quick`
- `just train-hier-only expert=...`
- `just eval`
- `just demo`

### 9.2 Direct script examples

- Full pipeline: `python3 scripts/pipeline.py`
- Stage 1 only: `python3 src/training/train.py --total-steps ... --n-envs ...`
- Stage 2 only: `python3 src/training/train_hierarchical.py --expert ...`

### 9.3 Primary entrypoint

`run.py` provides convenient command routing for:

- training
- evaluation
- demo
- dashboard/live dashboard
- tests
- exploration demo

## 10. Artifacts, Checkpoint Semantics, and Inference Requirements

### 10.1 SB3 model format

Models are saved in SB3 zip format.

A trained policy often requires companion normalization stats:

- Policy zip alone is not always sufficient.
- If training used `VecNormalize`, load matching `vec_normalize.pkl` for correct observation scaling.

### 10.2 Best vs final

- `best` is eval-selected checkpoint.
- `final` is endpoint checkpoint after all timesteps.

For deployment/evaluation, `best` often outperforms `final`, but verify empirically per run.

## 11. Debugging and Validation Strategy

### 11.1 Reward diagnostics

Use component logs and reward diagnostics to answer:

- Which terms dominate?
- Are penalties saturating or exploding?
- Is gait reward active only when command speed is meaningful?

### 11.2 Training health indicators

Monitor:

- Episode reward trend and variance.
- Entropy and action std behavior.
- PPO clip fraction and KL behavior.
- Success/failure pattern around mode transitions.
- Curriculum level distributions (if enabled).
- MoE gate utilization balance.

### 11.3 Common failure patterns

- Observation schema mismatch across env/extractor.
- Missing `vec_normalize.pkl` at inference.
- Reward terms over-penalizing early learning.
- Aggressive termination without sufficient grace windows.
- Jump FSM reward dominating non-jump behavior if mode logic is wrong.

## 12. How to Safely Modify the System

When changing any of these, update all dependent locations:

1. Observation dimensions or ordering.
2. Skill modes and one-hot encoding width.
3. Reward scales and per-mode multipliers.
4. History length assumptions.
5. Policy architecture constants in advanced modules.

Recommended change workflow:

1. Change env/layout first.
2. Update extractor/group indices.
3. Run smoke training (`just train-quick`).
4. Validate reward components and termination stats.
5. Run eval and demo for qualitative verification.

## 13. Suggested End-to-End Usage Patterns

### 13.1 Standard full run

1. `just install`
2. `just test`
3. `just train`
4. `just eval`
5. `just demo`

### 13.2 Fast iteration loop

1. `just train-quick`
2. Check logs and reward components.
3. Adjust one variable group at a time.
4. Re-run quick training.
5. Promote to longer run once stable.

### 13.3 Architecture experimentation

When adjusting transformer/MoE settings:

1. Keep env obs format fixed initially.
2. Benchmark only one architecture hyperparameter change per run.
3. Watch MoE gate usage and PPO stability metrics.
4. Record config deltas in run metadata for reproducibility.

## 14. Reproducibility Checklist

Before comparing runs, lock:

- Code revision.
- XML model and env file versions.
- Reward scales and mode multipliers.
- Observation schema.
- Total steps and PPO hyperparameters.
- Number of environments and hardware characteristics.
- Whether normalization stats were reused or retrained.

Store these in run-specific metadata (`training_summary.json` and stage configs).

## 15. Summary

This codebase implements a complete two-stage locomotion learning stack:

1. Stage 1 MLP PPO expert for robust baseline behavior.
2. Stage 2 hierarchical policy that adds temporal reasoning, structured sensory fusion, MoE specialization, and auxiliary objectives.

The system is modular enough for architecture research, but practical success depends heavily on consistency between environment observation design, wrapper/extractor assumptions, and normalization/checkpoint handling.
