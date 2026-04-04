# Reward v6 — Gait & Training Optimization

## Status: READY FOR TRAINING

## Changes from v5

### Reward Function (cheetah_env.py)
- r_smooth: -0.05 L2 → -0.01 L1 (5× reduction, L1 tolerates cyclic gait)
- r_gait: 2.0 → 3.5 (added stride_freq sub-reward; 4 sub-components)
- r_linvel: 4.0 → 5.0 
- r_dof_vel: -2e-4 → -1e-4 (halved)
- Trot symmetry: abs→sqrt for smoother gradient
- Foot clearance target: 5cm → 8cm
- Walk mode: r_gait mult 1.5→2.0, r_linvel mult 1.0→1.2
- Run mode: r_linvel mult 1.5→2.0, r_gait mult 1.0→1.5, r_smooth mult=0.5

### Training Parameters (train.py)
- LR schedule: min_lr=1e-5 floor (was decaying to 0)
- log_std_init: -0.5 → -1.0 (std≈0.37)
- n_epochs: 5 → 10
- batch_size: 4096 → 2048
- Gradient budget: 160 steps/rollout (was 40) → ~48.8K total at 10M steps

## Expected Improvements
- r_smooth penalty: ~-0.06/step (was -0.15 to -0.19)
- r_gait: higher magnitude due to stride_freq + larger clearance target + higher scale
- Learning should continue through end of training (LR floor)
- std should be learnable throughout (LR floor)
- More gradient updates → better convergence

## Validation
- 39/39 tests passing
- Devil's advocate caught trot symmetry bug (exp kernel rewarded wrong pattern → fixed to sqrt)

## Files Modified
- src/env/cheetah_env.py (reward v6)
- src/training/train.py (training params)
- training_config.json (config update)
- docs/architecture.md (docs update)
- CHANGELOG.md (changelog)
