# Mini Cheetah XML: Ours vs Official MIT Parameters

Source: [`mit-biomimetics/Cheetah-Software` — `MiniCheetah.h`](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/common/include/Dynamics/MiniCheetah.h)

## Summary

Our original `mini_cheetah.xml` has **significant differences** from the official MIT
parameters defined in the Cheetah-Software C++ codebase. A corrected model has been
created at `assets/mini_cheetah_official.xml`.

> **Note**: The trained policy (`checkpoints/best/best_model.zip`) was trained on the
> original XML and will **not** work correctly with the corrected model. Retraining would
> be required.

---

## Critical Differences

| Parameter | Our XML | Official MIT | Impact |
|-----------|---------|-------------|--------|
| **Body mass** | 6.0 kg | **3.3 kg** | Robot almost 2x too heavy — dramatically changes dynamics |
| **Body inertia** | diag(0.05, 0.15, 0.15) | **diag(0.0113, 0.0362, 0.0427)** | Wrong rotational dynamics, 3-4x too large |
| **Shin mass** | 0.280 kg | **0.064 kg** | 4.4x too heavy — enormous effect on leg swing dynamics |
| **Shin length** | 0.175 m | **0.195 m** | 11% shorter legs, affects stride and kinematics |
| **Total robot mass** | ~12.1 kg | **~8.3 kg** | 46% heavier than real robot |
| **Joint damping** | 0.5 Nm·s/rad | **0.01 Nm·s/rad** | 50x too much damping — masks dynamics |

## Moderate Differences

| Parameter | Our XML | Official MIT | Notes |
|-----------|---------|-------------|-------|
| **Body box size** | half(0.20, 0.05, 0.025) | half(0.19, 0.049, 0.05) | Height half-extent wrong (0.025 vs 0.05) |
| **Abad position (y)** | ±0.062 | **±0.049** | Uses link length, not body half-width |
| **Hip offset from abad** | ±0.035 | **±0.062** | Should be abad link length (0.062) |
| **Total leg offset (y)** | 0.097 m | **0.111 m** | Narrower stance in our model |
| **Knee torque limit** | ±17 Nm | **±28 Nm** | Knee has 9.33 gear ratio vs abad/hip at 6 |
| **Abad/hip torque limit** | ±17 Nm | **±18 Nm** | Close, but slightly different |
| **Hip COM** | (0, 0, -0.1045) | (0, ±0.016, -0.02) | Ours at mid-link, official near joint |
| **Foot mass** | 0.06 kg | ~0.01 kg (synthetic) | MIT has no separate foot body |

## Minor Differences

| Parameter | Our XML | Official MIT | Notes |
|-----------|---------|-------------|-------|
| **Abad inertia** | diag(0.001, 0.001, 0.001) | Full 3x3 tensor from CAD | Small body, minor effect |
| **Hip inertia** | diag(0.002, 0.002, 0.0003) | Full 3x3 tensor from CAD | Off-diagonals present |
| **Knee inertia** | diag(0.001, 0.001, 0.00005) | Rotated tensor from CAD | Small body |
| **Knee COM** | (0, 0, -0.0875) | (0, 0, -0.061) | Minor |
| **FR/FL y-sign** | FR at +y, FL at -y | FR at -y, FL at +y | Swapped sides (symmetric, no functional effect for RL) |

## Changes Required for Environment Code

If switching to `mini_cheetah_official.xml`, these environment changes would be needed:

1. **`cheetah_env.py`**: Update `max_torque` from 17 to handle different per-joint limits (18 for abad/hip, 28 for knee), or use the actuator limits built into the XML
2. **`cheetah_env.py`**: May need to adjust `r_height` target (0.30 m) — the lighter robot with different leg geometry may have a different natural standing height
3. **`cheetah_env.py`**: Retune reward weights — lighter robot will produce much less torque penalty
4. **`DEFAULT_STANCE`**: May need adjustment for the different leg geometry
5. **Retrain policy**: The trained model is invalidated by model changes

## Files

- `assets/mini_cheetah.xml` — Original (used for current trained policy)
- `assets/mini_cheetah_official.xml` — Corrected to match official MIT parameters
