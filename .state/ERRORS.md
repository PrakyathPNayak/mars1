# Error Log
All errors encountered and their resolutions.

## Default Stance Joint Mismatch
- **Error**: Robot fell at step 35 with default stance [0, -0.8, 1.6] per leg
- **Cause**: Knee joint range in MJCF is [-2.697, -0.524] (negative), but stance used positive knee angles
- **Fix**: Changed DEFAULT_STANCE to [0, 0.7, -1.4] per leg — robot stays stable 500+ steps at h=0.31m
