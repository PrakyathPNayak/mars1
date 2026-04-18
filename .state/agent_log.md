# Agent Log — v31s10g5

## Current State (2025-04-18)
Training v31s10g5 from s10g/600K checkpoint. PID ~898951. ~1.1M steps.

## v31s10g5 Changes
- r_wz_lin weight: 15.0 → 10.0 (reduce free reward for overshoot)
- r_wz_overshoot weight: -2.0 → -6.0 (stronger overshoot penalty)
- r_wz_overshoot computation added in v31s10g2 (commit 0ebeaa2)
- Speed_scale floor 0.12 for yaw (commit 6713705, v31s10g)
- yaw_gain = 0.90

## Evaluation History (v31s10g5)
| Test | 300K | 600K | 1M |
|------|------|------|-----|
| walk_fwd_0.5 | 73%, wz=0.230 | 76%, wz=0.206 | 80%, wz=0.173 |
| walk_fwd_1.0 | 76%, wz=0.031 | 76%, wz=0.026 | 80%, wz=0.010 |
| yaw+0.5 | 108% | 103% | 107% |
| yaw-0.5 | 164% | 166% | 161% |
| fwd+yaw | vx=81%,wz=175% | vx=81%,wz=167% | vx=84%,wz=164% |
| lat+0.3 | 72% | 73% | 68% |
| lat-0.3 | 101% | 99% | 95% |
| fwd+lat | vx=63%,vy=103% | vx=63%,vy=102% | vx=63%,vy=101% |
| run_1.0 | 103% | 103% | 104% |
| run_1.5 | 114% | 113% | 115% |
| crouch | 0.266 | 0.265 | 0.265 |
| jump h_max | 0.563 | 0.568 | 0.574 |

## Key Findings
1. Yaw speed_scale floor (v31s10g) was THE breakthrough — yaw survived past 1M
2. r_wz_lin overshoot modification (v31s10g4) was TOO AGGRESSIVE — killed yaw to 20%
3. Weight reduction approach (v31s10g5) is working: yaw- slowly improving 164→166→161%
4. yaw+ nearly perfect: 108→103→107%
5. Walk forward improving: 73→76→80%, wz drift 0.230→0.206→0.173
6. Crouch regressed from 0.083 (s10g) to 0.265 — investigating

## Failed Approaches
- v31s10g4: Modified r_wz_lin formula to decrease after 120% overshoot → killed yaw (20%)
- v31s10g2/g3: r_wz_overshoot at -2.0 weight → too weak (-0.099 avg penalty)

## Git Commits
- 6713705: v31s10g — yaw speed_scale floor + yaw_gain 0.90
- 0ebeaa2: v31s10g2 — add r_wz_overshoot penalty
- 3a0a684: v31s10g4 — FAILED r_wz_lin modification (reverted)
- ef621a2: v31s10g5 — weight reduction approach (current)

## Next Steps
1. Continue training to 2M+, monitor yaw- convergence
2. If yaw- plateaus at 160%+, increase penalty to -10.0
3. Investigate crouch regression (0.265 vs 0.083)
4. Investigate lat+ asymmetry (68% vs 95%)

---
## v31s6g4 — 500K Eval (rear_scale=1.0 symmetric fix)
Resume from v31s6g3 2M. Code: commit 65b1866.

| Scenario | v31s6g3 2M | v31s6g4 500K | Notes |
|----------|-----------|-------------|-------|
| walk_fwd | 110% | 20% 🔴 | crashed, wz=0.652 drift |
| walk_back | — | 26% | |
| lat_L | 96% | 89% | ok |
| lat_R | 100% | 79% | slight drop |
| yaw_L | 30% | 26% | similar |
| yaw_R | 7% | 20% ✅ | improved! |
| fwd_yaw_L | 109% | 51% | recovering |
| fwd_yaw_R | -6% | 25% ✅ | FIXED wrong direction |
| run_1.0 | 116% | 116% | perfect |
| run_2.0 | 0% | 0% | still dead |
| jump | 0.727 | 0.746 ✅ | new ATB! |
| crouch | 0.081 | 0.281 | still bad |

**Key findings:**
- Symmetry fix WORKING: yaw_R 7%→20%, fwd_yaw_R -6%→+25%
- walk_fwd temporary crash expected — model adapting to symmetric reference
- Jump new ATB 0.746
- Need more training time for walk_fwd recovery

## v31s6g4 — 1M Eval
| Scenario | 500K | 1M | Notes |
|----------|------|-----|-------|
| walk_fwd | 20% | 80% ✅ | RECOVERING! wz drift 0.652→0.010 |
| walk_back | 26% | 30% | |
| lat_L | 89% | 70% | dropped |
| lat_R | 79% | 82% | stable |
| yaw_L | 26% | 25% | stable |
| yaw_R | 20% | 21% | stable |
| fwd_yaw_L | 51% | 59% | improving |
| fwd_yaw_R | 25% | 40% ✅ | improving, symmetry working |
| run_1.0 | 116% | 113% | solid |
| run_2.0 | 0% | 0% | dead |
| jump | 0.746 | 0.447 🔴 | crashed - temporary? |
| crouch | 0.281 | 0.276 | still bad |

Trajectory positive. Walk/yaw recovering. Jump dipped — monitor.

## v31s6g4 — 1.5M Eval ⭐ BEST SYMMETRIC
| Scenario | 500K | 1M | 1.5M | Notes |
|----------|------|-----|------|-------|
| walk_fwd | 20% | 80% | 60% | dipped, wz near zero |
| walk_back | 26% | 30% | 32% | |
| lat_L | 89% | 70% | 91% | recovered |
| lat_R | 79% | 82% | 76% | stable |
| yaw_L | 26% | 25% | 27% | stable |
| yaw_R | 20% | 21% | 27% ✅ | SYMMETRIC with L! |
| fwd_yaw_L | 51% | 59% | 62% | improving |
| fwd_yaw_R | 25% | 40% | 59% ✅✅ | was -6%, now 59%! |
| run_1.0 | 116% | 113% | 111% | solid |
| run_2.0 | 0% | 0% | 0% | dead |
| jump | 0.746 | 0.447 | 0.745 ✅ | recovered from dip |
| crouch | 0.281 | 0.276 | 0.282 | still bad |

**Symmetry fix confirmed.** yaw L=R for first time ever.
**fwd_yaw_R: -6%→59%** in 1.5M steps. Root cause was rear_scale=1.3.
Saved as best_1.5M_symmetric.zip.

## v31s6g4 — 2M Eval ⭐⭐ BEST OVERALL
| Scenario | 500K | 1M | 1.5M | 2M | Notes |
|----------|------|-----|------|----|-------|
| walk_fwd | 20% | 80% | 60% | 84% ✅ | recovering, wz=-0.164 |
| walk_back | 26% | 30% | 32% | 33% | |
| lat_L | 89% | 70% | 91% | 85% | stable |
| lat_R | 79% | 82% | 76% | 76% | stable |
| yaw_L | 26% | 25% | 27% | 22% | slight dip |
| yaw_R | 20% | 21% | 27% | 30% ✅ | improving |
| fwd_yaw_L | 51% | 59% | 62% | 64% | steady |
| fwd_yaw_R | 25% | 40% | 59% | 84% ✅✅✅ | was -6%, now 84%! |
| run_1.0 | 116% | 113% | 111% | 111% | solid |
| run_2.0 | 0% | 0% | 0% | 0% | dead |
| jump | 0.746 | 0.447 | 0.745 | 0.760 ✅ | NEW ATB! |
| crouch | 0.281 | 0.276 | 0.282 | 0.276 | STUCK |

Saved as best_2M_yawR84_jump760.zip.
Next: investigate crouch stuck, boost pure yaw, let training continue.

## v31s6g4 — 2M CORRECTED EVAL (eval bug fixed)
**BUG FOUND**: `set_command()` called before `venv.reset()` wiped height ramp.
Crouch target was always reset to HEIGHT_DEFAULT. Fix: set command AFTER reset.

| Scenario | Old eval | Corrected | Notes |
|----------|---------|-----------|-------|
| walk_fwd | 84% | 84% | no change |
| walk_back | 33% | 34% | no change |
| lat_L | 85% | 86% | no change |
| lat_R | 76% | 77% | no change |
| yaw_L | 22% | 23% | no change |
| yaw_R | 30% | 29% | no change |
| fwd_yaw_L | 64% | 65% | no change |
| fwd_yaw_R | 84% | 80% | stochastic |
| run_1.0 | 111% | 110% | no change |
| jump | 0.760 | 0.446 | variable |
| crouch | 0.276 | **0.085** ✅✅✅ | WAS ALWAYS WORKING! |
| crouch_walk | — | **0.105** ✅ | vx=0.413 while crouched |

**Crouch was never broken — eval was broken.** Robot crouches to 0.085m.

## v31s10g6 — Moderate wz overshoot penalty (walk -10, run -5, deadzone 1.15x)
Started from s10g5/2M. Commit 19acedf (CLEAN — previous 6ba0dd6 was contaminated by parallel session).

### Evaluation Results
| Test | s10g5/2M (baseline) | s10g6/300K | s10g6/1M | s10g6/2M |
|------|---------------------|-----------|---------|---------|
| walk_fwd_0.5 | 84% wz=.125 | 84% wz=.124 | 87% wz=.056 | **98%** wz=.031 |
| walk_fwd_1.0 | 84% wz=-.040 | 84% wz=-.019 | 87% wz=-.129 | **91%** wz=.045 |
| yaw+0.5 | 104% | 111% | 101% | **96%** |
| yaw-0.5 | 173% | 156% | 164% | **149%** |
| fwd+yaw | vx=84%,wz=151% | vx=69%,wz=132% | vx=74%,wz=119% | vx=77%,wz=**120%** |
| lat+0.3 | 67% | 66% | 63% | **69%** |
| lat-0.3 | 98% | 95% | 98% | **93%** |
| run_1.0 | 105% | 104% | 105% | **108%** |
| run_1.5 | 117% | 11% | 74% | **75%** |
| crouch | 0.264 | 0.258 | 0.259 | **0.261** |
| jump | 0.586 | 0.592 | 0.597 | **0.600** |

### Key Findings
- yaw- improving: 173% → 149% (best ever!). Gradient=2.9 working.
- walk_fwd_0.5 near perfect at 98%
- yaw+ converged to 96% (excellent)
- fwd+yaw wz improved 151% → 120%
- run_1.5 still recovering (75%, was 117%)
- Training continues to 5M, PID 3073537

## v31s6g4 — 2.5M Eval
| Scenario | 2M | 2.5M | Notes |
|----------|-----|------|-------|
| walk_fwd | 84% | 26% 🔴 | crashed again |
| lat_L | 86% | 89% | stable |
| lat_R | 77% | 82% | improved |
| yaw_L | 23% | -0% 🔴🔴 | COLLAPSED (h=0.068, fell) |
| yaw_R | 29% | 36% ✅ | improving |
| fwd_yaw_L | 65% | 71% ✅ | improving |
| fwd_yaw_R | 80% | 100% ✅✅✅ | PERFECT |
| run_1.0 | 110% | 112% | solid |
| jump | 0.446 | 0.754 | variable |
| crouch | 0.085 | 0.085 | perfect |
| crouch_walk | 0.105 | 0.107 | perfect |

Multi-task interference: model trades walk_fwd for fwd_yaw. Oscillating.
Best balanced: 2M checkpoint. Best yaw_R: 2.5M.
If walk_fwd doesn't recover by 3M, may need to stop and analyze.

## v31s6g4 — 3M Eval + Yaw Rate Sweep
| Scenario | 2M | 3M | Notes |
|----------|-----|-----|-------|
| walk_fwd | 84% | 90% ✅ | recovered |
| lat_L | 86% | 94% ✅ | best ever |
| lat_R | 77% | 84% ✅ | improving |
| fwd_yaw_L | 65% | 79% ✅ | improving |
| fwd_yaw_R | 80% | 85% ✅ | solid |
| run_1.0 | 110% | 112% | solid |
| jump | varied | 0.754 | good |
| crouch | 0.085 | 0.083 ✅ | getting deeper |
| crouch_walk | 0.105 | 0.098 ✅ | lower |

**Yaw rate sweep (pure yaw, 3M model):**
| wz_cmd | yaw_L | yaw_R |
|--------|-------|-------|
| ±0.5 | 80% | 99% |
| ±0.8 | 55% | 73% |
| ±1.0 | 42% (fell once) | 48% |
| ±1.5 | 2% (collapsed) | 23% |

**yaw_L NOT broken** — eval at 1.5 was too aggressive. Works well at 0.5-0.8 rad/s.
Model is balanced and improving across all skills. Continue training.

## v31s6g4 — 4M Eval ⭐⭐⭐ BEST YAW + JUMP
| Scenario | 3M | 4M | Notes |
|----------|-----|-----|-------|
| walk_fwd | 90% | 89% | stable |
| walk_back | 35% | 36% | stable |
| lat_L | 94% | 84% | slight drop |
| lat_R | 84% | 77% | slight drop |
| yaw_L_0.5 | 80% | 93% ✅✅ | near perfect! |
| yaw_R_0.5 | 99% | 103% ✅✅ | perfect! |
| yaw_L_1.0 | 42% | 43% | symmetric with R |
| yaw_R_1.0 | 48% | 45% | symmetric with L |
| fwd_yaw_L | 79% | 52% 🔴 | oscillation |
| fwd_yaw_R | 85% | 92% ✅ | excellent |
| run_1.0 | 112% | 109% | solid |
| jump | 0.754 | 0.762 ✅ | NEW ATB |
| crouch | 0.083 | 0.080 ✅ | ground level |
| crouch_walk | 0.098 | 0.093 | h good, vx=0.083 slow |

Best checkpoint for: yaw symmetry, jump height, crouch depth.
Saved as best_4M_yaw93_jump762.zip.

## v31s10g7 — Asymmetric yaw_gain (0.90 pos, 0.60 neg) — YAW- SOLVED
Commit 5844fc7. Started from s10g6/2M.

### Results — YAW- OVERSHOOT FIXED
| Test | s10g5/2M (old baseline) | s10g7/300K | s10g7/1M | s10g7/2M |
|------|------------------------|-----------|---------|---------|
| walk_fwd_0.5 | 84% | 96% | 98% | **94%** |
| walk_fwd_1.0 | 84% | 91% | 94% | **90%** |
| yaw+0.5 | 104% | 94% | 89% | **88%** |
| yaw-0.5 | **173%** | 91% | 98% | **96%** |
| fwd+yaw+ | vx=84%,wz=151% | vx=79%,wz=116% | vx=75%,wz=111% | vx=74%,wz=**108%** |
| fwd+yaw- | N/A | N/A | N/A | vx=91%,wz=**100%** |
| lat+0.3 | 67% | 68% | 64% | **71%** |
| lat-0.3 | 98% | 93% | 94% | **96%** |
| run_1.0 | 105% | 108% | 108% | **107%** |
| run_1.5 | 117% | 114% | 104% | **103%** |
| crouch | 0.264 | 0.260 | 0.258 | **0.258** |
| jump | 0.586 | 0.607 | 0.620 | **0.622** |

### Remaining Issues
- yaw+ trending down (104→88%) — may need gain bump 0.90→0.95
- lat+ weak at 71% (lat- is 96%) — asymmetry investigation needed
- crouch stuck at 0.258 — user wants near ground level
- Training continues to 5M, PID 4005902

## v31s6g4 — 5M FINAL EVAL ⭐⭐⭐⭐
| Scenario | 3M | 4M | 5M | Notes |
|----------|-----|-----|-----|-------|
| walk_fwd | 90% | 89% | 87% | stable |
| walk_back | 35% | 36% | 24% | dropped |
| lat_L | 94% | 84% | 67% | regressed |
| lat_R | 84% | 77% | 80% | stable |
| yaw_L_0.5 | 80% | 93% | 101% ✅✅ | PERFECT |
| yaw_R_0.5 | 99% | 103% | 121% ✅ | overshooting |
| yaw_L_1.0 | 42% | 43% | 31% | dropped |
| yaw_R_1.0 | 48% | 45% | 42% | stable |
| fwd_yaw_L | 79% | 52% | 94% ✅✅ | RECOVERED |
| fwd_yaw_R | 85% | 92% | 110% ✅✅ | best ever |
| run_1.0 | 112% | 109% | 109% | solid |
| jump | 0.754 | 0.762 | 0.767 ✅ | ATB! |
| crouch | 0.083 | 0.080 | 0.083 | ground level |
| crouch_walk | 0.098 | 0.093 | 0.106 | vx=0.430! |

v31s6g4 training COMPLETE. 5M steps total.
**Best checkpoints:**
- 3M: best balanced (walk_fwd=90%, lat_L=94%)
- 4M: best yaw symmetry (L=93%, R=103%)
- 5M: best fwd_yaw (L=94%, R=110%), best jump (0.767)

**Remaining weaknesses:** lat_L regression, walk_back low, run_2.0 dead.
**Next:** v31s6g5 — boost lateral sampling, extend run range, resume from 3M balanced.
