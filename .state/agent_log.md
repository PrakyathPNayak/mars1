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

### s10g8 Evaluation Progress
| Test | 300K | 1M | 2M | Trend |
|------|------|-----|-----|-------|
| walk_fwd_0.5 | 103% | 99% | 100% | ✅ stable |
| walk_fwd_1.0 | 95% | 96% | 96% | ✅ stable |
| yaw+0.5 | 83% | 83% | 88% | ✅ recovering |
| yaw-0.5 | 109% | 106% | 110% | ⚠️ oscillating |
| lat+0.3 | 103% | 100% | 98% | ✅✅ SOLVED |
| lat-0.3 | 95% | 95% | 96% | ✅ stable |
| run_0.5 | 112% | 112% | 113% | ✅ SOLVED |
| run_1.0 | 107% | 108% | 108% | ✅ stable |
| run_1.5 | 76% | 118% | 74% | ⚠️ oscillating |
| run_2.0 | N/A | 0% | 0% | ❌ broken |
| crouch | 0.099 | 0.098 | 0.099 | ✅ SOLVED |
| jump | 0.635 | 0.637 | 0.647 | ✅ best ever |

### KEY: Remaining issues
- run_1.5 oscillating (76→118→74%) — policy can't stably maintain 1.5 m/s
- run_2.0 still 0% — max amplitude too aggressive
- yaw- slight overshoot 106-110% — within deadzone (1.15x)

### v31s10g9 — FAILED EXPERIMENT
- action_scale 0.15→0.20 for run: globally destabilized policy
- After 1M steps, run modes still overshooting (127%) or falling
- yaw- regressed from 110% to 125%
- LESSON: never change action_scale when resuming from trained checkpoint

### v31s10g10 — Run speed_scale cap at 0.20
Changes from s10g8:
- speed_scale cap = 0.20 for run (prevents unstable reference above 0.22)
- speed_scale floor = 0.12 for run (was 0.15 — less overshoot at low speeds)
- action_scale unchanged at 0.15

Zero-action verification:
| run_cmd | s10g8 | s10g10 |
|---------|-------|--------|
| 0.3 | 168% | 67% |
| 0.5 | 117% | 42% |
| 1.0 | 95% | 95% |
| 1.2 | FALLS | 81% |
| 1.5 | FALLS/137% | 69% |
| 2.0 | 0% | 56% |

ALL run speeds now STABLE at zero-action. Cap works.

### s10g10 / 600K Evaluation
| Test | Result | vs s10g8/2M |
|------|--------|-------------|
| walk_fwd_0.5 | 93% | ↓ adapting |
| walk_fwd_1.0 | 109% | ↑ |
| yaw+ | 87% | same |
| yaw- | 102% | ✅ FIXED (was 110%) |
| lat+ | 95% | same |
| lat- | 96% | same |
| run_0.5 | 50% | ↓ floor reduced |
| run_1.0 | 121% | ↑ overshooting |
| run_1.2 | 118% | ✅✅ WAS FALLING |
| run_1.5 | FALLS | needs time |
| run_2.0 | FALLS | needs time |
| crouch | 0.067 | ✅ (was 0.099) |
| jump | 0.644 | same |

KEY: run_1.2 UNLOCKED — first time working ever!
Training PID 1465538, waiting for 2M evaluation.

## v31s6g6 — fwd_yaw reward rebalance (2025-04-19)

### Problem
fwd_yaw tracking: -29% vx (robot walks BACKWARD while turning perfectly).
Root cause: wz reward total=27 (track=12 + lin=15) vs vx total=11 (track=8 + lin=3).
Model rationally ignores forward velocity — 2.5x more reward from yaw.

### Fix (commit 5bde947)
- r_vx_track_walk: 8→14, r_vx_lin: 3→4 (vx total: 11→18)
- r_wz_lin: 15→8 (wz total: 27→20)  
- r_wz_overshoot: -2→-6 (compensate reduced wz_lin)
- Added fwd+yaw sampling category (10% of walk)
- New balance: vx=18 vs wz=20 (was 11 vs 27)

### v31s6g5 final eval @ 3.1M (OLD rewards, pre-fix baseline)
| Scenario | Tracking |
|----------|----------|
| walk_fwd | 41% (regressed from 73% at 3M) |
| walk_back | 72% |
| lat_L | 84% |
| lat_R | 65% |
| yaw_L | 66% (regressed from 98% at 3M) |
| yaw_R | 34% |
| fwd_yaw_L | vx:-29%, wz:94% THE PROBLEM | 
| fwd_yaw_R | vx:33%, wz:35% |
| run_1.0 | 106% |
| run_2.0 | 0% (collapsed) |

### Training
- Resume from: v31s6g5 best_2M_run81_yaw90.zip (best all-around)
- PID: 2751368, started 19:16
- Watch: fwd_yaw recovery, yaw maintenance, run_2.0 stability


### v31s6g6b — yaw_gain fix for fwd+yaw (second training run)

Key discovery: reference trajectory produces vx=0% + wz=105% for fwd+yaw.
Yaw differential stride (L=0.55x, R=1.45x) destroys forward force.
Fix: reduce yaw_gain when BOTH fwd and yaw commanded:
  fwd+yaw+: 0.90→0.40 (ref: vx=26%, wz=66%)
  fwd+yaw-: 0.60→0.50 (ref: vx=48%, wz=33%)

Also committed: reward rebalance from v31s6g6 (vx=18, wz=20)

Resumed from v31s6g6 1M checkpoint (yaw=95/102%, run_2.0=72%)
PID: 515988, started 20:13


## v31s6g7 — re-applied reward rebalance + restart (2025-04-18)

### Problem discovered
Parallel session (commit d814713, v31s10g11) SILENTLY REVERTED our reward fix from 5bde947:
  - r_vx_track_walk: 14→8 (reverted)
  - r_vx_lin: 4→3 (reverted)
  - r_wz_lin: 8→10 (reverted)
v31s6g6b trained 580K steps with WRONG weights (vx=11, wz=22 — 2.0x ratio).
This explains why fwd_yaw was still -30% — the reward imbalance was NEVER actually fixed!

### Fix (commit bf1b0a6)
Re-applied: vx_track=14, vx_lin=4, wz_lin=8 (vx=18, wz=20, ratio 1.1x)
Tests: 10/10 pass.

### Reference trajectory analysis
Also tested rear_scale fix (1.0 for non-forward), but it HURT yaw effectiveness:
  yaw_L: 92% → 74% (20% drop). Not worth small forward-drift reduction.
  Decision: keep rear_scale=1.3 for all walk. Policy handles drift via penalties.

### Training started
- PID: 272311
- Resume from: v31s6g6 1M checkpoint (trained with CORRECT weights)
- Has BOTH fixes: reward rebalance (vx=18, wz=20) + yaw_gain reduction for fwd+yaw
- This is the FIRST training with BOTH fixes applied from start
- Target: 5M steps
- Will eval at 500K, 1M, 2M, 3M

### Baseline (v31s6g6 1M — our starting point)
| Metric | Value |
|--------|-------|
| walk_fwd | 56% |
| yaw_L | 95% |
| yaw_R | 102% |
| fwd_yaw_L vx | -27% |
| run_2.0 | 72% |
| jump | 0.767m |
| crouch | 0.087m |

### Key question: will fwd_yaw recover with correct weights?
Previously damaged by wz=27 (2.5x bias). Now training with vx=18 vs wz=20 (1.1x).
Starting from checkpoint that never saw wz=27 weights (v31s6g6 1M was correctly trained).

## s10g11 Evaluation — 2M steps (total ~4.4M from scratch)

### Eval bug fix
- Was using `raw.gait_mode = "walk"` — WRONG. Env uses `command_mode`, not `gait_mode`.
- Setting `gait_mode` just creates unused Python attribute. Observation always saw "stand" mode.
- Fix: use `raw.set_command(vx, vy, wz, mode, height)` which sets both `command` AND `command_mode`.

### Results comparison
| Test | s10g8/2.4M | s10g11/1.2M | s10g11/2M |
|------|-----------|-------------|-----------|
| walk_fwd_0.5 | 100% | 101% | 102% ✅ |
| walk_fwd_1.0 | 96% | 100% | 109% ⚠️ overshoot |
| yaw+0.5 | 88% | 89% | 89% stuck |
| yaw-0.5 | 110% | 109% | 109% still overshoot |
| lat+0.3 | 98% | 98% | 102% ✅ |
| lat-0.3 | 96% | 97% | 94% ✅ |
| run_0.5 | 113% | 117% | 116% ✅ |
| run_0.8 | ~100% | 98% | 100% ✅ perfect |
| run_1.0 | 108% | 124% | 123% ⚠️ overshoot |
| run_1.2 | FALLS | 123% | 123% ✅✅ FIXED |
| run_1.5 | 74% osc | FALLS | FALLS ❌ |
| run_2.0 | 0% | FALLS | FALLS ❌ |
| crouch | 0.099 | 0.081 | 0.082 ✅ |
| jump | 0.647 | 0.649 | 0.391 ❌ REGRESSED |

### Analysis
- run_1.2 FIXED — was falling, now 123%. Speed cap working.
- Jump REGRESSED 0.649→0.391 between 1.2M and 2M. Concerning.
- run_1.0/1.2 overshoot ~123% — both hit speed_scale=0.20 cap, policy adds too much residual.
- run_1.5/2.0 still fall — beyond action_scale capability with capped reference.
- yaw+ stuck at 89% — no improvement from s10g8.

### Next: wait for 3M, check if jump recovers or continues decline.

### v31s6g7 eval @ 500K — FWD_YAW FIXED!

| Scenario | g6b 500K (wrong) | g7 500K (correct) | Change |
|----------|----------|----------|---------|
| walk_fwd | 77% | 71% | -6% (will improve) |
| walk_back | 67% | 69% | +2% |
| lat_L | 140% | 171% | +31% (overshooting) |
| lat_R | 120% | 117% | -3% |
| yaw_L | 110% | 98% | -12% (symmetric!) |
| yaw_R | 93% | 98% | +5% (symmetric!) |
| fwd_yaw_L vx | -30% | **+80%** | **+110pp!!!** |
| fwd_yaw_L wz | ? | 85% | NEW |
| fwd_yaw_R vx | -25% | **+76%** | **+101pp!!!** |
| fwd_yaw_R wz | ? | 106% | NEW |
| run_1.0 | 97% | 95% | -2% |
| run_2.0 | 74% | 74% | 0% |
| jump | 0.609m | 0.768m | +0.159m |
| crouch | 0.088m | 0.089m | +0.001m |

KEY RESULT: fwd_yaw was -30% (backward) → +80% (forward). 
The reward rebalancing (vx=18 vs wz=20, was 11 vs 22) is THE fix.
Combined with yaw_gain reduction for fwd+yaw → policy can track both axes.

Remaining issues:
- lat_L overshooting 171% — may need vy_overshoot penalty increase
- walk_fwd at 71% — expect improvement with more training
- crouch 0.089m — close to 0.08 target, needs time

### v31s6g7 eval @ 1M — fwd_yaw CONFIRMED FIXED

| Scenario | g7 500K | g7 1M | Δ |
|----------|---------|-------|---|
| walk_fwd | 71% | 75% | +4% ↑ |
| walk_back | 69% | 75% | +6% ↑ |
| lat_L | 171% | 170% | -1% (still overshooting) |
| lat_R | 117% | 113% | -4% ↓ |
| yaw_L | 98% | 101% | +3% |
| yaw_R | 98% | 92% | -6% |
| fwd_yaw_L vx | 80% | 77% | -3% (stable) |
| fwd_yaw_L wz | 85% | 99% | +14% ↑↑ |
| fwd_yaw_R vx | 76% | 74% | -2% (stable) |
| fwd_yaw_R wz | 106% | 93% | -13% (converging) |
| run_1.0 | 95% | 96% | +1% |
| run_2.0 | 74% | 75% | +1% |
| jump | 0.768m | 0.764m | stable |
| crouch | 0.089m | 0.088m | -0.001m |

Analysis:
- fwd_yaw is FIXED and STABLE. vx=77%/74%, wz=99%/93%.
- walk improving (75%/75%). run stable (96%/75%).
- lat_L still overshooting at 170%. Reward math says 100% is optimal.
  Policy should converge — tracking penalty at 170% = 1.77/step.
- yaw_R slightly low (92%). Minor L/R asymmetry.
- Continuing training. Next eval at 2M.

## s10g11 @ 3M — Jump recovered, overshoot growing, yaw+ declining

| Test | s10g8 | 1.2M | 2M | 3M | Trend |
|------|-------|------|----|----|-------|
| walk_fwd_0.5 | 100% | 101% | 102% | 110% | ⚠️ overshoot rising |
| walk_fwd_1.0 | 96% | 100% | 109% | 111% | ⚠️ |
| yaw+0.5 | 88% | 89% | 89% | 83% | ❌ declining |
| yaw-0.5 | 110% | 109% | 109% | 114% | ⚠️ overshoot |
| lat+0.3 | 98% | 98% | 102% | 108% | ⚠️ |
| lat-0.3 | 96% | 97% | 94% | 103% | ok |
| run_0.8 | ~100% | 98% | 100% | 101% | ✅ |
| run_1.0 | 108% | 124% | 123% | 123% | ⚠️ stable overshoot |
| run_1.2 | FALLS | 123% | 123% | 120% | ✅ FIXED |
| run_1.5 | 74% | FALLS | FALLS | FALLS | ❌ |
| crouch | 0.099 | 0.081 | 0.082 | 0.082 | ✅ |
| jump | 0.647 | 0.649 | 0.391 | 0.647 | ✅ recovered |

### Analysis
- Jump dip at 2M was transient. Back to 0.647. Good.
- Systematic overshoot growing: vx, vy, wz all overshooting more with training.
- Cause: r_vx_lin (monotonic gradient) + no vx overshoot penalty → policy learns to always go faster.
- yaw+ declining: 89→83%. Training pulling policy toward locomotion, yaw+ signal too weak.
- 1.2M checkpoint may be best all-around (less overshoot, yaw+ 89%).
- Next iteration needs: vx/vy overshoot penalty (like r_wz_overshoot).

### Letting training continue to 5M. Will pick best checkpoint after.

## v31s6g7 @ 1.5M Eval

| Scenario | 1M | 1.5M | Trend |
|----------|-----|------|-------|
| walk_fwd | 75% | 85% | ↑↑ great |
| walk_back | 75% | 76% | stable |
| lat_L | 170% | 159% | ↓ converging |
| lat_R | 113% | 122% | ↑ slight diverge |
| yaw_L | 101% | 112% | ↑ overshooting |
| yaw_R | 92% | 79% | ↓↓ CONCERNING |
| fwd_yaw_L vx | 77% | 82% | ↑ |
| fwd_yaw_L wz | 99% | 132% | ↑↑ overshooting |
| fwd_yaw_R vx | 74% | 85% | ↑↑ |
| fwd_yaw_R wz | 93% | 91% | stable |
| run_1.0 | 96% | 100% | PERFECT |
| run_2.0 | 75% | 77% | stable |
| jump | 0.764m | 0.760m | stable |
| crouch | 0.088m | 0.088m | stuck |

Walk fwd big jump. fwd_yaw vx both up. run_1.0 perfect.
YAW ASYMMETRY growing: L=112% vs R=79% (33pp gap). Watch at 2M.
fwd_yaw_L wz overshooting 132%. lat_R still 122%.
Crouch stuck 0.088m. Jump stable ~0.76m.

## v31s6g8 — Yaw Symmetry Fix

**Changes:**
1. Pure yaw_gain: 0.90/0.60 → 0.80 symmetric (fixes 33pp gap at 1.5M)
2. fwd+yaw yaw_gain: restored conditional with 0.35 symmetric
3. Walk reward: re-applied vx=18, wz=20 (parallel session reverted AGAIN in dd30a62)
4. Kept: parallel session overshoot deadzone 1.10, walk penalties 5/3

**Commit:** 7b3a354, pushed to origin
**Resume from:** v31s6g7 1.5M checkpoint
**Training PID:** 1218218, target 5M steps

**Expected yaw at 500K:**
- yaw_L: 112% * (0.80/0.90) ≈ 100% (from linear scaling)
- yaw_R: 79% * (0.80/0.60) ≈ 105% (from linear scaling)
- fwd_yaw_L wz: should reduce from 132% with gain 0.35 (was 0.40)

## v31s6g8 @ 500K — YAW SYMMETRY FIX CONFIRMED

| Scenario | g7@1.5M | g8@500K | Change |
|----------|---------|---------|--------|
| walk_fwd | 85% | 85% | stable |
| walk_back | 76% | 82% | ↑ |
| lat_L | 159% | 149% | ↓ converging |
| lat_R | 122% | 133% | ↑ slightly |
| yaw_L | 112% | **98%** | ↓↓ FIXED |
| yaw_R | 79% | **96%** | ↑↑ FIXED |
| fwd_yaw_L vx | 82% | 84% | stable |
| fwd_yaw_L wz | 132% | **90%** | ↓↓ FIXED |
| fwd_yaw_R vx | 85% | 81% | slight drop |
| fwd_yaw_R wz | 91% | 89% | stable |
| run_1.0 | 100% | 100% | PERFECT |
| run_2.0 | 77% | 78% | stable |
| jump | 0.760m | 0.772m | ↑ |
| crouch | 0.089m | 0.089m | stable |

**KEY WINS:**
- Yaw gap: 33pp → 2pp. L=98%, R=96%. SYMMETRIC.
- fwd_yaw wz: 132%→90% (L), 91%→89% (R). Both near target.
- fwd_yaw symmetric: L vx=84%/wz=90%, R vx=81%/wz=89%.
- walk_back improved 76%→82%.

**Remaining issues:**
- lat_L=149%, lat_R=133% — still overshooting, need time to converge
- crouch=0.089m — stuck, may need separate attention
- All other metrics stable or improving

**Next:** monitor at 1M, 2M. Watch lateral convergence.

## v31s6g8 @ 1M — Yaw Asymmetry RETURNING

| Scenario | 500K | 1M | Trend |
|----------|------|-----|-------|
| walk_fwd | 85% | 90% | ↑↑ |
| walk_back | 82% | 78% | ↓ slight |
| lat_L | 149% | 149% | stable (overshoot) |
| lat_R | 133% | 130% | ↓ converging |
| yaw_L | 98% | 103% | ↑ slight |
| yaw_R | 96% | 83% | ↓↓ CONCERNING |
| fwd_yaw_L vx | 84% | 81% | stable |
| fwd_yaw_L wz | 90% | 126% | ↑↑ overshoot |
| fwd_yaw_R vx | 81% | 83% | stable |
| fwd_yaw_R wz | 89% | 109% | ↑ |
| run_1.0 | 100% | 101% | stable |
| run_2.0 | 78% | 78% | stable |
| jump | 0.772m | 0.773m | stable |
| crouch | 0.089m | 0.088m | minimal |

**Yaw asymmetry returning**: L=103% vs R=83% (20pp gap, was 2pp at 500K).
Symmetric gains (0.80) helped at 500K but policy adapting back to asymmetry.

**Root cause analysis**:
- Reference trajectory IS symmetric (verified mathematically)
- Balance corrections are OFF during yaw (trim_fade=0 at wz=0.5)
- PHYSICAL model is left-right symmetric
- Most likely: observation encoding NOT symmetric (FR,FL,RR,RL ordering).
  Network weights develop asymmetric features without bilateral symmetry enforcement.
  Papers (Walk These Ways, legged_gym) use symmetric losses or mirror augmentation.

**Decision**: Wait for 1.5M and 2M. If gap keeps growing, implement mirror augmentation.

**ALSO**: Parallel session committed AGAIN (36443ed) — reverted our yaw fix.
Our training PID 1218218 running with correct code (7b3a354).

## v31s6g8 @ 1.5M — YAW RECOVERING, WALK STRONG

| Scenario | 500K | 1M | 1.5M | Trend |
|----------|------|-----|------|-------|
| walk_fwd | 85% | 90% | 93% | ↑↑ best ever |
| walk_back | 82% | 78% | 89% | ↑↑ bounce |
| lat_L | 149% | 149% | 147% | ↓ slow |
| lat_R | 133% | 130% | 134% | ↔ |
| yaw_L | 98% | 103% | 98% | ✅ stable |
| yaw_R | 96% | 83% | 90% | ↑ recovering |
| fwd_yaw_L vx | 84% | 81% | 81% | stable |
| fwd_yaw_L wz | 90% | 126% | 120% | ↓ from peak |
| fwd_yaw_R vx | 81% | 83% | 83% | stable |
| fwd_yaw_R wz | 89% | 109% | 118% | ↑ |
| run_1.0 | 100% | 101% | 102% | stable |
| run_2.0 | 78% | 78% | 80% | ↑ |
| jump | 0.772m | 0.773m | 0.775m | stable |
| crouch | 0.089m | 0.088m | 0.088m | stuck |

**Yaw gap trajectory**: 2pp→20pp→8pp. Peaked at 1M, now recovering.
Symmetric gains working. Policy needed adaptation time.

**fwd_yaw wz**: Now symmetric! L=120%, R=118%. Both overshooting.
The wz_overshoot penalty (-10.0) should gradually correct this.

**Decision**: Continue training. All trends positive. Monitor at 2M, 2.5M.

## s10g12 FAILED — explicit penalty too weak
- Tightened deadzone 1.20→1.10, boosted weights
- walk_fwd_0.5 STILL 114% overshoot (penalty only 0.095/step = 1% of reward)
- Explicit penalty approach fundamentally too weak vs 13/step total reward

## s10g13 @ 1.5M — asymmetric tracking sigma
| Test | s10g11/1.2M | s10g13/1.5M | Delta |
|------|-------------|-------------|-------|
| walk_fwd_0.5 | 101% | 101% | ✅ HELD (was growing to 110%!) |
| walk_fwd_1.0 | 100% | 110% | ⚠️ still overshoot |
| yaw+0.5 | 89% | 86% | ⚠️ |
| yaw-0.5 | 109% | 121% | ❌ REGRESSED |
| lat+0.3 | 98% | 85% | ❌ declined |
| run_1.0 | 124% | 126% | ⚠️ unchanged |
| run_1.2 | 123% | 125% | ⚠️ unchanged |
| crouch | 0.081 | 0.079 | ✅ |
| jump | 0.649 | 0.395 | ❌ dip (may recover) |

Analysis: asymmetric sigma helps for walk_fwd_0.5 but NOT run mode.
Run overshoot likely from reference-residual mismatch (speed_scale cap).
yaw- regression from reward balance shift. Continuing to 3M.

## v31s6g9 — Mirror Augmentation for Bilateral Symmetry

**Started**: Resumed from v31s6g8 @ 2.2M, PID 1251744
**Key change**: Mirror augmentation (50% per episode)
- Swap FR↔FL, RR↔RL in obs/actions, negate abd
- Negate vy, wz, roll, lateral gravity, CPG phase
- Flip heightmap columns laterally
- Forces policy to learn symmetric behavior

**v31s6g8 final eval @ 2.2M:**
| Scenario | 2.0M | 2.2M | Trend |
|----------|------|------|-------|
| walk_fwd | 101% | 97% | ↓ (less overshoot) |
| walk_back | 86% | 87% | stable |
| lat_L | 155% | 155% | stuck overshoot |
| lat_R | 131% | 130% | stable |
| yaw_L | 117% | 109% | ↓ improving |
| yaw_R | 74% | 82% | ↑ improving |
| fwd_yaw_L wz | 145% | 135% | ↓ improving |
| fwd_yaw_R wz | 100% | 107% | ↑ slight |
| run_1.0 | 105% | 104% | stable |
| run_2.0 | 81% | 81% | stable |
| jump | 0.771 | 0.776 | stable |
| crouch | 0.088 | 0.088 | stuck |

Yaw gap: 43pp→27pp from 2M→2.2M. Improving but oscillatory.
Mirror augmentation should fix this structurally.


## s10g13 FINAL VERDICT: FAILED

**Evaluated at 2M steps — comprehensive failure:**
| Test | s10g11/1.2M | s10g13/2M | Verdict |
|------|-------------|-----------|---------|
| walk_fwd_0.5 | 101% | 95% | OK |
| walk_fwd_1.0 | 100% | 108% | worse |
| yaw+0.5 | 89% | 82% | ❌ declining |
| yaw-0.5 | 109% | 118% | ❌ worse |
| lat+0.3 | 98% | 86% | ❌ declined |
| run_1.0 | 123% | FALLS | ❌❌ BROKEN |
| run_1.2 | 123% | FALLS | ❌❌ BROKEN |
| run_1.5 | FALLS | 97% | weird |
| crouch | 0.081 | 0.082 | same |
| jump | 0.649 | 0.342 | ❌ worse |

Asymmetric vx tracking sigma catastrophically destabilized run mode.
Killed PID 1755761. Reverted to s10g11 base (d814713).

## v31s10g14 — Symmetric yaw_gain 0.90

**Root cause analysis of yaw asymmetry:**
- yaw+ at gain=0.90: 89% of cmd = 99% of ref. Near-perfect tracking.
- yaw- at gain=0.60: 109% of cmd = 182% of ref. Massive overshoot.
- Gain was reduced in s10g7 era when wz_overshoot penalty didn't exist.
- Now with -10.0*wz_overshoot + deadzone 1.15x, gradient at 170%: 9.84 vs 100%=22.0

**Fix**: yaw_gain = 0.90 symmetric (was 0.90 if wz>=0 else 0.60)
**Commit**: a520ea3
**Training**: PID 3116790, from s10g11/1.2M, lr=1e-4

## v31s6g9 @ 500K — Mirror Augmentation WORKING

| Scenario | g8@2.2M | g9@500K | Trend |
|----------|---------|---------|-------|
| walk_fwd | 97% | 91% | ↓ slight |
| walk_back | 87% | 93% | ↑ |
| lat_L | 155% | 152% | ↓ |
| lat_R | 130% | 126% | ↓ converging |
| yaw_L | 109% | 109% | stable |
| yaw_R | 82% | 91% | ↑↑ KEY WIN |
| fwd_yaw_L wz | 135% | 137% | stable |
| fwd_yaw_R wz | 107% | 107% | stable |
| run_1.0 | 104% | 103% | stable |
| run_2.0 | 81% | 81% | stable |
| jump | 0.776 | 0.781 | ↑ |
| crouch | 0.088 | 0.086 | ↑ |

**Yaw gap: 27pp → 18pp in 500K steps.** Mirror augmentation confirmed effective.
yaw_R improved 82%→91%. All other metrics stable or improving.


## v31s6g9 @ 1M — Yaw Gap Down to 13pp

| Scenario | g8@2.2M | g9@500K | g9@1M | Trend |
|----------|---------|---------|-------|-------|
| walk_fwd | 97% | 91% | 85% | ↓ expected dip |
| walk_back | 87% | 93% | 85% | ↓ from peak |
| lat_L | 155% | 152% | 149% | ↓ converging |
| lat_R | 130% | 126% | 115% | ↓↓ big improvement |
| yaw_L | 109% | 109% | 105% | ↓ less overshoot |
| yaw_R | 82% | 91% | 92% | ↑ improving |
| fwd_yaw_L wz | 135% | 137% | 127% | ↓ improving |
| fwd_yaw_R wz | 107% | 107% | 108% | stable |
| run_1.0 | 104% | 103% | 103% | stable |
| run_2.0 | 81% | 81% | 80% | stable |
| jump | 0.776 | 0.781 | 0.786 | ↑ |
| crouch | 0.088 | 0.086 | 0.086 | stable |

**Yaw gap trajectory: 43pp → 27pp → 18pp → 13pp.** Steady convergence confirmed.
**lat_R: 130→115%.** Symmetry enforcement fixing lateral too.
**walk_fwd: 97→85%.** Expected dip as policy learns both mirror frames. Should recover.


## v31s6g9 @ 1.5M — YAW GAP DOWN TO 6pp! MIRROR WORKING!

| Scenario | g8@2.2M | g9@500K | g9@1M | g9@1.5M | Trend |
|----------|---------|---------|-------|---------|-------|
| walk_fwd | 97% | 91% | 85% | 80% | ↓ temp cost |
| walk_back | 87% | 93% | 85% | 80% | ↓ |
| lat_L | 155% | 152% | 149% | 142% | ↓↓ |
| lat_R | 130% | 126% | 115% | 110% | ↓↓ almost there |
| yaw_L | 109% | 109% | 105% | 104% | → near perfect |
| yaw_R | 82% | 91% | 92% | 98% | ↑↑ NEARLY PERFECT |
| fwd_yaw_L wz | 135% | 137% | 127% | 115% | ↓↓ fixed |
| fwd_yaw_R wz | 107% | 107% | 108% | 116% | → symmetric! |
| run_1.0 | 104% | 103% | 103% | 101% | perfect |
| run_2.0 | 81% | 81% | 80% | 82% | stable |
| jump | 0.776 | 0.781 | 0.786 | 0.797 | ↑↑ best ever |
| crouch | 0.088 | 0.086 | 0.086 | 0.085 | ↓ improving |

**YAW GAP TRAJECTORY: 43pp → 27pp → 18pp → 13pp → 6pp**
**fwd_yaw wz NOW SYMMETRIC: 115%/116% (was 135%/107%)**
**Jump best ever: 0.797m**
**walk_fwd: 80% — expected dip from mirror. Monitor for recovery.**


## s10g14 FINAL VERDICT: FAILED

**yaw- DIVERGING: 155%→160% (1M→1.4M). Symmetric yaw_gain=0.90 causes runaway negative yaw.**
Physics creates ~1.8x reference in negative yaw direction.
Higher gain = more base motion = more total overshoot.
The reference trajectory is the WRONG lever for yaw- control.

Killed PID 3116790. Reverted to s10g11 base.

## v31s10g15 — Tighten wz Tracking (Walk Only)

**Root cause of yaw- at 109%:**
- wz_overshoot deadzone=1.15: yaw- at 109% is BELOW deadzone (0.545 < 0.575). ZERO penalty.
- wz_track sigma=0.08: at 109%, reward=97.5%. Almost no gradient.

**Three surgical changes (walk only, run untouched):**
1. wz sigma floor: 0.08→0.05 — 109% gets 96.0% reward (was 97.5%), 120% gets 82% (was 88%)
2. wz overshoot deadzone: 1.15→1.05 — 109% NOW triggers penalty (0.545 > 0.525)
3. walk wz_overshoot weight: -10→-15

**Expected gradient at key points:**
- 109% vs 100%: 0.78/step difference (was 0.30) — 2.6x stronger
- 120% vs 100%: 3.29/step difference — 23% of total reward

**Commit**: 8399f1c
**Training**: PID 1805244, from s10g11/1.2M, lr=1e-4
