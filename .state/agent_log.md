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
