# Footholds Optimization for Legged Robots Walking on Complex Terrain

**Authors:** Various
**Year:** 2023 | **Venue:** Frontiers of Mechanical Engineering (Springer)
**Links:** https://link.springer.com/article/10.1007/s11465-022-0742-y

---

## Abstract Summary
This paper develops continuous online optimization for safe and stable foothold selection on complex terrain. The method builds terrain heightmaps from onboard sensors and minimizes slippage and tripping risk through real-time foothold optimization. The approach is integrated with locomotion controllers for improved stability and maneuverability on uneven and cluttered terrain.

## Core Contributions
- Formulates foothold selection as a continuous optimization problem minimizing slippage and tripping risk
- Develops real-time terrain heightmap construction from onboard depth sensors for foothold evaluation
- Proposes a multi-objective cost function incorporating terrain slope, roughness, edge proximity, and support polygon stability
- Achieves real-time foothold optimization running at locomotion control frequency
- Demonstrates improved stability and reduced slippage on complex terrain including stairs, rubble, and gaps
- Integrates with existing locomotion controllers as a plug-in foothold optimizer
- Provides analysis of the optimization landscape showing the terrain features most critical for stable footholds

## Methodology Deep-Dive
The foothold optimization problem is formulated as minimizing a cost function over the 2D foot placement position (x, y) on the terrain surface. For each candidate foothold, the cost function evaluates multiple criteria: terrain slope at the contact point (steep slopes increase slippage risk), local roughness (high roughness reduces contact area), proximity to terrain edges (edges provide unstable support), and the resulting support polygon stability (how well the foothold contributes to the robot's overall balance).

The terrain heightmap is constructed in real-time from onboard depth cameras or LiDAR. The raw point cloud is processed into a 2.5D elevation map centered on the robot, with a resolution of 1-2cm per cell. The heightmap is updated at each control step as the robot moves, providing a rolling window of terrain information. Gradient and curvature maps are derived from the heightmap to compute slope and roughness features at each potential foothold location.

The optimization uses gradient-based methods (L-BFGS or projected gradient descent) since the cost function is differentiable with respect to foothold position. The terrain features (slope, roughness, edge distance) are computed as smooth functions of the heightmap, enabling gradient computation through automatic differentiation. The optimization is initialized at the default foothold position (from the nominal gait) and converges within 5-10 iterations, taking <1ms per foot on modern hardware.

The multi-objective cost function is formulated as a weighted sum: C(x,y) = w_slope · f_slope(x,y) + w_rough · f_rough(x,y) + w_edge · f_edge(x,y) + w_stability · f_stability(x,y). The weights are tuned based on the terrain type and can be adapted online. The slope cost penalizes footholds on surfaces steeper than a friction-dependent threshold. The roughness cost penalizes uneven contact surfaces. The edge cost applies a strong penalty near terrain discontinuities (cliffs, step edges). The stability cost evaluates the resulting support polygon when this foot is placed.

Integration with the locomotion controller is modular. The foothold optimizer receives the nominal foot trajectory from the gait controller and modifies only the touchdown position. The swing trajectory is then replanned to reach the optimized foothold, maintaining smooth leg motion. This plug-in design allows the optimizer to be added to any existing locomotion controller without modifying the core control logic.

## Key Results & Numbers
- Slippage reduction of 60-70% compared to nominal (unoptimized) footholds on complex terrain
- Real-time optimization at <1ms per foot, compatible with 100-500 Hz locomotion control
- Successful traversal of stairs with 15cm rise and 25cm run
- Rubble traversal with up to 8cm random height variation
- Gap crossing up to 20cm with optimized foot placement near stable edges
- Support polygon stability improved by 35% on average across terrain types
- Heightmap construction from depth camera at 30 Hz with 2cm resolution

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Foothold optimization directly complements Mini Cheetah's RL-based locomotion policy. The RL policy generates nominal foot trajectories, and the foothold optimizer refines touchdown positions for terrain safety. This hybrid approach combines RL's adaptability with optimization's precision for foot placement. The real-time optimization (<1ms per foot) is compatible with Mini Cheetah's 500 Hz PD control loop. The heightmap construction from depth cameras provides terrain awareness that enhances the RL policy's proprioceptive observations. The plug-in design means the optimizer can be added to the existing RL pipeline without retraining.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Foothold optimization informs the foot placement strategy at Cassie's Controller level in the hierarchy. For bipedal locomotion, precise foot placement is even more critical than for quadrupeds since each foot supports 50% of body weight. The terrain heightmap integration is directly relevant to the CPTE module — the heightmap features (slope, roughness, edge distance) provide structured terrain input for the contrastive encoder. The optimization cost function components (slope, edge proximity, stability) can be incorporated as reward terms or constraints in the Controller's policy. The Differentiable Capture Point module can use optimized footholds to compute more accurate capture point trajectories, improving dynamic balance during walking and running.

## What to Borrow / Implement
- Implement the heightmap-based foothold optimization as a post-processing layer on RL foot trajectories
- Use the terrain cost function components (slope, roughness, edge proximity) as reward shaping terms for RL training
- Integrate the heightmap construction pipeline for terrain awareness in both projects
- Adopt the gradient-based foothold optimization for real-time deployment
- Use the terrain features (slope, roughness, edge maps) as inputs to the CPTE module in Project B
- Combine foothold optimization with the Differentiable Capture Point for bipedal balance-aware foot placement
- Implement the support polygon stability metric as a constraint in the LCBF safety layer

## Limitations & Open Questions
- Heightmap construction requires good depth camera data — fails in poor lighting or with reflective surfaces
- The 2.5D heightmap cannot represent overhanging terrain (caves, tunnels, low bridges)
- Optimization may converge to local minima on highly complex terrain with many obstacles
- The cost function weights require manual tuning and may not generalize across all terrain types
- For bipedal robots, the support polygon concept must be replaced with more dynamic balance criteria
- The plug-in approach assumes the gait controller can follow modified foot trajectories — not always true for aggressive gaits
- Slippery terrain (ice, wet surfaces) may require dynamic friction estimation beyond static heightmap analysis
