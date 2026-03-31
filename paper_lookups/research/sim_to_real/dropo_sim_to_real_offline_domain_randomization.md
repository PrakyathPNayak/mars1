---
## 📂 FOLDER: research/sim_to_real/

### 📄 FILE: research/sim_to_real/dropo_sim_to_real_offline_domain_randomization.md

**Title:** DROPO: Sim-to-Real Transfer with Offline Domain Randomization
**Authors:** Gabriele Tiboni, Karol Arndt, Ville Kyrki
**Year:** 2023
**Venue:** Robotics and Autonomous Systems (Elsevier)
**arXiv / DOI:** 10.1016/j.robot.2023.104432

**Abstract Summary (2–3 sentences):**
DROPO introduces a framework that uses pre-collected offline data—including human demonstrations and limited real-world trajectories—to estimate optimal simulator parameter randomization distributions for sim-to-real transfer. Rather than relying on hand-tuned uniform randomization ranges, the method recovers dynamic parameter distributions that capture unmodeled phenomena, producing policies that transfer more reliably and safely to real hardware. The approach is demonstrated on both manipulation and locomotion tasks, showing improved transfer success rates over standard domain randomization techniques.

**Core Contributions (bullet list, 4–7 items):**
- Offline domain randomization framework that estimates simulator parameter distributions from pre-collected real-world data
- Bayesian inference approach for recovering posterior distributions over simulation parameters given observed trajectories
- Elimination of the need for online real-world interaction during randomization parameter tuning
- Safety-aware transfer pipeline that avoids dangerous trial-and-error on real hardware
- Theoretical analysis of identifiability conditions for simulation parameter estimation
- Demonstration on both manipulation (pushing, grasping) and locomotion tasks
- Comparison showing improved transfer over uniform and manually-tuned domain randomization

**Methodology Deep-Dive (3–5 paragraphs):**
The core insight of DROPO is that domain randomization distributions should be informed by real-world data rather than set arbitrarily. Standard domain randomization uniformly samples simulation parameters (masses, frictions, damping coefficients, actuator gains) over broad ranges, hoping that the real-world parameters fall somewhere within these ranges. This approach has two problems: if the ranges are too narrow, the real parameters may not be covered; if too broad, the policy must be robust to an unnecessarily large parameter space, leading to conservative behavior. DROPO addresses this by using a small set of offline real-world trajectories to infer the likely distribution of simulation parameters, producing a focused randomization distribution centered on the real-world dynamics.

The inference procedure uses likelihood-free Bayesian inference (approximate Bayesian computation, ABC, or neural density estimation). Given a set of real-world trajectories τ_real = {(s_t, a_t, s_{t+1})} collected from the physical system, the method evaluates candidate simulation parameter vectors ξ by simulating the same action sequences in the simulator and comparing the resulting state transitions. The discrepancy between simulated and real transitions is measured using a distance metric (e.g., mean squared error on next-state predictions). Parameters that produce simulated dynamics close to the real dynamics receive high likelihood scores. By running this process with many candidate parameter vectors sampled from a prior distribution, the method builds up an approximate posterior distribution P(ξ | τ_real) over simulation parameters. This posterior can then be used directly as the domain randomization distribution during policy training.

The offline data collection step is designed to be minimally invasive. The authors show that effective parameter estimation requires only a modest number of real-world trajectories (10–50 short trajectories), which can be collected through human teleoperation, scripted motions, or even passive observation of the robot under gravity. The key requirement is that the trajectories exercise the dynamic modes relevant to the task—for locomotion, this means the trajectories should include walking at various speeds and on different surfaces to excite the relevant friction and inertial parameters. The offline nature of the data collection means that no RL policy is deployed on the real robot during the randomization tuning phase, eliminating the risk of damage from poorly-trained policies.

Once the posterior distribution over simulation parameters is obtained, policy training proceeds using standard RL (e.g., PPO) with domain randomization, but instead of sampling parameters from a uniform distribution, parameters are sampled from the learned posterior. This produces policies that are robust to the actual range of dynamic uncertainty present in the real system, rather than a hypothetical worst-case range. The authors also propose an iterative refinement procedure where, after initial policy transfer, additional real-world data is collected under the transferred policy and used to update the posterior distribution. This iterative approach progressively narrows the simulation-to-reality gap, though the paper emphasizes that even a single round of offline estimation provides significant improvements.

The experimental evaluation compares DROPO against several baselines: (1) no domain randomization (direct transfer), (2) uniform domain randomization with hand-tuned ranges, (3) automatic domain randomization (ADR) that progressively expands ranges during training, and (4) system identification that estimates a single best-fit parameter vector. DROPO consistently outperforms all baselines on transfer success rates. The advantage is most pronounced in locomotion tasks where accurate friction and contact dynamics estimation significantly impacts policy performance. The locomotion experiments use a simulated quadruped with transfer to a different simulator acting as the "real world" (sim-to-sim transfer), while the manipulation experiments include real-robot transfer on a Franka Panda arm.

**Key Results & Numbers:**
- Improved sim-to-real transfer success rates by 15–30% over uniform domain randomization
- Effective parameter estimation from as few as 10–20 offline trajectories
- Safer parameter estimation process requiring no online policy deployment on real hardware
- Locomotion transfer experiments show 25% improvement in tracking accuracy after domain randomization optimization
- Manipulation tasks achieve 90%+ grasp success rate compared to 70% with uniform randomization
- Posterior distributions recover physically meaningful parameter ranges consistent with ground-truth values

**Relevance to Project A (Mini Cheetah):** HIGH — Directly applicable to optimizing the domain randomization distribution for Mini Cheetah sim-to-real transfer. Instead of manually tuning randomization ranges for mass, friction, motor gains, and latency, DROPO can estimate these distributions from a small number of real Mini Cheetah trajectories, potentially improving transfer quality for the PPO-trained locomotion policy.

**Relevance to Project B (Cassie HRL):** MEDIUM — Applicable to parameter estimation for the Cassie simulation environment, particularly for estimating contact dynamics and actuator characteristics. However, the hierarchical nature of the Cassie HRL system means that sim-to-real challenges exist at multiple levels, and DROPO addresses only the low-level dynamics estimation aspect.

**What to Borrow / Implement:**
- Offline trajectory collection protocol for estimating simulator parameter distributions
- Bayesian inference framework for domain randomization distribution optimization
- Posterior-based sampling replacing uniform randomization during PPO training
- Iterative refinement procedure for progressively improving parameter estimates after initial transfer
- Minimal data requirements (10–50 trajectories) making the approach practical for expensive real-robot data

**Limitations & Open Questions:**
- The quality of parameter estimation depends on the informativeness of the offline trajectories; poorly chosen trajectories may not excite relevant dynamics
- Assumes that the simulation model structure is correct and only parameters are uncertain; unmodeled dynamics (e.g., cable dynamics, sensor latencies) are not captured
- Likelihood-free inference methods can be computationally expensive for high-dimensional parameter spaces
- The approach has been primarily validated on sim-to-sim transfer for locomotion; real-robot locomotion transfer results would strengthen the paper
- Scaling to very high-dimensional parameter spaces (>20 parameters) may require more sophisticated inference methods
- The iterative refinement procedure requires additional real-world data collection, which may not always be feasible
---
