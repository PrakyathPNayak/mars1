# CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion

**Authors:** ClearLab, SUSTech (Southern University of Science and Technology)
**Year:** 2024 | **Venue:** arXiv/Conference
**Links:** https://clearlab-sustech.github.io/concurrentTS/

---

## Abstract Summary
Concurrent Teacher-Student (CTS) proposes a fundamental rethinking of the standard two-stage teacher-student paradigm for legged locomotion. In the conventional approach, a teacher policy is first trained to convergence with privileged information (terrain maps, exact friction, full state), and then a student policy is distilled from the teacher using only partial observations. CTS argues that this sequential approach introduces a critical information bottleneck: the student is forced to mimic a teacher that may exploit privileged information in ways the student cannot replicate.

Instead, CTS trains both the teacher and student policies concurrently using PPO in an asymmetric setup. The teacher receives full state information while the student receives only proprioceptive observations, but both are trained simultaneously with shared reward signals. The teacher provides a soft guidance signal to the student through a regularization term, while the student's learning also influences the teacher's behavior through a mutual information objective. This co-evolution produces a teacher that learns to generate behaviors that are achievable by the student.

The method demonstrates superior velocity tracking accuracy and agility metrics for blind quadruped locomotion compared to traditional two-stage distillation. The concurrent training naturally avoids the distribution shift problem because both policies evolve together, and the teacher implicitly learns to avoid strategies that rely on information the student cannot access.

## Core Contributions
- Novel concurrent training paradigm that replaces sequential two-stage teacher-student distillation
- Mutual regularization between teacher and student that produces student-aware teacher policies
- Elimination of distribution shift between teacher demonstrations and student experience
- Superior velocity tracking and agility on blind locomotion benchmarks compared to sequential distillation
- Simpler training pipeline with a single training phase instead of two separate phases
- Theoretical analysis showing concurrent training converges to a Pareto-optimal point between teacher and student performance
- Demonstrated real-world deployment on quadruped robots with blind locomotion

## Methodology Deep-Dive
The CTS framework operates on a shared PPO training loop where two policy networks are updated simultaneously. The teacher network receives an extended observation vector containing privileged information: ground-truth terrain height samples around each foot, exact friction coefficients, external force vectors, and precise body state. The student network receives only proprioceptive observations: joint positions, joint velocities, body angular velocity from IMU, gravity vector projection, and previous actions.

Both networks share the same reward function for locomotion objectives (velocity tracking, energy minimization, smoothness). However, CTS adds two key coupling terms. First, a distillation regularization loss encourages the student's action distribution to be close to the teacher's: L_distill = KL(pi_student || pi_teacher), weighted by a coefficient that increases during training. Second, a reverse regularization term penalizes the teacher for producing actions that the student consistently fails to reproduce: L_reverse = -alpha * log(pi_student(a_teacher | o_student)), which encourages the teacher to favor strategies the student can follow.

The training proceeds in synchronized epochs. Each epoch collects rollouts from both teacher and student policies interacting with the same set of randomized environments. The teacher uses its privileged observations for action selection, while the student uses proprioceptive observations only. Both experience the same terrains and randomized dynamics, ensuring consistent reward comparisons. The PPO updates for both networks use the standard clipped objective, with the additional coupling losses added to the total objective.

A critical design choice is the scheduling of the coupling coefficients. Early in training, the distillation weight is low, allowing both policies to explore independently. As training progresses, the distillation weight increases, pulling the student toward the teacher's behavior. The reverse regularization starts high and decreases, initially constraining the teacher strongly to student-feasible strategies and relaxing this constraint as the student improves. This schedule is crucial for stability and final performance.

The asymmetric critic architecture uses a shared value network backbone with separate heads for teacher and student. The teacher critic head receives privileged state features, while the student critic head receives only the student's observation history. This asymmetric critic design follows the standard approach in the literature but benefits from the concurrent training by producing more consistent value estimates across both policies.

## Key Results & Numbers
- 15-25% improvement in velocity tracking RMSE compared to sequential two-stage distillation baseline
- Better agility scores measured by maximum achievable turning rate and acceleration
- More stable training curves with less variance across random seeds
- Reduced total training time by approximately 40% (single phase vs. two phases)
- Successful real-world deployment on quadruped platforms for blind locomotion
- Student policy achieves 85-90% of teacher's privileged performance (vs. 70-80% for sequential)
- Robust to variations in terrain difficulty and domain randomization ranges

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
CTS could directly replace the sequential teacher-student pipeline planned for Mini Cheetah sim-to-real transfer. Instead of first training a privileged teacher with terrain height maps and then distilling to a proprioception-only student, both can be trained concurrently. This would simplify the training pipeline from two stages to one, reduce total training time, and produce a student policy that better matches the teacher's performance.

The concurrent approach is particularly valuable for Mini Cheetah's domain randomization strategy. Since the teacher is implicitly guided to avoid privileged-information-dependent strategies, the resulting student policy may be more robust to the sim-to-real gap. The velocity tracking improvements are directly relevant to Mini Cheetah's locomotion objectives, and the agility gains could improve performance on challenging terrain transitions.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The Dual Asymmetric-Context Transformer (DACT) in Project B implements a form of teacher-student learning that could benefit significantly from the concurrent training paradigm. Instead of training the privileged-context transformer first and then distilling to the proprioceptive-context version, both contexts could be trained simultaneously. The mutual regularization would ensure the proprioceptive context learns representations that are achievable without privileged information.

For the hierarchical architecture, concurrent training could be applied at each level independently: the Planner could have concurrent privileged/student versions, the Controller could benefit from concurrent balance training with/without exact contact information, and the Safety Layer's LCBF constraints could be learned concurrently with privileged and deployed observation spaces.

## What to Borrow / Implement
- Replace sequential teacher-student training with concurrent paradigm for both Mini Cheetah and Cassie policies
- Adopt the reverse regularization loss to encourage teacher policies to find student-achievable strategies
- Use the coupling coefficient scheduling strategy (low distillation early, increasing over training) for stable concurrent training
- Apply concurrent training to the DACT module in Project B for more efficient asymmetric context learning
- Benchmark concurrent vs. sequential distillation on both Mini Cheetah and Cassie to quantify benefits

## Limitations & Open Questions
- Concurrent training introduces additional hyperparameters (coupling coefficients, schedule) that require tuning
- The mutual regularization may limit the teacher's exploration, potentially missing high-performance strategies that could eventually be distilled
- Scalability to more complex hierarchical architectures (like Project B's 4-level hierarchy) is not demonstrated
- The interaction between concurrent training and curriculum learning (progressive terrain difficulty) needs investigation
