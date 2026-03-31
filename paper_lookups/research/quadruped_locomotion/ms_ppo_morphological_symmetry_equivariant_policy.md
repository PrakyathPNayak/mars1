---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/ms_ppo_morphological_symmetry_equivariant_policy.md

**Title:** MS-PPO: Morphological-Symmetry-Equivariant Policy for Legged Robot Locomotion
**Authors:** Jiayu Wen, Tao Zhang, Ziqing Wang, Wei Song, Rui Zhao
**Year:** 2024
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2512.00727

**Abstract Summary (2–3 sentences):**
MS-PPO encodes robot kinematic symmetry directly into the PPO policy network architecture, enforcing morphological symmetry equivariance as a hard architectural constraint rather than relying on symmetry reward shaping or data augmentation. The approach ensures that symmetric robot states produce consistent symmetric policy responses, improving sample efficiency and gait symmetry in legged robot locomotion. Experimental results show approximately 2x sample efficiency improvement and the elimination of symmetry reward terms while achieving better gait symmetry on real robots.

**Core Contributions (bullet list, 4–7 items):**
- Morphological symmetry equivariant network architecture for PPO policy and value functions
- Elimination of symmetry-related reward terms (symmetry loss, mirror penalty) through architectural enforcement
- Formal characterization of symmetry groups for common legged robot morphologies (quadrupeds, bipeds)
- 2x improvement in sample efficiency compared to standard PPO with symmetry rewards
- Better gait symmetry on real robots measured through contact timing and joint trajectory analysis
- Compatibility with existing PPO training pipelines requiring only network architecture modification
- Analysis of symmetry breaking during training and how equivariant architectures prevent it

**Methodology Deep-Dive (3–5 paragraphs):**
The paper begins by formally characterizing the symmetry group of legged robot morphologies. For a quadruped with bilateral symmetry (left-right mirror), the symmetry group is Z_2, consisting of the identity transformation and the left-right mirror reflection. This symmetry manifests in both the observation space and the action space: mirroring a robot's state (swapping left and right leg observations) should produce mirrored actions (swapping left and right leg commands). For robots with additional symmetries (e.g., front-back symmetry in some quadrupeds), larger symmetry groups apply. The authors derive the explicit permutation matrices that represent these symmetry transformations for common observation and action representations, including joint positions, joint velocities, body orientation (quaternion), angular velocity, and foot contact states.

The core technical contribution is an equivariant neural network architecture that guarantees the symmetry relationship π(g·s) = g·π(s) for all states s and symmetry transformations g, where g· denotes the action of the symmetry group on states and actions respectively. This is achieved using equivariant linear layers where the weight matrices are constrained to commute with the symmetry transformation matrices. Specifically, if P_s and P_a are the permutation matrices representing the symmetry in observation and action spaces, then the weight matrix W of each layer must satisfy P_a · W = W · P_s (for the first layer) and similar constraints for subsequent layers. These constraints reduce the effective number of free parameters in the network, which contributes to improved sample efficiency. The equivariant architecture is applied to both the policy network (actor) and the value function network (critic), ensuring consistent symmetry throughout the training process.

The practical implementation modifies the standard MLP architecture used in PPO by replacing standard linear layers with equivariant linear layers. The equivariant layer construction follows the approach of factorizing weight matrices into symmetric and anti-symmetric components, where each component satisfies the equivariance constraint independently. For the Z_2 symmetry group, this amounts to constraining certain blocks of the weight matrix to be equal (symmetric component) or negated (anti-symmetric component). The authors provide an efficient implementation that computes the equivariant forward pass without explicitly constructing the constrained weight matrices, instead using a reparameterization that maps unconstrained parameters to constrained weights. The activation functions are chosen to be compatible with equivariance (element-wise activations like ReLU and tanh preserve equivariance by default).

Training follows the standard PPO algorithm with the equivariant network architecture as a drop-in replacement for the standard MLP. The key advantage is that symmetry-related reward terms—which are commonly used in legged locomotion RL but introduce additional hyperparameters and can conflict with task objectives—are no longer needed. Standard PPO reward functions for locomotion (velocity tracking, orientation penalties, smoothness penalties, energy penalties) are used without any symmetry-specific additions. The authors show that standard PPO with symmetry rewards often exhibits symmetry breaking during training, where the policy converges to an asymmetric gait despite the reward penalty. This occurs because the symmetry reward is a soft constraint that competes with other reward components, and the optimization may find local optima where slightly asymmetric gaits achieve higher total reward. The equivariant architecture prevents this by construction.

The experimental evaluation includes both simulation and real-robot experiments on quadruped and biped platforms. In simulation, the authors compare MS-PPO against three baselines: (1) standard PPO without symmetry handling, (2) PPO with symmetry reward terms, and (3) PPO with data augmentation (mirroring trajectories during training). MS-PPO achieves approximately 2x faster convergence measured by the number of environment steps to reach a target performance level. On real robots, the gait symmetry is quantified by measuring the temporal difference between left and right foot contact timings and the amplitude difference between left and right joint trajectories. MS-PPO produces significantly more symmetric gaits than all baselines, with the symmetry improvement being most pronounced during dynamic gaits (trotting, bounding) where asymmetric gaits are more likely to emerge with standard training.

**Key Results & Numbers:**
- 2x sample efficiency improvement compared to standard PPO with symmetry rewards
- Elimination of symmetry reward terms and their associated hyperparameters
- Gait symmetry improved by 60–80% on real robots (measured by left-right contact timing difference)
- Compatible with standard PPO pipelines with minimal code changes (network architecture only)
- Parameter count reduced by approximately 40% due to equivariance constraints
- Consistent performance improvement across quadruped (12 DoF) and biped (10 DoF) platforms
- No symmetry breaking observed during training, unlike soft-constraint baselines

**Relevance to Project A (Mini Cheetah):** HIGH — Directly applicable to exploiting the Mini Cheetah's bilateral symmetry for more efficient PPO training. The 12 DoF Mini Cheetah has clear left-right symmetry that can be encoded using the Z_2 equivariant architecture. The 2x sample efficiency improvement and elimination of symmetry reward terms would directly benefit the training pipeline, and the improved gait symmetry would translate to more robust real-world locomotion.

**Relevance to Project B (Cassie HRL):** MEDIUM — Symmetry concepts are applicable to Cassie's sagittal-plane symmetric morphology, particularly for the low-level controller that generates joint commands. However, the hierarchical structure means that symmetry needs to be considered at multiple levels, and higher-level components (planner, safety filter) may not benefit as directly from morphological symmetry equivariance.

**What to Borrow / Implement:**
- Z_2 equivariant network architecture for Mini Cheetah's bilateral symmetry
- Permutation matrix derivation for the specific observation and action representations used
- Equivariant linear layer implementation as a drop-in replacement for standard MLP layers
- Elimination of symmetry reward terms, simplifying the reward function design
- Real-robot gait symmetry metrics (contact timing difference, joint trajectory amplitude difference) for evaluation

**Limitations & Open Questions:**
- Limited to discrete symmetry groups (Z_2, Z_4); continuous symmetries require different mathematical treatment
- The approach assumes perfect symmetry in the robot morphology; manufacturing asymmetries or wear may break the assumption
- Equivariant architectures constrain the representational capacity of the network; very complex asymmetric behaviors may be harder to learn
- The paper focuses on locomotion; applicability to loco-manipulation tasks where symmetry may be broken by the task is unexplored
- Integration with hierarchical architectures (where different levels may have different symmetry properties) is not addressed
- The 2x sample efficiency claim is based on specific experimental configurations; gains may vary with different reward functions and tasks
---
