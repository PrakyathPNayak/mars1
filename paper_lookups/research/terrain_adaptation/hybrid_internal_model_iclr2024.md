# Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response

**Authors:** (ICLR 2024)
**Year:** 2024 | **Venue:** ICLR
**Links:** [OpenReview](https://openreview.net/forum?id=93LoCyww8o)

---

## Abstract Summary
This paper introduces the Hybrid Internal Model (HIM) for learning agile legged locomotion using only proprioceptive sensors, without cameras, LiDAR, or other exteroceptive sensing. The core idea is to learn a latent state representation that encodes terrain properties (friction, elevation, compliance) through contrastive learning on proprioceptive history, combined with a simulated robot response mechanism that uses a learned dynamics model to predict how the robot would respond to hypothetical actions, providing additional implicit terrain information.

The hybrid model combines two complementary information streams: (1) a contrastive terrain encoder that maps proprioceptive history to a terrain latent vector, capturing static terrain properties, and (2) a learned dynamics model that predicts the robot response to planned actions given the current state, capturing dynamic terrain interactions. The simulated response is computed by forward-rolling the dynamics model for several steps with candidate actions, and the discrepancy between predicted and actual responses reveals terrain properties that the contrastive encoder alone might miss (e.g., subtle surface compliance that only manifests under load).

The framework achieves state-of-the-art blind locomotion performance on a Unitree Go1 quadruped, traversing challenging terrains including stairs, gaps, high obstacles, and deformable surfaces. The hybrid model significantly outperforms both pure contrastive encoding and pure dynamics prediction baselines, demonstrating that the two information streams are complementary. Sim-to-real transfer is achieved with no real-world fine-tuning, leveraging domain randomization and the hybrid model's robustness to sensor noise.

## Core Contributions
- Proposes the Hybrid Internal Model (HIM) combining contrastive terrain encoding with learned dynamics-based simulated response for proprioceptive-only terrain awareness
- Introduces the simulated robot response concept: using a learned forward dynamics model to generate hypothetical rollouts that reveal terrain properties through prediction errors
- Demonstrates that contrastive learning and dynamics prediction capture complementary terrain information, achieving 20-35% higher success rate than either alone
- Achieves sample-efficient contrastive training with a novel hard-negative mining strategy that selects terrain pairs with similar but not identical properties
- Validates on a Unitree Go1 quadruped with zero-shot sim-to-real transfer across 8 terrain types, including challenging scenarios like gaps and high obstacles
- Shows robustness to proprioceptive sensor noise (up to 20% additive Gaussian noise) due to the redundancy of the hybrid representation
- Provides theoretical analysis showing that the hybrid model terrain estimation error is bounded by the sum of contrastive encoder error and dynamics model prediction error, both of which decrease with training data

## Methodology Deep-Dive
The HIM architecture consists of three modules trained jointly: the contrastive terrain encoder E_c, the dynamics model M_d, and the locomotion policy pi. The contrastive encoder E_c takes a history of proprioceptive observations (joint positions, velocities, torques, body IMU) over a window of T=50 steps and produces a terrain latent vector z_c of dimension 16. The encoder uses a GRU recurrent network followed by a linear projection, enabling variable-length history processing.

The contrastive training follows the InfoNCE framework but with a key modification: hard-negative mining. Standard contrastive learning treats any different terrain as a negative pair, but this can lead to a coarse latent space that only distinguishes very different terrains. HIM's hard-negative mining specifically selects terrain parameters that are close but distinct (e.g., friction 0.5 vs 0.6) as negatives, forcing the encoder to learn fine-grained terrain distinctions. The mining strategy maintains a priority queue of terrain parameter pairs ranked by their contrastive loss contribution, and samples negatives proportionally to their difficulty.

The dynamics model M_d is a neural network that predicts the next proprioceptive state given the current state, action, and terrain latent: s_hat_{t+1} = M_d(s_t, a_t, z_c). The simulated response mechanism generates hypothetical trajectories by rolling out M_d for K=5 steps with a sequence of candidate actions (typically the policy's planned actions for the next K steps). The prediction residuals delta_k = s_{t+k} - s_hat_{t+k} for k=1...K are concatenated into a response vector z_r that captures terrain information not encoded in z_c. Intuitively, if the dynamics model accurately captures the robot's behavior on known terrain, large residuals indicate unknown terrain properties.

The policy pi receives the concatenation of the current state s_t, terrain latent z_c, and response vector z_r as input, and outputs joint position targets processed through a PD controller. The policy is trained with PPO in IsaacGym with 4096 parallel environments. The three modules are trained jointly: the contrastive loss trains E_c, the dynamics prediction loss trains M_d, and the PPO loss trains pi, with shared gradients flowing through the terrain latent z_c.

Domain randomization is applied extensively: terrain parameters (friction 0.1-2.0, restitution 0-1, ground stiffness, height field variations), robot parameters (mass +/-20%, motor strength +/-15%, joint damping +/-30%), and sensor noise (Gaussian noise on all proprioceptive channels). The randomization ranges are chosen to be wider than expected real-world variation, following the established practice from RMA and similar works.

The sim-to-real transfer pipeline is straightforward: the trained policy, encoder, and dynamics model are exported to ONNX format and deployed on the Unitree Go1's onboard Jetson Xavier NX. No real-world fine-tuning or adaptation is performed. The GRU encoder maintains a hidden state across timesteps, providing temporal context for terrain estimation. The dynamics model runs in parallel with the policy, computing the response vector from the most recent K-step rollout.

## Key Results & Numbers
- Traversal success rate across 8 terrain types: HIM 87.3%, contrastive-only 65.1%, dynamics-only 58.7%, no terrain encoding 42.5%
- Stairs (15cm): 92% success (HIM) vs 71% (contrastive-only) vs 55% (RMA baseline)
- Gaps (20cm): 78% success (HIM) vs 52% (contrastive-only) vs 38% (RMA)
- Deformable surface: 85% success (HIM) vs 61% (dynamics-only), demonstrating that dynamics response captures compliance better than contrastive encoding
- Sim-to-real transfer: 82% real-world success rate averaged across terrains, with no fine-tuning
- Hard-negative mining improves contrastive encoder quality by 18% (measured by terrain classification accuracy of the latent space)
- Inference time: 1.2ms total (0.3ms encoder, 0.5ms dynamics rollout, 0.4ms policy) on Jetson Xavier NX at 50Hz control
- Training: 1 billion environment steps in IsaacGym, ~6 hours on 1 NVIDIA A100

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The HIM framework is directly applicable to Mini Cheetah's blind locomotion task. The proprioceptive-only approach matches Mini Cheetah's sensor suite (joint encoders, IMU, no cameras in the base configuration). The hybrid terrain encoding (contrastive + dynamics response) provides richer terrain awareness than either approach alone, which is critical for the diverse terrains in Mini Cheetah's outdoor deployment scenarios.

Key implementation details transferable to Project A: (1) The GRU-based history encoder with T=50 step window is directly usable with Mini Cheetah's 50Hz control frequency (1 second of history). (2) The hard-negative mining strategy improves fine-grained terrain distinction, important for Mini Cheetah navigating mixed outdoor terrains. (3) The dynamics response mechanism (K=5 step rollout) adds only 0.5ms overhead, feasible within Mini Cheetah's control budget. (4) The domain randomization ranges reported in the paper provide a starting point for Mini Cheetah's MuJoCo training.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The HIM paper provides both validation and specific architectural guidance for Project B's CPTE (Contrastive Proprioceptive Terrain Encoder). The contrastive terrain encoding component of HIM is essentially the CPTE approach, and the paper's hard-negative mining strategy should be directly adopted to improve terrain discrimination in the CPTE's latent space.

The dynamics response mechanism offers an additional signal that could enhance the CPTE. In Project B's architecture, the dynamics model corresponds to the RSSM (Dreamer) world model at the Controller level. The RSSM's prediction residuals (difference between predicted and observed next state) can serve as the response vector z_r, providing terrain information complementary to the CPTE's contrastive latent z_c. This creates a natural integration point: the CPTE output z_c and the RSSM response z_r are concatenated and fed to the Planner level's Dual Asymmetric-Context Transformer.

The GRU encoder architecture choice (vs the TCN used in other papers) warrants consideration for Cassie's CPTE. GRUs handle variable-length sequences naturally, which is useful for Cassie's variable gait frequencies across different locomotion primitives. However, the TCN's parallelism may be preferred for training speed. An ablation comparing both architectures for the CPTE is recommended.

## What to Borrow / Implement
- Adopt the hard-negative mining strategy for contrastive training of Project B's CPTE, selecting terrain parameter pairs with small but meaningful differences as negative examples
- Implement the dynamics response mechanism using Project B's RSSM prediction residuals as complementary terrain information alongside the CPTE latent
- Use the GRU-based history encoder architecture as a candidate for the CPTE, comparing against TCN in ablation studies
- Apply the reported domain randomization ranges as starting points for both Project A (MuJoCo Mini Cheetah) and Project B (MuJoCo Cassie)
- Deploy the hybrid terrain representation (contrastive + dynamics response) for Project A's Mini Cheetah blind locomotion training

## Limitations & Open Questions
- The dynamics model adds computational overhead and introduces a second learned component that can fail independently; error analysis of cascaded failures is limited
- Hard-negative mining requires terrain parameter knowledge during training, which is available in simulation but complicates curriculum design for gradually introduced terrains
- The paper evaluates on a quadruped (Go1), not a biped; the higher balance challenge of bipedal locomotion may require larger latent dimensions or longer history windows
- The response mechanism assumes the dynamics model is reasonably accurate; during early training when the dynamics model is poor, the response vector may be uninformative or misleading
