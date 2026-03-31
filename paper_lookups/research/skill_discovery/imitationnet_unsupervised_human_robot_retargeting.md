# ImitationNet: Unsupervised Human-to-Robot Motion Retargeting via Shared Latent Space

**Authors:** Various
**Year:** 2023 | **Venue:** arXiv 2023
**Links:** https://arxiv.org/abs/2309.05310

---

## Abstract Summary
ImitationNet introduces an unsupervised deep learning method for translating human motions to robot actions without requiring paired demonstration data. The approach constructs a joint latent space between humans and robots using adaptive contrastive learning, enabling direct motion control and interpolation from varied input modalities including text, video, and key poses. This eliminates the laborious process of manually creating paired human-robot motion datasets.

## Core Contributions
- Proposes unsupervised cross-morphology motion retargeting without paired training data
- Constructs a shared latent space between human and robot motion representations via adaptive contrastive learning
- Supports multi-modal input: text descriptions, video demonstrations, and key pose specifications
- Demonstrates smooth motion interpolation in the shared latent space
- Validates on real robot hardware, showing practical deployment viability
- Eliminates the need for manual motion correspondence annotation
- Enables zero-shot transfer to new motion types not seen during training

## Methodology Deep-Dive
The core idea is to learn separate encoders for human and robot motion that map into a shared latent space where semantically similar motions are close regardless of morphology. The human encoder processes motion capture sequences (joint angles, positions, velocities), while the robot encoder handles the robot's own kinematic representation. Both encoders are trained simultaneously without explicit paired correspondences.

The adaptive contrastive learning objective is key to the approach. Rather than requiring explicit human-robot motion pairs, the method uses temporal consistency and motion semantics to align the two latent spaces. Positive pairs are generated through data augmentation and temporal proximity, while negatives come from temporally distant or semantically different motions. The adaptive component adjusts the contrastive temperature and hard-negative mining strategy during training to handle the domain gap between human and robot kinematics.

For multi-modal input, separate front-end encoders process text (via CLIP-like embeddings), video (via pose estimation + temporal encoding), and key poses (via direct kinematic encoding). Each modality is mapped into the same shared latent space, where the robot decoder generates corresponding joint commands. This enables a unified interface for specifying desired motions regardless of input type.

The robot decoder maps from latent space to joint-space trajectories, incorporating kinematic constraints and joint limits specific to the target robot. A physics-based refinement step ensures dynamically feasible motions. The entire pipeline—from input modality to robot commands—is differentiable, enabling end-to-end fine-tuning.

Real-robot deployment involves a simple calibration step to account for dynamics not captured in the kinematic model. The latent space structure allows smooth interpolation between motions, enabling novel behavior synthesis by traversing between known motion embeddings.

## Key Results & Numbers
- Achieves cross-morphology retargeting without any paired human-robot data
- Supports text, video, and key-pose inputs through a unified latent space
- Real-robot deployment validated with smooth, natural-looking motions
- Latent space interpolation produces plausible intermediate motions
- Outperforms supervised baselines that use limited paired data
- Retargeting errors measured in joint-angle RMSE are competitive with supervised methods
- Generalization to unseen motion types demonstrated via zero-shot latent space traversal

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
While ImitationNet focuses on humanoid morphologies, the shared latent space concept could be adapted for retargeting animal locomotion data (e.g., from dog motion capture) to the Mini Cheetah's 12 DoF kinematic structure. The unsupervised nature is attractive since paired quadruped demonstration data is scarce. The multi-modal input could allow specifying desired gaits via text ("trotting at 1.5 m/s") or video references. However, the morphological gap between humans and quadrupeds is larger than human-to-humanoid, requiring significant adaptation of the contrastive learning objective.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Cross-morphology retargeting from human motion capture to Cassie is directly relevant since both are bipedal. The shared latent space approach could provide natural gait priors for the Primitives level without requiring expensive paired Cassie demonstrations. The adaptive contrastive learning aligns well with CPTE (Contrastive Pre-trained Terrain Encoder) methodology already in the hierarchy. The multi-modal input could enable the Planner level to specify locomotion goals via text or key-pose waypoints. The latent space structure maps naturally to the skill embedding space used by DIAYN/DADS for skill discovery.

## What to Borrow / Implement
- Adapt the shared latent space framework for human-to-Cassie motion retargeting
- Use adaptive contrastive learning to align human MoCap with Cassie joint trajectories
- Integrate the multi-modal input interface at the Planner level for flexible goal specification
- Leverage latent space interpolation for smooth transitions between locomotion primitives
- Combine with AMP: use retargeted motions as reference for adversarial motion prior training

## Limitations & Open Questions
- Unsupervised alignment may produce semantically incorrect correspondences for complex motions
- Morphological gap between humans and quadrupeds may be too large for direct application to Project A
- Real-time inference speed not reported—may be too slow for 500 Hz control
- Physics-based refinement step may not generalize across terrain types
- Open question: How does retargeting quality degrade with increasing morphological difference?
- No explicit handling of contact dynamics, which are crucial for locomotion
