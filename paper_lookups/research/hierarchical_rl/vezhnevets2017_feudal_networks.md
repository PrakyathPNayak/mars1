# FeUdal Networks for Hierarchical Reinforcement Learning

**Authors:** Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, Koray Kavukcuoglu
**Year:** 2017 | **Venue:** ICML
**Links:** https://proceedings.mlr.press/v70/vezhnevets17a.html

---

## Abstract Summary
FeUdal Networks (FuN) introduce a principled hierarchical reinforcement learning architecture inspired by feudal reinforcement learning (Dayan & Hinton, 1993). The framework decomposes decision-making into a Manager module that sets abstract goals at a coarse temporal resolution and a Worker module that produces primitive actions to achieve those goals. The Manager operates in a latent state space and communicates directional subgoals rather than absolute target states, enabling more flexible and transferable goal representations.

The key insight is that the Manager should not need to know the details of the Worker's action space, and the Worker should not need to understand the Manager's long-term planning objectives. This separation of concerns enables each module to learn at its natural temporal abstraction level. The Manager uses a dilated LSTM to maintain a longer temporal perspective, while the Worker conditions its policy on the current goal embedding using a learned linear transformation.

FuN is trained end-to-end with a novel transition policy gradient for the Manager, which decouples Manager gradient computation from the Worker's behavior. This avoids the need for the Manager to differentiate through the Worker's policy, making training more stable and scalable. The architecture achieves state-of-the-art results on challenging Atari games with sparse rewards, particularly Montezuma's Revenge.

## Core Contributions
- **Manager-Worker hierarchy** with two modules operating at different temporal resolutions, enabling temporal abstraction in goal-setting
- **Directional goal-setting** in a learned latent space rather than absolute goal coordinates, improving generalization and transfer
- **Dilated LSTM** for the Manager to maintain information over longer time horizons without increasing computational cost
- **Transition policy gradient** that allows the Manager to be trained without differentiating through the Worker's policy, decoupling the two learning problems
- **End-to-end trainable** architecture that does not require pre-defined subgoal spaces or reward functions for the Worker
- **Intrinsic reward** mechanism where the Worker receives intrinsic reward based on cosine similarity between its state transitions and the Manager's goal directions

## Methodology Deep-Dive
The FuN architecture consists of two recurrent neural network modules. The Manager module receives a state representation $s_t$ (output of a shared perception module, typically a CNN for pixel observations) and produces a goal vector $g_t$ in a learned latent space. Critically, this goal is normalized to unit length, so it represents a *direction* in latent space rather than an absolute target. The Manager's LSTM is dilated, meaning it operates at a coarser time resolution (every $c$ steps, typically $c=10$), allowing it to reason over longer temporal horizons without the vanishing gradient problems of standard RNNs over long sequences.

The Worker module receives the same state representation and the current goal $g_t$ from the Manager. It maintains an embedding matrix $\Phi$ and computes its policy as $\pi_W(a|s_t, g_t) = \text{softmax}(U_t w_t)$, where $U_t$ is a matrix of action embeddings from the Worker's LSTM and $w_t = \Phi g_t$ is a goal-conditioned weight vector. This multiplicative interaction between goals and actions allows the Worker to flexibly adjust its behavior based on the Manager's directions without requiring separate policies per goal.

The Manager is trained using a novel transition policy gradient. Rather than using the environment's extrinsic reward to train the Worker and a separate reward for the Manager, FuN uses the cosine similarity between the direction of the agent's state transitions $\Delta s_t = s_{t+c} - s_t$ and the Manager's goal $g_t$ as the Manager's training signal. The gradient for the Manager is $\nabla_\theta \log \pi_M(g_t|s_t) \cdot A_t^M$, where $A_t^M$ is the advantage computed from extrinsic rewards at the Manager's temporal resolution. This is a policy gradient that does not require backpropagation through the Worker.

The Worker receives an intrinsic reward $r_t^I = \frac{1}{c} \sum_{i=1}^{c} \cos(\Delta s_{t-i}, g_{t-i})$, which measures how well the agent's recent state transitions have aligned with the Manager's goals. This intrinsic reward is combined with the extrinsic environment reward to train the Worker, encouraging it to follow the Manager's directions while also maximizing task reward.

The perception module is shared between Manager and Worker, typically a convolutional network that processes raw pixel observations into the state representation $s_t$. The entire architecture—perception, Manager LSTM, Worker LSTM, and embedding matrices—is trained end-to-end using a combination of the transition policy gradient (for the Manager) and a standard policy gradient with the mixed intrinsic/extrinsic reward (for the Worker).

## Key Results & Numbers
- **Montezuma's Revenge:** Achieved a score of ~400 (SOTA at the time), significantly outperforming flat A3C baselines (~0) without any demonstrations or hand-crafted rewards
- **Atari Suite:** Improved performance over A3C baseline on 31 out of 57 Atari games
- **Exploration efficiency:** In sparse-reward environments, FuN discovers reward signals much faster than flat policies
- **Temporal abstraction:** Manager goals meaningfully correspond to directional movement in state space, verified via visualization
- **Ablation studies:** Removing directional goals, dilated LSTM, or intrinsic reward each significantly degrades performance, validating each component's contribution

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
FuN's hierarchical goal-setting architecture provides a principled framework for decomposing navigation tasks on the Mini Cheetah. A Manager module operating at lower temporal resolution could set waypoint-like goals (e.g., "move in this direction at this speed"), while a Worker module handles the detailed gait generation and motor commands. The directional goal representation is particularly appealing for locomotion since the Manager need not specify exact joint configurations, only the desired direction of travel in a learned state space.

However, FuN was designed for discrete-action environments (Atari) and would need substantial adaptation for continuous-control locomotion. The Manager's dilated LSTM is interesting for maintaining awareness over multiple gait cycles, but the direct applicability to MuJoCo-based PPO training is limited. The intrinsic reward mechanism based on state-transition direction could be adapted to encourage exploration during curriculum learning phases.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
FuN is foundational to Project B's 4-level hierarchy. The Manager-Worker decomposition directly informs the Planner→Primitives interface, where the Planner sets abstract goals (target velocities, gait modes) and the Primitives level selects appropriate motor behaviors. The key principle of communicating goals in a learned latent space (rather than explicit action commands) is exactly what the Dual Asymmetric-Context Transformer architecture implements at the Planner level.

The transition policy gradient is particularly relevant: in Project B's hierarchy, the Planner should be trainable without needing to differentiate through the entire Primitives→Controller→Safety stack. FuN's approach of measuring "did the agent move in the direction I specified" as the Manager's reward signal can be adapted for the Planner's training—measuring whether the Primitives achieved the intended gait transition or velocity target. The directional goal-setting concept also informs how the Planner communicates with the MC-GAT Primitive Selector, using learned embeddings rather than hand-designed command vectors.

## What to Borrow / Implement
- **Directional goal communication** between hierarchy levels: use normalized goal vectors in learned latent space for Planner→Primitives interface
- **Dilated temporal processing** for the Planner module: operate at a coarser time resolution (e.g., every 10-50 control steps) to enable longer-horizon reasoning
- **Transition policy gradient** for training higher levels without differentiating through lower levels, enabling modular training of the hierarchy
- **Intrinsic reward based on goal alignment** as an auxiliary training signal for the Primitives level
- **End-to-end trainable hierarchy** principle: avoid hand-designed interfaces between levels where possible

## Limitations & Open Questions
- **Discrete actions only:** FuN was designed for Atari's discrete action space; adaptation to continuous locomotion control requires significant architectural changes
- **Two-level only:** The architecture supports only Manager-Worker; extending to 4+ levels (as in Project B) is non-trivial and may require different gradient estimation strategies at each level
- **Goal space is implicit:** The Manager's goal space is learned end-to-end, making it hard to interpret or constrain for safety-critical applications
- **Scalability to high-dimensional continuous control:** No evidence that the directional goal-setting works well when the state space is high-dimensional proprioceptive data rather than learned image features
