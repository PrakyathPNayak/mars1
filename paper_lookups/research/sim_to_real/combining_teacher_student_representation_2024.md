# Combining Teacher-Student with Representation Learning: A Concurrent Approach for Legged Locomotion

**Authors:** (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2405.10830)

---

## Abstract Summary
This paper presents a method that combines teacher-student policy transfer with explicit representation learning to improve sim-to-real transfer for legged locomotion. The core idea is that the student network should not merely imitate the teacher's actions but should simultaneously learn a compressed, informative representation of the privileged information that the teacher has access to. By jointly training the student's action prediction and representation learning objectives, the student develops an internal model of the environment state that enables more adaptive deployment behavior, going beyond pure behavioral cloning of the teacher.

The representation learning component uses a contrastive learning objective: the student's recurrent encoder (processing proprioceptive history) is trained to produce latent codes that are close to the teacher's privileged latent codes for the same environmental state and distant for different states. This contrastive objective is combined with the standard behavioral cloning loss (MSE on action predictions) and optionally a reconstruction loss (predicting privileged features from the student's latent). The three-part loss ensures the student's representation captures both behavioral (action-relevant) and environmental (state-relevant) information.

Experiments on simulated and real quadruped platforms (Unitree A1, Go1) demonstrate that the combined approach outperforms standard teacher-student distillation (action-only cloning) and representation-only methods (contrastive learning without teacher actions). The improvement is most significant on challenging terrains where the student must infer terrain properties from proprioceptive history. Real-world deployment shows improved terrain adaptation, faster gait switching, and more robust blind locomotion compared to baselines.

## Core Contributions
- Joint framework combining teacher-student distillation with contrastive representation learning in a single training objective
- Contrastive loss that aligns student and teacher latent spaces while preserving discriminative structure
- Optional privileged feature reconstruction loss that provides additional supervision for student representation quality
- Recurrent (GRU) and transformer-based student encoders for proprioceptive history processing, with comparative evaluation
- Demonstration that representation quality directly correlates with deployment robustness on unseen terrains
- 15–25% improvement over action-only distillation on rough terrain locomotion tasks
- Real-world validation on Unitree A1 and Go1 showing improved terrain adaptation and gait switching

## Methodology Deep-Dive
The teacher policy π_T(a | o_T) is trained using PPO with full privileged observations o_T = [o_proprio, o_terrain, o_dynamics] including ground-truth terrain heightmap, friction coefficients, and body dynamics parameters. The teacher produces actions and a latent representation l_T = E_T(o_T) ∈ R^d from its encoder. Once the teacher converges, it is frozen and used to generate a dataset of (o_proprio_history, o_T, a_T, l_T) tuples for student training.

The student encoder E_S processes a window of proprioceptive observations H_t = [o_proprio_{t-K}, ..., o_proprio_t] through either a GRU (hidden size 256, 2 layers) or a causal transformer (4 layers, 4 heads, d_model=256). The encoder output is a latent vector l_S = E_S(H_t) ∈ R^d that should capture sufficient environmental information for action prediction.

The training loss combines three objectives: (1) Behavioral cloning: L_BC = ||π_S(a | l_S) - a_T||², matching the teacher's actions; (2) Contrastive alignment: L_contrast = -log[exp(sim(l_S, l_T) / τ) / Σ_j exp(sim(l_S, l_T^j) / τ)], an InfoNCE loss where sim is cosine similarity, τ is temperature (0.07), and l_T^j are latent codes from other states in the batch serving as negatives; (3) Reconstruction: L_recon = ||D(l_S) - o_terrain||², a small MLP decoder D predicting terrain heightmap from the student's latent. The total loss is L = L_BC + α * L_contrast + β * L_recon with α=0.5, β=0.1.

The contrastive loss is the key innovation over standard distillation. Pure behavioral cloning (L_BC only) produces a student that matches actions but may fail to generalize because its latent space lacks structure. The contrastive loss imposes geometric structure: similar environmental states map to nearby latent codes, enabling smooth interpolation and generalization to unseen terrain configurations. The negative samples in the InfoNCE loss are critical — they prevent mode collapse where all states map to the same latent code.

The transformer-based student encoder uses causal masking to ensure temporal causality and positional encodings to capture temporal structure in the proprioceptive history. The final token's representation is used as l_S. Compared to the GRU encoder, the transformer achieves 5–8% better performance on tasks requiring long-range temporal reasoning (e.g., inferring terrain type from multiple footstep contacts over 1–2 seconds).

An ablation study demonstrates the contribution of each loss component: L_BC alone achieves baseline performance; adding L_contrast improves rough terrain performance by 15%; adding L_recon provides an additional 5–10% improvement. The contrastive loss contributes most when terrain variation is high (many terrain types), while the reconstruction loss helps most when terrain features have direct action relevance (e.g., step height directly determines foot placement).

## Key Results & Numbers
- Combined approach outperforms action-only distillation by 15–25% on rough terrain locomotion (return metric)
- Contrastive alignment alone (no reconstruction) provides 15% improvement over baseline distillation
- Adding reconstruction provides additional 5–10% improvement, primarily on tasks with high terrain-action correlation
- Transformer student encoder outperforms GRU by 5–8% on tasks requiring long-range temporal reasoning (>1s)
- Latent space probing: combined approach terrain classification accuracy 87% vs. action-only 62% vs. contrastive-only 82%
- Real-world Unitree A1: traverses 20cm gaps and 15° slopes with 90% success vs. 72% for action-only distillation
- Real-world Go1: gait switching latency reduced from 0.8s to 0.3s with combined approach
- Training time: teacher (4h) + student with combined loss (2h) = 6h total on 4 GPUs

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This approach directly enhances the Mini Cheetah teacher-student pipeline by adding representation learning objectives. The contrastive alignment loss ensures the student encoder learns a structured latent space that captures terrain properties, improving generalization to unseen terrains during deployment. The transformer-based student encoder is particularly relevant for Mini Cheetah's 12-DoF proprioceptive observation space, where temporal patterns over multiple gait cycles encode terrain information.

The practical training recipe (α=0.5, β=0.1) and the finding that contrastive loss contributes most of the improvement provide a clear implementation guide. The technique is additive — it can be incorporated into existing Mini Cheetah training pipelines with minimal architectural changes (add contrastive loss head to student encoder).

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is directly relevant to Cassie's Dual Asymmetric-Context Transformer. The contrastive representation learning framework provides a principled method for training the deployment context stream to learn structured representations that mirror the privileged context stream's information content. The InfoNCE loss between privileged and deployment context latents is analogous to (and can replace) simpler MSE alignment losses.

The transformer-based student encoder validates the use of causal transformers for proprioceptive history processing in Cassie's architecture. The finding that transformers outperform GRUs for long-range temporal reasoning supports Cassie's choice of transformer architecture over recurrent alternatives. The contrastive loss also provides a natural pre-training objective for the deployment context stream before end-to-end fine-tuning.

## What to Borrow / Implement
- Add InfoNCE contrastive loss (α=0.5) between student and teacher latent spaces in Mini Cheetah's distillation pipeline
- Implement terrain reconstruction auxiliary loss (β=0.1) from student latent to heightmap as additional representation supervision
- Use transformer-based student encoder (4 layers, 4 heads) for Mini Cheetah proprioceptive history when long-range temporal reasoning is needed
- Apply contrastive alignment as the cross-context training objective for Cassie's Dual Asymmetric-Context Transformer
- Add latent space probing (terrain classification from l_S) as a diagnostic for representation quality during training

## Limitations & Open Questions
- Teacher must be fully trained before student training begins (sequential, not concurrent); combining with CTS-style concurrent training is an open question
- Contrastive loss quality depends heavily on batch size and negative sampling strategy; small batches may lead to poor representation learning
- The reconstruction loss requires defining which privileged features to predict, introducing a manual design choice
- Transformer student encoder's quadratic attention cost may limit proprioceptive history length on resource-constrained embedded systems
