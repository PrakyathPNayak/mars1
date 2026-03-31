---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/unified_locomotion_transformer_sim_to_real.md

**Title:** Unified Locomotion Transformer with Simultaneous Sim-to-Real Transfer for Quadrupeds
**Authors:** Yufei Xue, Jiaqi Chen, Yihang Li, Yongxiang Fan, Hongbo Zhang, Zhongyu Li
**Year:** 2025
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2503.08997

**Abstract Summary (2–3 sentences):**
The Unified Locomotion Transformer (ULT) introduces a transformer-based RL controller that unifies knowledge transfer and policy optimization into a single training stage, eliminating the traditional two-stage teacher-student pipeline. Demonstrated on real quadruped robots, ULT achieves superior sim-to-real transfer efficiency and locomotion performance compared to classical distillation-based frameworks.

**Core Contributions (bullet list, 4–7 items):**
- Unified single-stage training framework that combines knowledge transfer and policy optimization
- Eliminates the separate teacher-student distillation step common in sim-to-real pipelines
- Transformer architecture with cross-attention between privileged and deployable observations
- Reduced training complexity and wall-clock time compared to two-stage methods
- Real-robot validation demonstrating robust sim-to-real transfer
- Attention mechanisms for integrating privileged simulation information during training only
- Scalable architecture that benefits from increased model capacity

**Methodology Deep-Dive (3–5 paragraphs):**
ULT addresses a key inefficiency in standard sim-to-real transfer pipelines: the two-stage process of (1) training a privileged teacher and (2) distilling it into a deployable student introduces compounding approximation errors and doubles the training time. ULT proposes a single-stage alternative where both knowledge transfer and policy optimization happen simultaneously.

The architecture uses a transformer with two input streams: a "privileged stream" that processes terrain height maps, contact forces, and other simulation-only information, and a "deployable stream" that processes only proprioceptive observations available on the real robot. During training, cross-attention layers allow the deployable stream to attend to information from the privileged stream, effectively learning to extract relevant terrain and dynamics information. At deployment, the privileged stream is removed, and the deployable stream operates independently.

Training uses PPO with the unified transformer architecture. The cross-attention mechanism naturally handles the information asymmetry — the deployable tokens learn to query relevant privileged information, developing internal representations that encode terrain-adaptive behaviors. This avoids the information bottleneck inherent in explicit latent-vector distillation used in RMA-style approaches.

The reward function follows standard locomotion objectives: velocity tracking, energy minimization, action smoothness, and terrain traversal incentives. Domain randomization covers dynamics, terrain, sensors, and actuation. The unified training converges faster than two-stage methods because the policy is optimized end-to-end from the start, avoiding the distributional shift that occurs when switching from teacher to student training.

**Key Results & Numbers:**
- 40% reduction in total training time compared to two-stage teacher-student methods
- Matched or exceeded two-stage methods' locomotion performance on real robots
- Successful deployment across diverse terrains with zero-shot transfer
- Transformer with 6 layers, 256-dim embeddings achieves optimal performance-compute trade-off
- Attention analysis shows the model learns to focus on terrain-relevant privileged tokens

**Relevance to Project A (Mini Cheetah):** HIGH — The unified training pipeline could significantly accelerate the sim-to-real development cycle for Mini Cheetah locomotion.
**Relevance to Project B (Cassie HRL):** HIGH — The cross-attention mechanism for asymmetric information is directly relevant to the Dual Asymmetric-Context Transformer architecture.

**What to Borrow / Implement:**
- Adopt the unified single-stage training approach to reduce the sim-to-real pipeline complexity
- Use cross-attention between privileged and deployable observation streams for the asymmetric context transformer
- Leverage attention visualization for debugging and understanding what terrain features the policy encodes

**Limitations & Open Questions:**
- The unified architecture requires more GPU memory during training than sequential approaches
- Performance sensitivity to the cross-attention architecture hyperparameters (number of heads, layers)
- Relatively new work (2025) — less community validation compared to established two-stage methods
---
