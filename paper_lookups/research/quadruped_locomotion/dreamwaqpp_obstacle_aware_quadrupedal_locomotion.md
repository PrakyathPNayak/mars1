---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/dreamwaqpp_obstacle_aware_quadrupedal_locomotion.md

**Title:** DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion With Resilient Multi-Modal Reinforcement Learning
**Authors:** I Made Aswin Nahrendra, Byeongho Yu, Minho Oh, Hyun Myung
**Year:** 2024
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2409.19709

**Abstract Summary (2–3 sentences):**
DreamWaQ++ extends DreamWaQ by integrating both proprioceptive and exteroceptive (depth camera) sensor modalities within a resilient multi-modal RL framework. The system maintains robust locomotion even under sensor failures or degradation, achieving obstacle-aware navigation with significantly improved agility and terrain adaptability compared to its predecessor.

**Core Contributions (bullet list, 4–7 items):**
- Resilient multi-modal fusion of proprioceptive and exteroceptive (depth) sensing
- Graceful degradation under sensor failure — maintains locomotion if cameras fail
- Obstacle-aware locomotion enabling active avoidance and step-over behaviors
- Improved terrain adaptability and agility compared to proprioceptive-only DreamWaQ
- Multi-modal attention mechanism for dynamically weighting sensor modalities
- Real-world deployment demonstrating robustness to sensor noise and occlusion
- Extended evaluation across diverse indoor and outdoor obstacle courses

**Methodology Deep-Dive (3–5 paragraphs):**
DreamWaQ++ addresses the fundamental limitation of its predecessor: while DreamWaQ's proprioceptive-only approach is robust, it cannot anticipate obstacles or terrain discontinuities that require proactive foot placement. DreamWaQ++ adds depth camera input while maintaining resilience to exteroceptive sensor failures — a critical requirement for real-world deployment where sensors can be occluded, damaged, or noisy.

The architecture features a multi-modal encoder with separate branches for proprioceptive and exteroceptive inputs. A proprioceptive branch processes joint states and IMU data through a temporal network (similar to DreamWaQ), while a visual branch processes depth images through a convolutional encoder. A multi-modal attention mechanism dynamically weights the contribution of each modality based on estimated reliability, enabling graceful degradation — if the depth camera fails, the system smoothly transitions to proprioceptive-only operation.

Training uses a teacher-student framework in Isaac Gym with PPO. The teacher has access to ground-truth terrain geometry, while the student learns from noisy sensor inputs. A key training innovation is "modal dropout" — randomly masking the exteroceptive input during training episodes. This forces the proprioceptive branch to maintain standalone locomotion capability, ensuring the robot doesn't become overly dependent on vision.

The reward function extends DreamWaQ's with additional terms for obstacle clearance, step-over height, and proactive velocity modulation near obstacles. Domain randomization covers camera noise, latency, field-of-view variations, and random occlusions in addition to the standard dynamics randomization.

**Key Results & Numbers:**
- 35% improvement in obstacle course completion rate compared to DreamWaQ
- Maintained 90% of performance with complete camera failure (graceful degradation)
- Successfully navigated obstacles up to 25 cm height with step-over behavior
- Stable locomotion at speeds up to 2.0 m/s on mixed terrain with obstacles
- Real-world deployment on Unitree Go1 across indoor obstacle courses and outdoor trails

**Relevance to Project A (Mini Cheetah):** HIGH — The resilient multi-modal approach provides a natural upgrade path from proprioceptive-only locomotion to obstacle-aware navigation for Mini Cheetah.
**Relevance to Project B (Cassie HRL):** MEDIUM — The multi-modal attention mechanism and graceful degradation concepts are relevant to the CPTE (contrastive terrain encoder) component.

**What to Borrow / Implement:**
- Implement modal dropout during training to ensure proprioceptive fallback capability
- Adopt the multi-modal attention mechanism for sensor fusion in terrain-aware locomotion
- Use the graceful degradation framework as a safety feature for real-world deployment

**Limitations & Open Questions:**
- Depth camera provides limited range; cannot plan for distant terrain features
- Multi-modal training requires significantly more compute than proprioceptive-only training
- The attention-based weighting of modalities may not adapt quickly enough to sudden sensor failures
---
