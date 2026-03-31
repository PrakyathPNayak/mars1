# MACRPO: Multi-Agent Cooperative Recurrent Policy Optimization

**Authors:** Various
**Year:** 2024 | **Venue:** Frontiers in Robotics and AI
**Links:** https://www.frontiersin.org/articles/10.3389/frobt.2024.1394209

---

## Abstract Summary
Extends PPO with LSTM-based recurrent policies for multi-agent cooperative settings under partial observability. Demonstrates that recurrent policies significantly outperform feedforward alternatives when agents must coordinate while having limited observation of other agents and the environment. The work provides a scalable framework for cooperative multi-agent RL that maintains training stability while enabling emergent coordination behaviors.

## Core Contributions
- Introduces MACRPO: a multi-agent extension of PPO with LSTM-based recurrent policies for cooperative tasks
- Demonstrates that LSTM-PPO outperforms feedforward PPO by 25-40% in POMDP multi-agent settings
- Proposes a centralized-training-decentralized-execution (CTDE) framework with shared recurrent critic
- Shows emergent coordination behaviors arise from independent recurrent policies without explicit communication
- Achieves scalable training with up to 8 cooperative agents while maintaining training stability
- Provides ablation studies isolating the contributions of recurrence, centralized training, and reward sharing
- Validates on cooperative locomotion and manipulation tasks requiring multi-agent coordination

## Methodology Deep-Dive
MACRPO builds on the Proximal Policy Optimization (PPO) algorithm, extending it to multi-agent cooperative settings with three key modifications: LSTM-based recurrent policies, centralized training with a shared value function, and a cooperative reward structure. Each agent maintains its own LSTM-based policy network that maps local observations (partial view of the environment and other agents) to actions. The LSTM hidden state accumulates information over time, enabling agents to infer unobserved aspects of the environment and other agents' behaviors from interaction history.

The centralized-training-decentralized-execution (CTDE) paradigm is implemented through a shared critic network. During training, the critic receives the concatenated observations and hidden states of all agents, allowing it to estimate the joint value function that accounts for cooperative dynamics. During execution, each agent's policy operates independently using only its local observation and LSTM hidden state. The shared critic provides a more accurate training signal than independent critics, as it can reason about the joint state and the effects of cooperation. This reduces the non-stationarity problem inherent in independent multi-agent learning.

The cooperative reward structure combines individual task rewards with a shared team reward. Individual rewards incentivize each agent's local objectives (e.g., velocity tracking for a locomotion agent, reaching for a manipulation agent), while the team reward incentivizes global objectives that require coordination (e.g., synchronized movement, load sharing, formation maintenance). The balance between individual and team rewards is controlled by a mixing coefficient that is annealed during training: early training emphasizes individual rewards (for basic skill learning), while later training increases team reward weight (for coordination).

The LSTM architecture uses a 2-layer LSTM with 128 hidden units per agent, processing a sequence of the last 16 observations. A key design choice is separate LSTM modules for the policy and value networks, preventing the value function's training dynamics from interfering with the policy's memory formation. The paper shows that shared LSTMs between policy and value lead to training instability in multi-agent settings, while separate LSTMs maintain stable learning dynamics.

Scalability is addressed through parameter sharing and batched environment execution. Agents in the same role share policy parameters (but maintain independent LSTM hidden states), reducing the number of trainable parameters from O(N) to O(1) in the number of agents per role. Training uses parallel environments (1024) with batched LSTM forward passes for computational efficiency.

## Key Results & Numbers
- LSTM-PPO outperforms feedforward PPO by 25-40% in cooperative POMDP tasks
- Centralized critic improves learning speed by 2x compared to independent critics
- Scalable to 8 cooperative agents with < 15% performance degradation vs. 2 agents
- Separate policy/value LSTMs reduce training variance by 30% vs. shared LSTMs
- Cooperative reward annealing improves final performance by 10% vs. fixed mixing
- Emergent coordination behaviors: synchronized gait, load distribution, formation maintenance
- Training time scales sub-linearly with number of agents due to parameter sharing
- 2-layer LSTM (128 hidden) optimal; 3-layer provides marginal improvement at 50% higher cost

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
While Mini Cheetah is a single-agent system, the recurrent PPO framework is applicable when operating under partial observability (blind locomotion without vision or terrain estimation). The LSTM-PPO architecture can serve as a drop-in replacement for the feedforward PPO policy when Mini Cheetah needs to infer terrain properties from proprioceptive history. The finding that 2-layer LSTM with 128 hidden units is optimal aligns with the memory architecture study's recommendations. The multi-agent aspect becomes relevant if considering multi-robot coordination scenarios (e.g., cooperative payload transport) or if modeling each leg as a semi-independent agent in a cooperative framework, though this is a non-standard approach.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
The CTDE framework conceptually parallels Project B's hierarchical architecture, where different hierarchy levels can be viewed as cooperating agents: the Planner "agent" provides high-level commands, the Primitives "agent" selects motion skills, the Controller "agent" generates joint commands, and the Safety "agent" enforces constraints. The centralized critic's ability to see all agents' states during training mirrors the Dual Asymmetric-Context Transformer's access to privileged information. The cooperative reward structure with annealing provides a template for designing inter-level reward signals in the hierarchy. The LSTM recurrence findings complement the Transformer-based architecture choice by providing a baseline comparison. The Option-Critic framework in Project B shares the hierarchical cooperative structure with MACRPO's multi-agent approach.

## What to Borrow / Implement
- Implement LSTM-PPO as a baseline for blind locomotion on Mini Cheetah (2-layer, 128 hidden, 16-step history)
- Adopt separate LSTM modules for policy and value networks to improve training stability
- Apply the cooperative reward annealing strategy to inter-level reward design in Project B's hierarchy
- Use the CTDE framework as a conceptual template for training hierarchical policies with privileged critics
- Implement parameter sharing across identical components (e.g., per-leg controllers) if applicable
- Use the emergent coordination analysis methodology to study inter-level coordination in Project B

## Limitations & Open Questions
- Multi-agent cooperative framework may be over-engineered for single-robot locomotion tasks
- The analogy between multi-agent cooperation and hierarchical control is imperfect; hierarchy has explicit information flow
- LSTM-based policies have been largely superseded by Transformer-based approaches in recent work
- Scalability beyond 8 agents and to heterogeneous agent teams is not demonstrated
- Communication between agents is implicit through the environment; explicit communication channels may improve coordination
- How to handle asynchronous agent execution (different agents operating at different frequencies) is not addressed
- Real-world deployment of multi-agent LSTM policies on resource-constrained hardware is not evaluated
