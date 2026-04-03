"""
Advanced Hierarchical Transformer Policy for Quadruped Locomotion.

Architecture inspired by:
  - ULT (2503.08997): Unified transformer framework with simultaneous knowledge transfer
  - TERT (2212.07740): Terrain Transformer with privileged training
  - MSTA (2409.03332): Masked Sensory-Temporal Attention for sensor generalization
  - SET (2410.13496): State Estimation Transformers with causal masking
  - MoE-Loco (2503.08564): Mixture of Experts for multitask locomotion
  - MS-PPO (2512.00727): Morphological-Symmetry-Equivariant Policy
  - DWL (2408.14472): Denoising World Model auxiliary prediction
  - LP-ACRL (2601.17428): Learning Progress-based Automatic Curriculum

Key innovations:
  1. Temporal Transformer Encoder: Processes observation history with causal attention
  2. Sensory-Group Attention: Structured attention over joint/velocity/IMU/command groups
  3. Mixture of Experts: Terrain-specialized expert sub-networks with learned gating
  4. Morphological Symmetry: Encodes quadruped bilateral symmetry to reduce params
  5. Auxiliary World Model Head: Next-state prediction for representation learning
  6. Adaptive Curriculum: Learning progress-based terrain difficulty adjustment
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────

OBS_DIM = 49
ACT_DIM = 12
HISTORY_LEN = 16  # Number of past observations to keep

# Sensory groups for structured attention
# Must match cheetah_env.py observation layout exactly (49 dims).
SENSORY_GROUPS = {
    "joint_pos":   (0,  12),   # 12 joint positions
    "joint_vel":   (12, 24),   # 12 joint velocities
    "base_linvel": (24, 27),   # 3 base linear velocity
    "base_angvel": (27, 30),   # 3 base angular velocity
    "gravity":     (30, 33),   # 3 projected gravity
    "prev_action": (33, 45),   # 12 previous action
    "command":     (45, 49),   # 4: vx, vy, wz, target_height
}

# Leg symmetry mapping: FR<->FL, RR<->RL (3 joints per leg)
# Index mapping for bilateral symmetry reflection
LEG_PAIRS = [(0, 3), (1, 4), (2, 5),   # FR joints <-> FL joints
             (6, 9), (7, 10), (8, 11)]  # RR joints <-> RL joints


# ── Observation Normalization ─────────────────────────────────────────

class RunningNormalizer(nn.Module):
    """Online running mean/std normalization for observations.

    Critical for transformer-based policies which are sensitive to input scale.
    Uses Welford's algorithm for numerical stability.
    """

    def __init__(self, shape: int, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("count", torch.tensor(1e-4))
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Update running statistics (call during training only)."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics."""
        return torch.clamp(
            (x - self.mean) / (self.var.sqrt() + 1e-8),
            -self.clip, self.clip
        )


# ── Positional Encoding ──────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ── Sensory Group Encoder ────────────────────────────────────────────

class SensoryGroupEncoder(nn.Module):
    """Encode each sensory group independently then fuse via cross-attention.

    Inspired by MSTA (2409.03332) which uses sensor-level attention.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # Per-group linear projections
        self.group_encoders = nn.ModuleDict()
        for name, (start, end) in SENSORY_GROUPS.items():
            dim = end - start
            self.group_encoders[name] = nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )

        # Cross-group attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True, dropout=0.05
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) single observation

        Returns:
            fused: (batch, d_model) fused sensory representation
        """
        group_tokens = []
        for name, (start, end) in SENSORY_GROUPS.items():
            group_obs = obs[..., start:end]
            token = self.group_encoders[name](group_obs)
            group_tokens.append(token)

        # Stack into sequence: (batch, n_groups, d_model)
        tokens = torch.stack(group_tokens, dim=-2)

        # Self-attention across sensory groups
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)

        # Mean pool across groups
        fused = tokens.mean(dim=-2)
        return fused


# ── Morphological Symmetry Module ────────────────────────────────────

class SymmetryAugmenter(nn.Module):
    """Encodes bilateral symmetry of the quadruped.

    Inspired by MS-PPO (2512.00727). Uses precomputed permutation indices
    for efficient reflection without re-encoding through the full network.
    Instead, uses a lightweight symmetry projection on the raw reflected obs.
    """

    def __init__(self, d_model: int = 128, obs_dim: int = OBS_DIM):
        super().__init__()
        # Lightweight encoder for reflected obs (much cheaper than full sensory encoder)
        self.reflect_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Precompute reflection permutation indices for efficiency
        self._build_reflection_indices(obs_dim)

    def _build_reflection_indices(self, obs_dim):
        """Build permutation + sign arrays for vectorized reflection."""
        perm = list(range(obs_dim))
        sign = [1.0] * obs_dim

        for left, right in LEG_PAIRS:
            # Swap joint positions
            perm[left], perm[right] = right, left
            # Swap joint velocities
            perm[left + 12], perm[right + 12] = right + 12, left + 12
            # Swap previous actions
            perm[left + 33], perm[right + 33] = right + 33, left + 33

        # Negate lateral velocity and yaw rate
        sign[25] = -1.0  # vy
        sign[29] = -1.0  # wz
        sign[46] = -1.0  # vy_cmd
        sign[47] = -1.0  # wz_cmd
        # sign[48] = +1.0 (target_height is symmetric, no flip needed)

        self.register_buffer("_perm", torch.tensor(perm, dtype=torch.long))
        self.register_buffer("_sign", torch.tensor(sign, dtype=torch.float32))

    def reflect_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Vectorized reflection (no loops, GPU-friendly)."""
        return obs[..., self._perm] * self._sign

    def forward(
        self, obs: torch.Tensor, encoded: torch.Tensor,
        encode_fn=None  # encode_fn no longer needed
    ) -> torch.Tensor:
        """
        Args:
            obs: raw observation (batch, obs_dim)
            encoded: encoded observation (batch, d_model)
            encode_fn: UNUSED (kept for API compat), replaced by lightweight encoder

        Returns:
            augmented: (batch, d_model) symmetry-augmented representation
        """
        reflected_obs = self.reflect_obs(obs)
        reflected_encoded = self.reflect_encoder(reflected_obs)
        combined = torch.cat([encoded, reflected_encoded], dim=-1)
        return self.projection(combined)


# ── Temporal Transformer Block ────────────────────────────────────────

class TemporalTransformerBlock(nn.Module):
    """Single transformer block with causal masking for temporal processing.

    Inspired by SET (2410.13496) and TERT (2212.07740).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256,
                 dropout: float = 0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # Self-attention with causal mask
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ── Mixture of Experts ────────────────────────────────────────────────

class ExpertNetwork(nn.Module):
    """Single expert MLP."""

    def __init__(self, d_input: int, d_output: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with learned gating and load-balancing.

    Inspired by MoE-Loco (2503.08564). Includes load-balancing loss to
    prevent expert collapse (all traffic going to one expert).
    """

    def __init__(self, d_input: int, d_output: int, n_experts: int = 4,
                 d_hidden: int = 128, load_balance_coeff: float = 0.01):
        super().__init__()
        self.n_experts = n_experts
        self.load_balance_coeff = load_balance_coeff
        self.experts = nn.ModuleList([
            ExpertNetwork(d_input, d_output, d_hidden)
            for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_experts),
        )
        # Track load-balance loss for logging
        self._last_balance_loss = 0.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: weighted combination of experts
            gate_weights: expert selection probabilities for analysis
        """
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-2
        )  # (batch, n_experts, d_output)

        output = torch.einsum("bn,bnd->bd", gate_weights, expert_outputs)

        # Load-balancing loss: encourage uniform expert usage
        # (from Switch Transformer / MoE-Loco)
        # f_i = fraction of tokens dispatched to expert i
        # p_i = mean gate probability for expert i
        # loss = N * sum(f_i * p_i)  -- minimized when uniform
        if self.training:
            f = gate_weights.mean(dim=0)  # (n_experts,)
            p = F.softmax(gate_logits, dim=-1).mean(dim=0)  # (n_experts,)
            balance_loss = self.n_experts * (f * p).sum()
            self._last_balance_loss = balance_loss.item()
            # Straight-through estimator: adds zero to forward pass but
            # backpropagates the balance loss gradient through gate params
            auxiliary = self.load_balance_coeff * balance_loss
            output = output + (auxiliary - auxiliary.detach())
            # Store for external access
            self._balance_loss = balance_loss

        return output, gate_weights

    def get_balance_loss(self) -> torch.Tensor:
        """Return the load-balance loss (call after forward)."""
        if hasattr(self, '_balance_loss'):
            return self._balance_loss * self.load_balance_coeff
        return torch.tensor(0.0)


# ── Auxiliary World Model Head ────────────────────────────────────────

class WorldModelHead(nn.Module):
    """Predicts next observation given current state + action.

    Inspired by DWL (2408.14472). Acts as auxiliary loss to improve
    representation quality.
    """

    def __init__(self, d_model: int = 128, action_dim: int = ACT_DIM,
                 obs_dim: int = OBS_DIM):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model + action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next observation."""
        combined = torch.cat([latent, action], dim=-1)
        return self.predictor(combined)


# ── Main Architecture: Hierarchical Transformer Policy ────────────────

class HierarchicalTransformerPolicy(nn.Module):
    """Complete hierarchical transformer architecture for quadruped locomotion.

    Architecture overview:
        1. Sensory Group Encoder: Process each sensor type independently
        2. Symmetry Augmentation: Exploit bilateral symmetry
        3. Temporal Transformer: Process observation history with causal attention
        4. Mixture of Experts: Terrain-specialized action generation
        5. Value Head: Separate value estimation branch
        6. World Model Head: Auxiliary next-state prediction
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACT_DIM,
        d_model: int = 128,
        n_transformer_layers: int = 3,
        n_heads: int = 4,
        n_experts: int = 4,
        history_len: int = HISTORY_LEN,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.history_len = history_len

        # 1. Observation Normalization
        self.obs_normalizer = RunningNormalizer(obs_dim)

        # 2. Sensory Group Encoder
        self.sensory_encoder = SensoryGroupEncoder(d_model)

        # 3. Morphological Symmetry
        self.symmetry_aug = SymmetryAugmenter(d_model, obs_dim)

        # 3. Temporal Transformer
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=history_len)
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_transformer_layers)
        ])

        # 4. Policy Head via MoE
        self.policy_moe = MixtureOfExperts(
            d_model, action_dim, n_experts=n_experts, d_hidden=256
        )
        self.action_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # 5. Value Head (separate)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # 6. World Model Head (auxiliary)
        self.world_model = WorldModelHead(d_model, action_dim, obs_dim)

        # 7. Gait Phase Oscillator
        self.phase_oscillator = GaitPhaseOscillator(d_model)

        # 8. Terrain Estimator
        self.terrain_estimator = TerrainEstimator(d_model)

        # 9. Contrastive Temporal Head (auxiliary)
        self.contrastive_head = ContrastiveTemporalHead(d_model)

        # Fusion layer for phase + terrain + transformer features
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Observation history buffer (managed externally during rollout)
        self._history_buffer = None
        self._batch_size = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with careful scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small init for policy output
        for expert in self.policy_moe.experts:
            nn.init.orthogonal_(expert.net[-1].weight, gain=0.01)
        # Small init for value output
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _generate_causal_mask(self, seq_len: int, device: torch.device
                              ) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        return mask

    def encode_single_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a single observation through sensory groups."""
        return self.sensory_encoder(obs)

    def forward(
        self,
        obs_history: torch.Tensor,
        return_value: bool = True,
        action_for_wm: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_history: (batch, seq_len, obs_dim) observation history
            return_value: whether to compute value estimate
            action_for_wm: (batch, action_dim) action for world model prediction
            step_count: (batch,) episode step count for phase oscillator

        Returns:
            dict with 'action_mean', 'action_log_std', 'value', 'gate_weights',
            'next_obs_pred', 'latent', 'terrain_latent', 'contrastive_loss'
        """
        batch_size, seq_len, _ = obs_history.shape
        device = obs_history.device

        # Normalize observations
        flat_obs = obs_history.reshape(-1, self.obs_dim)
        if self.training:
            self.obs_normalizer.update(flat_obs)
        flat_obs = self.obs_normalizer.normalize(flat_obs)

        # Encode each timestep through sensory groups
        encoded = self.encode_single_obs(flat_obs)
        encoded = encoded.reshape(batch_size, seq_len, self.d_model)

        # Apply symmetry augmentation on the latest observation
        latest_obs = flat_obs.reshape(batch_size, seq_len, self.obs_dim)[:, -1]
        latest_encoded = encoded[:, -1]
        sym_encoded = self.symmetry_aug(latest_obs, latest_encoded)
        # Replace last timestep with symmetry-augmented version
        encoded = encoded.clone()
        encoded[:, -1] = sym_encoded

        # Add positional encoding
        encoded = self.pos_encoding(encoded)

        # Temporal transformer with causal masking
        causal_mask = self._generate_causal_mask(seq_len, device)
        x = encoded
        for layer in self.temporal_layers:
            x = layer(x, causal_mask)

        # Temporal output: last token as transformer features
        transformer_latent = x[:, -1]  # (batch, d_model)

        # Terrain estimation from observation history
        norm_history = flat_obs.reshape(batch_size, seq_len, self.obs_dim)
        terrain_features, terrain_latent = self.terrain_estimator(norm_history)

        # Gait phase signal
        if step_count is None:
            step_count = torch.zeros(batch_size, device=device)
        command = latest_obs[..., 45:48]  # velocity command
        phase_features = self.phase_oscillator(step_count, command)

        # Fuse transformer + terrain + phase features
        latent = self.feature_fusion(torch.cat([
            transformer_latent, terrain_features, phase_features
        ], dim=-1))

        # Policy (MoE)
        action_mean, gate_weights = self.policy_moe(latent)

        results = {
            "action_mean": action_mean,
            "action_log_std": self.action_log_std.expand_as(action_mean),
            "gate_weights": gate_weights,
            "latent": latent,
            "terrain_latent": terrain_latent,
        }

        # Value
        if return_value:
            results["value"] = self.value_head(latent)

        # World Model prediction
        if action_for_wm is not None:
            results["next_obs_pred"] = self.world_model(latent, action_for_wm)

        # Contrastive temporal loss (auxiliary)
        if self.training:
            results["contrastive_loss"] = self.contrastive_head.compute_loss(x)

        return results


# ── SB3-Compatible Feature Extractor ──────────────────────────────────

class TransformerFeaturesExtractor(nn.Module):
    """Wraps the hierarchical transformer for use as an SB3 features extractor.

    Manages observation history buffer internally.
    """

    def __init__(self, observation_space, d_model: int = 128,
                 history_len: int = HISTORY_LEN):
        super().__init__()
        self.d_model = d_model
        self.history_len = history_len
        self.obs_dim = observation_space.shape[0]
        self.features_dim = d_model  # output dimension

        self.sensory_encoder = SensoryGroupEncoder(d_model)
        self.symmetry_aug = SymmetryAugmenter(d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=history_len)
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, 4, d_model * 2, 0.05)
            for _ in range(3)
        ])

        # History buffer
        self._history = None
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _generate_causal_mask(self, seq_len: int, device: torch.device
                              ) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        return mask

    def _update_history(self, obs: torch.Tensor) -> torch.Tensor:
        """Maintain rolling history buffer."""
        batch_size = obs.shape[0]
        device = obs.device

        if self._history is None or self._history.shape[0] != batch_size:
            self._history = obs.unsqueeze(1).repeat(1, self.history_len, 1)
        else:
            self._history = torch.cat([
                self._history[:, 1:],
                obs.unsqueeze(1)
            ], dim=1)

        return self._history.clone()

    def encode_single(self, obs: torch.Tensor) -> torch.Tensor:
        return self.sensory_encoder(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) current observation

        Returns:
            features: (batch, d_model) extracted features
        """
        # Update history
        history = self._update_history(obs)
        batch_size, seq_len, _ = history.shape
        device = obs.device

        # Encode each timestep
        flat = history.reshape(-1, self.obs_dim)
        encoded = self.encode_single(flat)
        encoded = encoded.reshape(batch_size, seq_len, self.d_model)

        # Symmetry on latest
        latest_enc = encoded[:, -1]
        sym_enc = self.symmetry_aug(obs, latest_enc)
        encoded = encoded.clone()
        encoded[:, -1] = sym_enc

        # Positional encoding + temporal transformer
        encoded = self.pos_encoding(encoded)
        causal_mask = self._generate_causal_mask(seq_len, device)
        x = encoded
        for layer in self.temporal_layers:
            x = layer(x, causal_mask)

        return x[:, -1]  # Final token

    def reset_history(self):
        """Reset history buffer (call on env reset)."""
        self._history = None


# ── MoE Action Head (SB3-compatible) ─────────────────────────────────

class MoEActionNet(nn.Module):
    """MoE-based action network compatible with SB3's ActorCritcPolicy."""

    def __init__(self, feature_dim: int, action_dim: int = ACT_DIM,
                 n_experts: int = 4):
        super().__init__()
        self.moe = MixtureOfExperts(feature_dim, action_dim, n_experts, d_hidden=256)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        action_mean, _ = self.moe(features)
        return action_mean


class ValueNet(nn.Module):
    """Separate value network head."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ── Adaptive Curriculum ───────────────────────────────────────────────

class AdaptiveCurriculum:
    """Learning Progress-based Automatic Curriculum.

    Inspired by LP-ACRL (2601.17428). Estimates online learning progress
    and adaptively adjusts terrain difficulty without fixed thresholds.
    """

    TERRAIN_LEVELS = [
        {"name": "flat",         "height_noise": 0.0,  "step_height": 0.0,  "slope": 0.0},
        {"name": "slight_rough", "height_noise": 0.02, "step_height": 0.0,  "slope": 0.0},
        {"name": "rough",        "height_noise": 0.05, "step_height": 0.0,  "slope": 0.05},
        {"name": "rough_slope",  "height_noise": 0.05, "step_height": 0.0,  "slope": 0.15},
        {"name": "low_steps",    "height_noise": 0.03, "step_height": 0.05, "slope": 0.0},
        {"name": "high_steps",   "height_noise": 0.03, "step_height": 0.15, "slope": 0.0},
        {"name": "stairs",       "height_noise": 0.02, "step_height": 0.20, "slope": 0.0},
        {"name": "rough_stairs", "height_noise": 0.05, "step_height": 0.20, "slope": 0.05},
    ]

    def __init__(self, n_envs: int, window_size: int = 50,
                 progress_threshold: float = 0.05):
        self.n_envs = n_envs
        self.window_size = window_size
        self.progress_threshold = progress_threshold
        self.levels = np.zeros(n_envs, dtype=int)

        # Rolling reward windows per env per level
        self.reward_history = {
            i: {l: [] for l in range(len(self.TERRAIN_LEVELS))}
            for i in range(n_envs)
        }

    def record_episode(self, env_id: int, reward: float):
        """Record episode reward and compute learning progress."""
        level = self.levels[env_id]
        history = self.reward_history[env_id][level]
        history.append(reward)

        # Keep window bounded
        if len(history) > self.window_size * 2:
            history[:] = history[-self.window_size * 2:]

        # Compute learning progress: improvement in recent vs old rewards
        if len(history) >= self.window_size:
            mid = len(history) // 2
            old_mean = np.mean(history[:mid])
            new_mean = np.mean(history[mid:])
            progress = (new_mean - old_mean) / (abs(old_mean) + 1e-8)

            # If performance is high and learning progress is plateauing, advance
            if progress < self.progress_threshold and new_mean > 0:
                self.levels[env_id] = min(
                    level + 1, len(self.TERRAIN_LEVELS) - 1
                )
            # If performance is declining significantly, retreat
            elif new_mean < old_mean * 0.5 and level > 0:
                self.levels[env_id] = max(level - 1, 0)

    def get_terrain_config(self, env_id: int) -> dict:
        return self.TERRAIN_LEVELS[self.levels[env_id]]

    def summary(self) -> str:
        dist = np.bincount(self.levels, minlength=len(self.TERRAIN_LEVELS))
        avg = self.levels.mean()
        parts = [f"L{i}:{c}" for i, c in enumerate(dist) if c > 0]
        return (f"Curriculum: avg_level={avg:.1f}, "
                f"distribution=[{', '.join(parts)}]")


# ── Gait Phase Oscillator ─────────────────────────────────────────────

class GaitPhaseOscillator(nn.Module):
    """Learned phase oscillator for periodic gait generation.

    Inspired by DeepGait and central pattern generators (CPG).
    Provides a structured phase signal that encodes gait timing,
    reducing the policy's burden of learning periodicity from scratch.

    Produces 4 phase signals (one per leg) with learned frequency and offsets.
    """

    def __init__(self, d_model: int = 128, n_legs: int = 4):
        super().__init__()
        self.n_legs = n_legs
        # Learned base frequency (Hz) and phase offsets per leg
        self.base_freq = nn.Parameter(torch.tensor(2.0))  # ~2 Hz trot
        self.phase_offsets = nn.Parameter(
            torch.tensor([0.0, 3.14159, 3.14159, 0.0])  # trot pattern
        )
        # Speed-dependent frequency modulation from command
        self.freq_modulator = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        # Project phase into d_model space
        self.phase_encoder = nn.Sequential(
            nn.Linear(n_legs * 2, d_model),  # sin + cos per leg
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

    def forward(self, step_count: torch.Tensor, command: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            step_count: (batch,) integer step within episode
            command: (batch, 3) velocity command [vx, vy, wz]
        Returns:
            phase_features: (batch, d_model) encoded phase signal
        """
        # Modulate frequency based on command speed
        freq_mod = self.freq_modulator(command).squeeze(-1)  # (batch,)
        freq = torch.abs(self.base_freq) + freq_mod  # (batch,)

        # Phase per leg: 2*pi*freq*t + offset
        t = step_count.float() * 0.02  # Convert steps to seconds (dt=0.02)
        phase = 2 * math.pi * freq.unsqueeze(-1) * t.unsqueeze(-1) + self.phase_offsets
        # (batch, n_legs)

        # Encode as sin/cos
        phase_signal = torch.cat([
            torch.sin(phase), torch.cos(phase)
        ], dim=-1)  # (batch, n_legs*2)

        return self.phase_encoder(phase_signal)


# ── Terrain Estimator ─────────────────────────────────────────────────

class TerrainEstimator(nn.Module):
    """Estimates terrain properties from proprioceptive history.

    Inspired by DreamWaQ++ and DWL: infers terrain characteristics
    (roughness, slope, compliance) from foot contact patterns and
    IMU readings over a temporal window.
    """

    def __init__(self, d_model: int = 128, terrain_dim: int = 8):
        super().__init__()
        self.terrain_dim = terrain_dim
        # Processes IMU + foot contact signals
        # Uses gravity vector (3) + base angular velocity (3) + joint velocities (12)
        input_dim = 3 + 3 + 12  # gravity + angvel + joint_vel
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.terrain_head = nn.Sequential(
            nn.Linear(32, terrain_dim),
            nn.LayerNorm(terrain_dim),
        )
        # Project terrain features into model space
        self.terrain_proj = nn.Sequential(
            nn.Linear(terrain_dim, d_model),
            nn.SiLU(),
        )

    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_history: (batch, seq_len, obs_dim) observation history
        Returns:
            terrain_features: (batch, d_model) projected terrain features
            terrain_latent: (batch, terrain_dim) raw terrain embedding
        """
        # Extract proprioceptive signals
        gravity = obs_history[..., 30:33]    # (batch, seq, 3)
        angvel = obs_history[..., 27:30]     # (batch, seq, 3)
        joint_vel = obs_history[..., 12:24]  # (batch, seq, 12)

        x = torch.cat([gravity, angvel, joint_vel], dim=-1)  # (batch, seq, 18)
        x = x.transpose(1, 2)  # (batch, 18, seq) for Conv1d

        x = self.temporal_conv(x).squeeze(-1)  # (batch, 32)
        terrain_latent = self.terrain_head(x)   # (batch, terrain_dim)
        terrain_features = self.terrain_proj(terrain_latent)  # (batch, d_model)

        return terrain_features, terrain_latent


# ── Contrastive Temporal Loss ─────────────────────────────────────────

class ContrastiveTemporalHead(nn.Module):
    """Auxiliary contrastive loss for temporal coherence.

    Encourages temporally adjacent states to have similar representations
    while distant states should differ. Inspired by CPC/BYOL applied to
    locomotion observations.
    """

    def __init__(self, d_model: int = 128, projection_dim: int = 64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, projection_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self._last_loss = 0.0

    def compute_loss(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: (batch, seq_len, d_model) transformer output
        Returns:
            contrastive_loss: scalar
        """
        if temporal_features.shape[1] < 2:
            return torch.tensor(0.0, device=temporal_features.device)

        # Use last two timesteps as positive pair
        z1 = self.projector(temporal_features[:, -2])  # (batch, proj_dim)
        z2 = self.projector(temporal_features[:, -1])  # (batch, proj_dim)

        # Asymmetric: predict z2 from z1
        p1 = self.predictor(z1)

        # Cosine similarity loss (BYOL-style, no negatives needed)
        p1_norm = F.normalize(p1, dim=-1)
        z2_norm = F.normalize(z2.detach(), dim=-1)
        loss = 2 - 2 * (p1_norm * z2_norm).sum(dim=-1).mean()

        self._last_loss = loss.item()
        return loss


# ── Privileged Distillation Interface ─────────────────────────────────

class PrivilegedEncoder(nn.Module):
    """Encodes privileged information available only in simulation.

    For teacher-student distillation (sim-to-real transfer):
    Teacher has access to terrain height map, contact forces, etc.
    Student must learn to infer this from proprioception alone.

    Inspired by RMA (2107.04034) and DWL (2408.14472).
    """

    def __init__(self, privileged_dim: int = 32, d_model: int = 128):
        super().__init__()
        # Privileged info: terrain heightmap samples + contact forces
        # 16 height samples + 4 contact force magnitudes + 4 contact states
        self.encoder = nn.Sequential(
            nn.Linear(privileged_dim, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(privileged_obs)


# ── Utility: count parameters ────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable parameters by component."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    components = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters() if p.requires_grad)
        components[name] = n
    return {"total": total, **components}


# ── Model factory ────────────────────────────────────────────────────

def create_policy(
    obs_dim: int = OBS_DIM,
    action_dim: int = ACT_DIM,
    d_model: int = 128,
    n_layers: int = 3,
    n_experts: int = 4,
    history_len: int = HISTORY_LEN,
) -> HierarchicalTransformerPolicy:
    """Create a hierarchical transformer policy with recommended defaults."""
    return HierarchicalTransformerPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        d_model=d_model,
        n_transformer_layers=n_layers,
        n_heads=4,
        n_experts=n_experts,
        history_len=history_len,
        dropout=0.05,
    )
