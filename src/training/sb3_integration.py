"""
Stable-Baselines3 integration for the Hierarchical Transformer Policy.

Provides:
  - TransformerActorCriticPolicy: drop-in replacement for SB3's MlpPolicy
  - HistoryWrapper: Gymnasium wrapper that provides observation history
  - WorldModelCallback: auxiliary loss for representation learning

Compatible with SB3 PPO (tested with v2.7.x).
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
)

from src.training.advanced_policy import (
    SensoryGroupEncoder,
    SymmetryAugmenter,
    TemporalTransformerBlock,
    SinusoidalPositionalEncoding,
    MixtureOfExperts,
    WorldModelHead,
    RunningNormalizer,
    HISTORY_LEN,
    OBS_DIM,
    ACT_DIM,
)

# ── Observation History Wrapper ───────────────────────────────────────

class ActionSmoothingWrapper(gym.ActionWrapper):
    """Exponential moving average filter on actions for temporal coherence.

    Inspired by DiffuseLoco (2404.19264) action consistency. Prevents
    jerky behavior from noisy policy outputs.
    """

    def __init__(self, env: gym.Env, alpha: float = 0.8):
        super().__init__(env)
        self._alpha = alpha
        self._prev_action = None

    def action(self, action):
        if self._prev_action is None:
            self._prev_action = action.copy()
        smoothed = self._alpha * action + (1 - self._alpha) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)


class HistoryWrapper(gym.ObservationWrapper):
    """Wraps env to stack observation history along a new dimension.

    Transforms obs space from Box(48,) to Box(history_len * 48,) flattened.
    On reset, fills history with the initial observation to avoid discontinuity.
    Uses a smooth fade-in to reduce the impact of sudden history resets on
    the temporal transformer during auto-reset in vectorized envs.
    """

    def __init__(self, env: gym.Env, history_len: int = HISTORY_LEN):
        super().__init__(env)
        self.history_len = history_len
        self._obs_dim = env.observation_space.shape[0]

        # New observation space: (history_len * obs_dim,) flattened for SB3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(history_len * self._obs_dim,),
            dtype=np.float32,
        )
        self._history = deque(maxlen=history_len)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill history with repeated initial observation (uniform start)
        self._history.clear()
        for _ in range(self.history_len):
            self._history.append(obs.copy())
        return self._flatten_history(), info

    def observation(self, obs):
        self._history.append(obs.copy())
        return self._flatten_history()

    def _flatten_history(self) -> np.ndarray:
        return np.concatenate(list(self._history), axis=0).astype(np.float32)


# ── SB3 Features Extractor ───────────────────────────────────────────

class TransformerExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor using the hierarchical transformer.

    Takes flattened observation history (history_len * obs_dim,) and produces
    a latent feature vector (d_model,).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.05,
        history_len: int = HISTORY_LEN,
        obs_dim: int = OBS_DIM,
    ):
        super().__init__(observation_space, features_dim=d_model)
        self.d_model = d_model
        self.obs_dim = obs_dim
        self.history_len = history_len

        # Observation normalization
        self.obs_normalizer = RunningNormalizer(obs_dim)

        # Sensory group encoder
        self.sensory_encoder = SensoryGroupEncoder(d_model)

        # Symmetry augmenter
        self.symmetry_aug = SymmetryAugmenter(d_model, obs_dim)

        # Temporal transformer
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=history_len)
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        import math
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

    def _encode_single(self, obs: torch.Tensor) -> torch.Tensor:
        return self.sensory_encoder(obs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch, history_len * obs_dim) flattened history

        Returns:
            features: (batch, d_model)
        """
        batch_size = observations.shape[0]
        device = observations.device

        # Unflatten: (batch, history_len, obs_dim)
        obs_history = observations.reshape(batch_size, self.history_len, self.obs_dim)

        # Normalize observations
        flat = obs_history.reshape(-1, self.obs_dim)
        if self.training:
            self.obs_normalizer.update(flat)
        flat = self.obs_normalizer.normalize(flat)

        # Encode each timestep
        encoded = self._encode_single(flat)
        encoded = encoded.reshape(batch_size, self.history_len, self.d_model)

        # Symmetry augmentation on latest observation
        latest_obs = flat.reshape(batch_size, self.history_len, self.obs_dim)[:, -1]
        latest_enc = encoded[:, -1]
        sym_enc = self.symmetry_aug(latest_obs, latest_enc)
        encoded = encoded.clone()
        encoded[:, -1] = sym_enc

        # Positional encoding
        encoded = self.pos_encoding(encoded)

        # Causal temporal transformer
        causal_mask = self._generate_causal_mask(self.history_len, device)
        x = encoded
        for layer in self.temporal_layers:
            x = layer(x, causal_mask)

        # Return last token as features
        return x[:, -1]


# ── MoE Policy and Value Networks ────────────────────────────────────

class MoEPolicyNet(nn.Module):
    """Policy network using Mixture of Experts."""

    def __init__(self, feature_dim: int, action_dim: int = ACT_DIM,
                 n_experts: int = 4):
        super().__init__()
        self.moe = MixtureOfExperts(
            feature_dim, action_dim, n_experts=n_experts, d_hidden=256
        )
        self.latent_dim_pi = action_dim  # SB3 reads this

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        action_mean, _ = self.moe(features)
        return action_mean


class CriticNet(nn.Module):
    """Value network head."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.latent_dim_vf = 1  # SB3 reads this

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ── SB3 Actor-Critic Policy ──────────────────────────────────────────

class TransformerActorCriticPolicy(ActorCriticPolicy):
    """Complete SB3 Actor-Critic policy backed by the hierarchical transformer.

    Drop-in replacement for "MlpPolicy" with PPO.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_experts: int = 4,
        history_len: int = HISTORY_LEN,
        obs_dim: int = OBS_DIM,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.history_len = history_len
        self.obs_dim = obs_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerExtractor,
            features_extractor_kwargs=dict(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                history_len=history_len,
                obs_dim=obs_dim,
            ),
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Override to use MoE-based actor and critic instead of MLP."""
        feature_dim = self.features_dim  # d_model from extractor

        self.mlp_extractor = _MoEExtractor(
            feature_dim=feature_dim,
            action_dim=self.action_space.shape[0],
            n_experts=self.n_experts,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Build the policy. Called by ActorCriticPolicy.__init__."""
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf

        self.action_dist = DiagGaussianDistribution(latent_dim_pi)
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=-0.5
        )
        self.value_net = nn.Linear(latent_dim_vf, 1)

        # Initialize weights
        import math
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.orthogonal_(self.value_net.weight, gain=1.0)

        # Optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


class _MoEExtractor(nn.Module):
    """Wraps MoE as an SB3-compatible mlp_extractor.

    SB3 expects mlp_extractor to:
      - have .latent_dim_pi
      - have .latent_dim_vf
      - return (pi_features, vf_features) from forward()
    """

    def __init__(self, feature_dim: int, action_dim: int = ACT_DIM,
                 n_experts: int = 4):
        super().__init__()
        self.moe = MixtureOfExperts(
            feature_dim, action_dim, n_experts=n_experts, d_hidden=256
        )
        self.vf_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.latent_dim_pi = action_dim
        self.latent_dim_vf = 128

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_out, _ = self.moe(features)
        vf_out = self.vf_net(features)
        return pi_out, vf_out

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        pi_out, _ = self.moe(features)
        return pi_out

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.vf_net(features)


# ── World Model Auxiliary Loss Callback ──────────────────────────────

class WorldModelCallback(BaseCallback):
    """Trains a world model auxiliary head alongside PPO.

    Inspired by DWL (2408.14472): uses auxiliary next-state prediction to
    improve the quality of learned representations.
    """

    def __init__(
        self,
        wm_lr: float = 1e-4,
        wm_coeff: float = 0.1,
        update_freq: int = 2048,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.wm_lr = wm_lr
        self.wm_coeff = wm_coeff
        self.update_freq = update_freq
        self._world_model = None
        self._wm_optimizer = None
        self._step_count = 0

    def _on_training_start(self) -> None:
        policy = self.model.policy
        feature_dim = policy.features_dim
        device = self.model.device

        self._world_model = WorldModelHead(
            d_model=feature_dim,
            action_dim=self.model.action_space.shape[0],
            obs_dim=OBS_DIM,
        ).to(device)

        # Only optimize the world model head's own parameters.
        # The features extractor is updated by PPO's optimizer.
        # This avoids the dual-optimizer instability issue.
        self._wm_optimizer = torch.optim.Adam(
            self._world_model.parameters(),
            lr=self.wm_lr,
        )

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._step_count % self.update_freq != 0:
            return True

        try:
            buf = self.model.rollout_buffer
            if buf.full:
                self._train_world_model(buf)
        except Exception:
            pass

        return True

    def _train_world_model(self, buf) -> None:
        """Train world model on rollout buffer data.

        Uses detached features (no gradient to feature extractor) to avoid
        conflicting with PPO's own optimizer on the shared backbone.
        """
        device = self.model.device
        policy = self.model.policy

        # Sample mini-batch from buffer
        batch_size = min(256, buf.buffer_size)
        indices = np.random.randint(0, buf.buffer_size, size=batch_size)

        obs = torch.tensor(buf.observations[indices], dtype=torch.float32, device=device)
        actions = torch.tensor(buf.actions[indices], dtype=torch.float32, device=device)

        # For next obs, use index + 1 (when available)
        next_indices = np.minimum(indices + 1, buf.buffer_size - 1)
        next_obs = torch.tensor(
            buf.observations[next_indices], dtype=torch.float32, device=device
        )

        # Get features with stop-gradient (world model trains its own head only)
        with torch.no_grad():
            features = policy.extract_features(obs, policy.features_extractor)

        # World model prediction (only WM head has gradients)
        pred_next_obs = self._world_model(features, actions.squeeze(1))

        # Only predict the obs_dim portion (last obs_dim elements in flattened history)
        target = next_obs[:, -OBS_DIM:]
        loss = F.mse_loss(pred_next_obs, target) * self.wm_coeff

        self._wm_optimizer.zero_grad()
        loss.backward()
        self._wm_optimizer.step()

        if self.verbose > 0:
            print(f"  [WM] loss={loss.item():.4f}")


# ── Curriculum Callback ──────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Adaptive curriculum based on learning progress.

    Logs curriculum statistics and adjusts terrain difficulty levels.
    """

    def __init__(self, curriculum, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum = curriculum
        self._episode_rewards = {}

    def _on_step(self) -> bool:
        # Track episode rewards
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            env_id = i % self.curriculum.n_envs
            if env_id not in self._episode_rewards:
                self._episode_rewards[env_id] = 0.0

            # Accumulate reward
            rewards = self.locals.get("rewards", [])
            if i < len(rewards):
                self._episode_rewards[env_id] += rewards[i]

            # Check for episode end
            dones = self.locals.get("dones", [])
            if i < len(dones) and dones[i]:
                self.curriculum.record_episode(
                    env_id, self._episode_rewards[env_id]
                )
                self._episode_rewards[env_id] = 0.0

        return True

    def _on_rollout_end(self) -> None:
        if self.verbose > 0:
            print(f"  {self.curriculum.summary()}")
