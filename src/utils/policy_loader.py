"""
Utility for loading a trained policy with matching VecNormalize stats.

When a model is trained with VecNormalize (observation normalization),
the inference side MUST apply the same normalization — otherwise the
policy receives unnormalized observations and outputs garbage actions.

Usage:
    from src.utils.policy_loader import load_policy_for_inference
    policy, normalize_fn = load_policy_for_inference("checkpoints/best/best_model.zip")
    obs, _ = env.reset()
    action, _ = policy.predict(normalize_fn(obs), deterministic=True)
"""
import os
import numpy as np
from collections import deque
from typing import Callable, Optional, Tuple

# Observation dimension produced by MiniCheetahEnv (single step, no history)
# v21: 49 base + 1 height + 4 foot_contacts + 5 skill + 2 cpg + 6 cmd_decomp = 67
_BASE_OBS_DIM = 67


class HistoryAwarePolicy:
    """Wraps a history-based policy so callers can pass single-step raw observations.

    At inference time the env is used directly (no HistoryWrapper), so we
    replicate what HistoryWrapper does during training: maintain a rolling
    deque of the last `history_len` normalized observations and concatenate
    them before passing to the inner policy.
    """

    def __init__(self, policy, normalize_fn: Callable, history_len: int):
        self._policy = policy
        self._normalize = normalize_fn
        self._history_len = history_len
        self._history: deque = deque(maxlen=history_len)

    def reset_history(self, obs: np.ndarray) -> None:
        """Fill history buffer with the initial observation (call after env.reset())."""
        norm = self._normalize(obs)
        self._history.clear()
        for _ in range(self._history_len):
            self._history.append(norm.copy())

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Accept a single raw observation, maintain history, return action."""
        if len(self._history) == 0:
            self.reset_history(obs)
        norm = self._normalize(obs)
        self._history.append(norm.copy())
        stacked = np.concatenate(list(self._history)).astype(np.float32)
        return self._policy.predict(stacked, deterministic=deterministic)


def load_policy_for_inference(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[object, Callable]:
    """Load a trained PPO policy and its VecNormalize observation stats.

    Returns:
        (policy, normalize_fn) where normalize_fn maps raw obs -> normalized obs.
        If no VecNormalize stats are found, normalize_fn is identity.
        If no checkpoint is found, returns (None, identity).
    """
    identity = lambda obs: obs

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[WARN] stable-baselines3 not installed. Using PD controller.")
        return None, identity

    # Search for checkpoint
    candidates = [checkpoint_path] if checkpoint_path else []
    candidates += [
        "checkpoints/best/best_model.zip",
        "checkpoints/cheetah_final.zip",
    ]

    policy = None
    for ckpt in candidates:
        if ckpt and os.path.exists(ckpt):
            # Override learning_rate with a plain float so that checkpoints
            # saved with a lambda/closure lr_schedule (which can't be
            # reconstructed at load time) don't raise TypeError on _setup_model.
            try:
                policy = PPO.load(ckpt, device=device,
                                  custom_objects={"learning_rate": 3e-4,
                                                  "clip_range": 0.2})
            except Exception as e:
                print(f"[WARN] PPO.load failed ({e}), retrying with lr override")
                policy = PPO.load(ckpt, device=device,
                                  custom_objects={"learning_rate": 3e-4})
            print(f"[OK] Loaded policy: {ckpt}")
            break

    if policy is None:
        print("[WARN] No checkpoint found. Using PD standing controller.")
        return None, identity

    # Detect whether this is a history-based policy (hierarchical Transformer).
    # If the policy's observation space is larger than _BASE_OBS_DIM and evenly
    # divisible, it was trained with HistoryWrapper (history_len × obs_dim).
    policy_obs_dim = policy.observation_space.shape[0]
    history_len = 1
    if policy_obs_dim > _BASE_OBS_DIM and policy_obs_dim % _BASE_OBS_DIM == 0:
        history_len = policy_obs_dim // _BASE_OBS_DIM
        print(f"[INFO] History-based policy detected (obs_dim={policy_obs_dim}, "
              f"history_len={history_len}, base_obs_dim={_BASE_OBS_DIM})")

    # Load VecNormalize stats for observation normalization.
    # Search in the same directory as the checkpoint first, then parent, then default.
    ckpt_dir = os.path.dirname(ckpt) if ckpt else ""
    ckpt_parent = os.path.dirname(ckpt_dir) if ckpt_dir else ""
    norm_candidates = [
        os.path.join(ckpt_dir, "vec_normalize.pkl"),
        os.path.join(ckpt_parent, "vec_normalize.pkl"),
        "checkpoints/vec_normalize.pkl",
    ]
    norm_path = None
    for nc in norm_candidates:
        if nc and os.path.exists(nc):
            norm_path = nc
            break
    if norm_path:
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            import pickle
            with open(norm_path, "rb") as f:
                vec_norm = pickle.load(f)
            obs_rms = vec_norm.obs_rms
            clip_obs = vec_norm.clip_obs
            epsilon = vec_norm.epsilon

            # Handle obs-dim mismatch: stats saved with old dim (e.g. 48) but
            # env now produces larger obs (e.g. 49 after adding target_height).
            # Pad mean with 0 and var with 1 for the extra dimensions so the
            # new features pass through un-normalized until new stats are saved.
            saved_dim = obs_rms.mean.shape[0]

            def normalize_fn(obs: np.ndarray) -> np.ndarray:
                """Apply the saved running mean/var normalization."""
                obs_dim = obs.shape[0]
                if obs_dim != saved_dim:
                    mean = np.zeros(obs_dim, dtype=np.float64)
                    var  = np.ones(obs_dim,  dtype=np.float64)
                    mean[:saved_dim] = obs_rms.mean
                    var[:saved_dim]  = obs_rms.var
                else:
                    mean = obs_rms.mean
                    var  = obs_rms.var
                normalized = (obs - mean) / np.sqrt(var + epsilon)
                return np.clip(normalized, -clip_obs, clip_obs).astype(np.float32)

            print(f"[OK] Loaded VecNormalize stats: {norm_path}")
        except Exception as e:
            print(f"[WARN] Could not load VecNormalize stats: {e}")
            normalize_fn = identity
    else:
        print("[INFO] No VecNormalize stats found (checkpoints/vec_normalize.pkl). "
              "Using raw observations.")
        normalize_fn = identity

    # Wrap history-based policies so callers can pass single-step raw obs.
    # The wrapper maintains the rolling deque and applies normalization internally.
    if history_len > 1:
        wrapped = HistoryAwarePolicy(policy, normalize_fn, history_len)
        print(f"[INFO] Wrapped in HistoryAwarePolicy (history_len={history_len}). "
              "Call policy.reset_history(obs) after env.reset().")
        return wrapped, identity

    return policy, normalize_fn
