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
from typing import Callable, Optional, Tuple


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
            policy = PPO.load(ckpt, device=device)
            print(f"[OK] Loaded policy: {ckpt}")
            break

    if policy is None:
        print("[WARN] No checkpoint found. Using PD standing controller.")
        return None, identity

    # Load VecNormalize stats for observation normalization
    norm_path = "checkpoints/vec_normalize.pkl"
    if os.path.exists(norm_path):
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
            return policy, normalize_fn
        except Exception as e:
            print(f"[WARN] Could not load VecNormalize stats: {e}")
            return policy, identity
    else:
        print("[INFO] No VecNormalize stats found (checkpoints/vec_normalize.pkl). "
              "Using raw observations.")
        return policy, identity
