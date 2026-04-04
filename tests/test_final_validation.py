"""Final validation test for the advanced transformer architecture."""
import sys, os
sys.path.insert(0, '/home/prakyathpnayak/Documents/programming/MARS/miniproject')
os.chdir('/home/prakyathpnayak/Documents/programming/MARS/miniproject')

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env.cheetah_env import MiniCheetahEnv
from src.training.sb3_integration import (
    TransformerActorCriticPolicy, HistoryWrapper,
    ActionSmoothingWrapper, WorldModelCallback, CurriculumCallback
)
from src.training.advanced_policy import AdaptiveCurriculum

N_ENVS = 4
HISTORY = 16

def mk(rank):
    def _init():
        env = MiniCheetahEnv(render_mode='none', randomize_domain=True, episode_length=200)
        env = ActionSmoothingWrapper(env, alpha=0.8)
        env = HistoryWrapper(env, history_len=HISTORY)
        env.reset(seed=rank)
        return env
    return _init

print(f'Creating {N_ENVS} envs...')
vec_env = DummyVecEnv([mk(i) for i in range(N_ENVS)])
eval_env = DummyVecEnv([mk(999)])

def lr_sched(p):
    prog = 1.0 - p
    if prog < 0.05: return max(prog / 0.05, 1e-6)
    dp = (prog - 0.05) / 0.95
    return 0.5 * (1.0 + np.cos(np.pi * dp))

print('Building model...')
model = PPO(
    policy=TransformerActorCriticPolicy,
    env=vec_env,
    learning_rate=lambda p: 3e-4 * lr_sched(p),
    n_steps=128, batch_size=64, n_epochs=3,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(d_model=128, n_heads=4, n_layers=3, n_experts=4, history_len=HISTORY, obs_dim=54),
    verbose=0, device='cpu',
)

total = sum(p.numel() for p in model.policy.parameters())
print(f'Total params: {total:,}')

curriculum = AdaptiveCurriculum(n_envs=N_ENVS)
callbacks = [
    WorldModelCallback(wm_lr=1e-4, wm_coeff=0.1, update_freq=128, verbose=0),
    CurriculumCallback(curriculum, verbose=0),
]

print('Training 512 steps...')
model.learn(total_timesteps=512, callback=callbacks)
print('[OK] Training completed')

print('Evaluating...')
rewards = []
for ep in range(3):
    obs = eval_env.reset()
    total_rew = 0.0
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = eval_env.step(action)
        total_rew += rew[0]
        if done[0]: break
    rewards.append(total_rew)
print(f'[OK] Eval rewards: {[round(r,1) for r in rewards]}')

# Save/load test
import tempfile
with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
    path = f.name
model.save(path)
loaded = PPO.load(path, env=vec_env, device='cpu')
obs = vec_env.reset()
a1, _ = model.predict(obs, deterministic=True)
a2, _ = loaded.predict(obs, deterministic=True)
assert np.allclose(a1, a2, atol=1e-5), 'Save/load mismatch!'
print('[OK] Save/load roundtrip verified')
os.unlink(path)

vec_env.close()
eval_env.close()
print()
print('FINAL VALIDATION PASSED')
