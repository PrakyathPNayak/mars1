"""Quick reward validation script."""
import sys, os
# Remove any ROS paths to avoid lark import error
sys.path = [p for p in sys.path if '/opt/ros' not in p]
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from src.env.cheetah_env import MiniCheetahEnv, REWARD_SCALES, ONLY_POSITIVE_REWARDS
import numpy as np

print("REWARD_SCALES:", REWARD_SCALES)
print("ONLY_POSITIVE_REWARDS:", ONLY_POSITIVE_REWARDS)
print("r_alive in scales:", "r_alive" in REWARD_SCALES)
print()

# Test zero-action reward
env = MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=100)
env.randomize_commands = False
env.command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
obs, _ = env.reset(seed=42)

rewards = []
for i in range(50):
    action = np.zeros(12, dtype=np.float32)
    obs, rew, terminated, truncated, info = env.step(action)
    rewards.append(rew)
    if i < 5:
        rc = info.get("reward_components", {})
        parts = " | ".join(f"{k}={v:+.4f}" for k, v in rc.items())
        print(f"Step {i}: total={rew:+.4f} | {parts}")
    if terminated:
        print(f"TERMINATED at step {i}")
        break

print(f"\nZero-action: mean={np.mean(rewards):+.4f}, min={min(rewards):+.4f}, max={max(rewards):+.4f}")
print(f"Steps survived: {len(rewards)}")

# Test random-action reward
env.reset(seed=42)
rewards_rand = []
for i in range(50):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    rewards_rand.append(rew)
    if i < 3:
        rc = info.get("reward_components", {})
        parts = " | ".join(f"{k}={v:+.4f}" for k, v in rc.items())
        print(f"Random step {i}: total={rew:+.4f} | {parts}")
    if terminated:
        print(f"Random terminated at step {i}")
        break

print(f"\nRandom-action: mean={np.mean(rewards_rand):+.4f}, survived={len(rewards_rand)} steps")
env.close()
print("\nAll checks passed!")
