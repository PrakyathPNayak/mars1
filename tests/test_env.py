"""Unit and integration tests for the Unitree Go1 environment."""
import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_env_creation():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
    obs, info = env.reset()
    assert obs.shape == (61,), f"Expected obs shape (61,), got {obs.shape}"
    assert obs.dtype == np.float32
    env.close()
    print("  [PASS] test_env_creation")


def test_step():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs2, reward, done, truncated, info = env.step(action)
    assert obs2.shape == (61,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    env.close()
    print("  [PASS] test_step")


def test_multiple_steps():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
    obs, _ = env.reset()
    for _ in range(50):
        action = np.zeros(12, dtype=np.float32)
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    # Robot should be standing with zero action
    assert obs is not None
    env.close()
    print("  [PASS] test_multiple_steps")


def test_domain_randomization():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=True)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=43)
    # Different seeds should give slightly different observations (noise)
    env.close()
    print("  [PASS] test_domain_randomization")


def test_keyboard_controller():
    from src.control.keyboard_controller import KeyboardController
    kb = KeyboardController()
    kb._keys_held.add("w")
    kb.speed_level = kb.SPEED_TROT
    kb._update_from_held_keys()
    assert kb.vx > 0, "Forward key should give positive vx"
    kb._keys_held.clear()
    kb.vx = kb.vy = kb.wz = 0.0

    kb._keys_held.add("a")
    kb._update_from_held_keys()
    assert kb.vy > 0, "Left key should give positive vy"
    kb._keys_held.clear()
    kb.vx = kb.vy = kb.wz = 0.0
    print("  [PASS] test_keyboard_controller")


def test_exploration_policy():
    from src.control.exploration_policy import ExplorationPolicy
    policy = ExplorationPolicy(turn_gain=1.5, forward_speed=1.5)
    policy.set_target_heading(math.pi / 2)

    vx, vy, wz = policy.get_command(current_yaw=0.0)
    assert wz > 0, f"Expected positive wz (turn left), got {wz}"

    vx, vy, wz = policy.get_command(current_yaw=math.pi / 2)
    assert abs(wz) < 0.3, f"Expected near-zero wz when aligned, got {wz}"
    assert vx > 0.5, f"Expected forward motion when aligned, got {vx}"

    policy.set_target_waypoint(5.0, 5.0, 0.0, 0.0)
    vx2, vy2, wz2 = policy.get_command(0.0, (0.0, 0.0))
    assert vx2 >= 0
    print("  [PASS] test_exploration_policy")


def test_command_setting():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
    env.reset()
    env.set_command(1.5, 0.0, 0.0, "walk")
    assert env.command[0] == 1.5
    env.set_exploration_heading(math.pi / 4, speed=2.0)
    assert env.command_mode == "walk"
    env.close()
    print("  [PASS] test_command_setting")


def test_episode_termination():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=10)
    obs, _ = env.reset()
    reached_truncation = False
    for _ in range(15):
        obs, _, done, truncated, _ = env.step(np.zeros(12, dtype=np.float32))
        if truncated:
            reached_truncation = True
            break
        if done:
            break
    assert reached_truncation or done, "Episode should terminate"
    env.close()
    print("  [PASS] test_episode_termination")


def test_action_space():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
    assert env.action_space.shape == (12,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0
    env.close()
    print("  [PASS] test_action_space")


def test_observation_space():
    from src.env.cheetah_env import MiniCheetahEnv
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
    assert env.observation_space.shape == (61,)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "obs not in observation space"
    env.close()
    print("  [PASS] test_observation_space")


def run_all_tests():
    tests = [
        test_env_creation,
        test_step,
        test_multiple_steps,
        test_domain_randomization,
        test_keyboard_controller,
        test_exploration_policy,
        test_command_setting,
        test_episode_termination,
        test_action_space,
        test_observation_space,
    ]
    passed = failed = 0
    print("\n=== Unitree Go1 Test Suite ===\n")
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
