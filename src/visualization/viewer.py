"""
Interactive demo viewer and video recording for Unitree Go1.
"""
import time
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def run_interactive_demo(
    model_path: str = None,
    use_trained: bool = True,
    render_mode: str = "human",
    use_terminal_input: bool = False,
    terrain_type: str = "mixed",
    terrain_difficulty: float = 0.6,
):
    """Launch the interactive keyboard-controlled demo.

    Args:
        use_terminal_input: If True, use terminal-based input (no need to
            click the MuJoCo viewer window). Default False uses pynput.
        terrain_type: Terrain for the demo. Default "mixed" includes slopes,
            stairs, rough patches, and gaps for a challenging experience.
            Options: flat, rough, slope_up, slope_down, stairs_up,
            stairs_down, gaps, stepping_stones, random_blocks, mixed.
        terrain_difficulty: Difficulty level 0.0 (easy) to 1.0 (hard).
            Default 0.6 provides moderate challenge with visible obstacles.
    """
    from src.env.cheetah_env import MiniCheetahEnv
    from src.control.exploration_policy import ExplorationPolicy
    from src.utils.policy_loader import load_policy_for_inference

    if use_terminal_input:
        from src.control.terminal_input import TerminalKeyController, print_terminal_bindings
    else:
        from src.control.keyboard_controller import KeyboardController

    policy = None
    normalize_fn = lambda obs: obs
    if use_trained:
        policy, normalize_fn = load_policy_for_inference(model_path)

    # v24: Interactive demo now uses challenging terrain by default (mixed, difficulty=0.6)
    # instead of flat ground, so the user experiences slopes, stairs, and rough patches.
    env = MiniCheetahEnv(
        render_mode=render_mode,
        randomize_domain=False,
        episode_length=100000,
        terrain_type=terrain_type,
        terrain_difficulty=terrain_difficulty,
        use_terrain=True,
    )

    if use_terminal_input:
        kb = TerminalKeyController()
        kb.start()
        print_terminal_bindings()
    else:
        kb = KeyboardController()
        kb.start()
        kb.print_bindings()

    explore_policy = ExplorationPolicy()
    obs, _ = env.reset()
    total_reward = 0.0
    step = 0
    start_time = time.time()

    print("\n[Demo] Running. Press ESC or Ctrl+C to quit.\n")

    try:
        while True:
            vx, vy, wz, mode = kb.get_command()

            if mode == "explore":
                current_yaw = 0.0  # simplified; would extract from obs
                vx, vy, wz = explore_policy.get_command(current_yaw)

            env.set_command(vx, vy, wz, mode)

            if policy is not None:
                action, _ = policy.predict(normalize_fn(obs), deterministic=True)
            else:
                q_current = obs[:12]
                action = -q_current * 0.1  # simple PD to default stance

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 100 == 0:
                elapsed = time.time() - start_time
                fps = step / max(elapsed, 1e-6)
                print(f"\r[Demo] step={step:6d} | reward={total_reward / step:+.3f} "
                      f"| cmd=({vx:.1f},{vy:.1f},{wz:.1f}) | mode={mode:8s} "
                      f"| fps={fps:.0f}  ", end="", flush=True)

            if done or truncated:
                obs, _ = env.reset()
                print(f"\n[Demo] Episode ended at step {step}. Resetting.")

    except KeyboardInterrupt:
        print("\n[Demo] Stopped by user.")
    finally:
        kb.stop()
        env.close()
        print(f"[Demo] Total steps: {step}, avg reward: {total_reward / max(step, 1):.3f}")


def record_rollout(output_path: str = "logs/rollout.mp4", n_steps: int = 500):
    """Record a rollout to video file."""
    from src.env.cheetah_env import MiniCheetahEnv
    from src.utils.policy_loader import load_policy_for_inference
    import imageio

    env = MiniCheetahEnv(render_mode="rgb_array", randomize_domain=False)
    obs, _ = env.reset()
    frames = []

    policy, normalize_fn = load_policy_for_inference()

    for _ in range(n_steps):
        if policy:
            action, _ = policy.predict(normalize_fn(obs), deterministic=True)
        else:
            action = np.zeros(12)
        obs, _, done, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if done or truncated:
            obs, _ = env.reset()

    env.close()
    if frames:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        imageio.mimsave(output_path, frames, fps=50)
        print(f"Rollout saved: {output_path} ({len(frames)} frames)")
    else:
        print("No frames captured.")


if __name__ == "__main__":
    run_interactive_demo()
