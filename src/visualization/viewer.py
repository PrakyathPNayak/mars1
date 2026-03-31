"""
Interactive demo viewer and video recording for Mini Cheetah.
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
):
    """Launch the interactive keyboard-controlled demo.

    Args:
        use_terminal_input: If True, use terminal-based input (no need to
            click the MuJoCo viewer window). Default False uses pynput.
    """
    from src.env.cheetah_env import MiniCheetahEnv
    from src.control.exploration_policy import ExplorationPolicy

    if use_terminal_input:
        from src.control.terminal_input import TerminalKeyController, print_terminal_bindings
    else:
        from src.control.keyboard_controller import KeyboardController

    policy = None
    if use_trained:
        try:
            from stable_baselines3 import PPO
            candidates = [
                "checkpoints/best/best_model.zip",
                "checkpoints/cheetah_final.zip",
            ]
            if model_path:
                candidates.insert(0, model_path)
            for ckpt in candidates:
                if os.path.exists(ckpt):
                    policy = PPO.load(ckpt)
                    print(f"[Demo] Loaded policy: {ckpt}")
                    break
            if policy is None:
                print("[Demo] No checkpoint found. Using PD standing controller.")
        except ImportError:
            print("[Demo] SB3 not available. Using PD controller.")

    env = MiniCheetahEnv(
        render_mode=render_mode,
        randomize_domain=False,
        episode_length=100000,
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
                action, _ = policy.predict(obs, deterministic=True)
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
    import imageio

    env = MiniCheetahEnv(render_mode="rgb_array", randomize_domain=False)
    obs, _ = env.reset()
    frames = []

    policy = None
    try:
        from stable_baselines3 import PPO
        for ckpt in ["checkpoints/best/best_model.zip", "checkpoints/cheetah_final.zip"]:
            if os.path.exists(ckpt):
                policy = PPO.load(ckpt)
                break
    except Exception:
        pass

    for _ in range(n_steps):
        if policy:
            action, _ = policy.predict(obs, deterministic=True)
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
