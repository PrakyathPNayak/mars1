"""
Main entrypoint for Unitree Go1 RL Locomotion.

Usage:
  python3 run.py demo         # Interactive keyboard demo
  python3 run.py train        # Start training
  python3 run.py train --resume checkpoints/cheetah_ppo_500000_steps.zip
  python3 run.py eval         # Evaluate best checkpoint
  python3 run.py dashboard    # Generate training dashboard
  python3 run.py test         # Run all tests
  python3 run.py explore --heading 45
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Unitree Go1 RL Locomotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    demo_p = sub.add_parser("demo", help="Interactive keyboard demo")
    demo_p.add_argument("--no-policy", action="store_true")
    demo_p.add_argument("--record", action="store_true")
    demo_p.add_argument("--terminal-input", action="store_true",
                        help="Read keyboard from terminal instead of pynput")

    train_p = sub.add_parser("train", help="Train PPO policy")
    train_p.add_argument("--steps", type=int, default=5_000_000)
    train_p.add_argument("--envs", type=int, default=8)
    train_p.add_argument("--resume", type=str, default=None)

    eval_p = sub.add_parser("eval", help="Evaluate checkpoint")
    eval_p.add_argument("--checkpoint", default="checkpoints/best/best_model.zip")
    eval_p.add_argument("--episodes", type=int, default=20)
    eval_p.add_argument("--render", action="store_true")

    sub.add_parser("dashboard", help="Generate training dashboard plot")
    sub.add_parser("live", help="Launch live training dashboard (rich)")
    sub.add_parser("test", help="Run tests")

    explore_p = sub.add_parser("explore", help="Exploration demo")
    explore_p.add_argument("--heading", type=float, default=0.0,
                           help="Target heading in degrees")

    args = parser.parse_args()

    if args.command == "demo":
        from src.visualization.viewer import run_interactive_demo, record_rollout
        if args.record:
            record_rollout()
        else:
            run_interactive_demo(
                use_trained=not args.no_policy,
                use_terminal_input=args.terminal_input,
            )

    elif args.command == "train":
        from src.training.train import train
        import types
        a = types.SimpleNamespace(
            total_steps=args.steps,
            n_envs=args.envs,
            resume=args.resume,
        )
        train(a)

    elif args.command == "eval":
        from scripts.evaluate import evaluate
        evaluate(args.checkpoint, args.episodes, args.render)

    elif args.command == "dashboard":
        from src.visualization.stats_dashboard import TrainingDashboard
        dash = TrainingDashboard()
        dash.run()

    elif args.command == "live":
        from src.visualization.live_dashboard import run_dashboard
        run_dashboard(launch_tb=True)

    elif args.command == "test":
        from tests.test_env import run_all_tests
        sys.exit(0 if run_all_tests() else 1)

    elif args.command == "explore":
        import math
        import numpy as np
        from src.env.cheetah_env import MiniCheetahEnv
        from src.control.exploration_policy import ExplorationPolicy

        heading_rad = math.radians(args.heading)
        print(f"Exploration demo: heading={args.heading} degrees")

        env = MiniCheetahEnv(render_mode="human", randomize_domain=False)
        ctrl = ExplorationPolicy()
        ctrl.set_target_heading(heading_rad)
        env.set_exploration_heading(heading_rad)

        obs, _ = env.reset()
        for step in range(2000):
            vx, vy, wz = ctrl.get_command(0.0)
            env.set_command(vx, vy, wz, "explore")
            action = np.zeros(12, dtype=np.float32)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
        env.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
