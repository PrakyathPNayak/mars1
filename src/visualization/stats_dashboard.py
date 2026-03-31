"""
Matplotlib live dashboard for training progress.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingDashboard:
    def __init__(self, log_dir: str = "logs/training"):
        self.log_dir = Path(log_dir)

    def read_monitor_csv(self):
        for f in self.log_dir.glob("*.monitor.csv"):
            try:
                lines = f.read_text().strip().split("\n")
                if len(lines) < 3:
                    continue
                import csv
                import io
                reader = csv.DictReader(io.StringIO("\n".join(lines[1:])))
                rewards, lengths = [], []
                for row in reader:
                    rewards.append(float(row["r"]))
                    lengths.append(float(row["l"]))
                return np.array(rewards), np.array(lengths)
            except Exception:
                continue
        return np.array([]), np.array([])

    def generate_plot(self, output_path: str = "logs/training_dashboard.png"):
        rewards, lengths = self.read_monitor_csv()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("MIT Mini Cheetah — Training Dashboard", fontsize=14)

        if len(rewards) > 0:
            ax = axes[0, 0]
            window = min(100, len(rewards))
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(rewards, alpha=0.3, color="steelblue", linewidth=0.5)
            if len(smoothed) > 0:
                ax.plot(range(window - 1, len(rewards)), smoothed,
                        color="steelblue", linewidth=2)
            ax.set_title("Episode Reward")
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.plot(lengths, alpha=0.5, color="coral", linewidth=0.8)
            ax.set_title("Episode Length")
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            recent = rewards[-min(500, len(rewards)):]
            ax.hist(recent, bins=30, color="teal", alpha=0.7, edgecolor="white")
            ax.set_title("Reward Distribution (last 500 eps)")
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.text(0.5, 0.5,
                    f"Episodes: {len(rewards)}\n"
                    f"Mean reward (last 100): {np.mean(rewards[-100:]):.2f}\n"
                    f"Max reward: {rewards.max():.2f}\n"
                    f"Mean length: {lengths.mean():.0f}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
            ax.set_title("Summary")
            ax.axis("off")
        else:
            axes[0, 0].text(0.5, 0.5, "No training data yet",
                            transform=axes[0, 0].transAxes, ha="center", fontsize=14)
            for ax in axes.flat:
                ax.axis("off")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        print(f"Dashboard saved: {output_path}")

    def run(self):
        self.generate_plot()


if __name__ == "__main__":
    dash = TrainingDashboard()
    dash.run()
