"""
SB3 callback for logging per-component reward breakdown during training.

Logs individual reward terms (velocity tracking, height, torque, etc.) to:
  - TensorBoard (via SB3's logger)
  - CSV file for the live dashboard

Usage:
    from src.training.reward_logger import RewardComponentCallback
    callbacks.append(RewardComponentCallback(log_dir="logs/training"))
"""
import csv
from pathlib import Path
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class RewardComponentCallback(BaseCallback):
    """Log per-component reward breakdown every `log_freq` timesteps.

    Averages across ALL parallel envs at each step, and keeps a rolling
    window of the most recent samples so logged values are smooth but
    up-to-date.  Fieldnames are auto-detected from the first batch so
    this callback stays in sync when reward components change.
    """

    def __init__(self, log_dir="logs/training", log_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self._csv_path = self.log_dir / "reward_components.csv"
        self._csv_file = None
        self._csv_writer = None
        # Rolling window — NOT cleared on each log so values stay smooth
        self._step_buf: deque = deque(maxlen=500)
        self._fields = None
        self._last_log_ts: int = 0

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            # Average reward components across ALL active envs this step
            comp_acc: dict[str, list[float]] = {}
            for info in infos:
                for k, v in info.get("reward_components", {}).items():
                    comp_acc.setdefault(k, []).append(float(v))
            if comp_acc:
                self._step_buf.append(
                    {k: sum(v) / len(v) for k, v in comp_acc.items()}
                )

        if (
            self.num_timesteps - self._last_log_ts >= self.log_freq
            and self._step_buf
        ):
            self._last_log_ts = self.num_timesteps

            # Average over the rolling window
            avg: dict[str, float] = {}
            for k in self._step_buf[0]:
                vals = [d[k] for d in self._step_buf if k in d]
                avg[k] = sum(vals) / len(vals) if vals else 0.0

            # Log to TensorBoard
            for k, v in avg.items():
                self.logger.record(f"reward/{k}", v)

            # Open CSV lazily on first flush so fieldnames match actual components
            if self._csv_writer is None:
                self._fields = ["step"] + list(avg.keys())
                self._csv_file = open(self._csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=self._fields, extrasaction="ignore"
                )
                self._csv_writer.writeheader()

            row: dict = {"step": self.num_timesteps}
            row.update(avg)
            self._csv_writer.writerow(row)
            self._csv_file.flush()

            if self.verbose >= 1:
                total = avg.get("r_total", 0.0)
                # Show top 8 components by absolute magnitude (skip r_total)
                sorted_comps = sorted(
                    ((k, v) for k, v in avg.items() if k != "r_total"),
                    key=lambda x: -abs(x[1]),
                )[:8]
                parts = "  ".join(f"{k}={v:+.4f}" for k, v in sorted_comps)
                print(
                    f"  [RewardLog] ts={self.num_timesteps:,}  "
                    f"r_total={total:+.4f}  |  {parts}"
                )

        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()
