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
    """Log per-component reward breakdown every `log_freq` steps.

    Fieldnames are auto-detected from the first batch of data so this
    callback stays in sync even when reward components change.
    """

    def __init__(self, log_dir="logs/training", log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self._csv_path = self.log_dir / "reward_components.csv"
        self._csv_file = None
        self._csv_writer = None
        self._step_buf = deque(maxlen=2048)
        self._fields = None  # determined on first flush

    def _on_training_start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # CSV file and writer are opened lazily once we know the field names

    def _on_step(self):
        # Sample from the first env in the vectorized set
        try:
            envs = self.training_env
            # Access the underlying env to compute reward components
            # We read the info dict which we'll add component data to
            infos = self.locals.get("infos", [])
            if infos and "reward_components" in infos[0]:
                comp = infos[0]["reward_components"]
                self._step_buf.append(comp)
        except Exception:
            pass

        if self.n_calls % self.log_freq == 0 and self._step_buf:
            # Average over buffer
            avg = {}
            keys = self._step_buf[0].keys()
            for k in keys:
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

            row = {"step": self.num_timesteps}
            row.update(avg)
            self._csv_writer.writerow(row)
            self._csv_file.flush()

            if self.verbose >= 1:
                parts = " | ".join(f"{k}={v:+.3f}" for k, v in avg.items())
                print(f"  [RewardLog] step={self.num_timesteps}: {parts}")

            self._step_buf.clear()

        return True

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()
