"""
Curriculum learning: gradually increase terrain difficulty during training.
"""
import numpy as np


class TerrainCurriculum:
    TERRAIN_LEVELS = [
        {"name": "flat",         "height_noise": 0.0,  "step_height": 0.0,  "slope": 0.0},
        {"name": "slight_rough", "height_noise": 0.02, "step_height": 0.0,  "slope": 0.0},
        {"name": "rough",        "height_noise": 0.05, "step_height": 0.0,  "slope": 0.05},
        {"name": "rough_slope",  "height_noise": 0.05, "step_height": 0.0,  "slope": 0.15},
        {"name": "low_steps",    "height_noise": 0.03, "step_height": 0.05, "slope": 0.0},
        {"name": "high_steps",   "height_noise": 0.03, "step_height": 0.15, "slope": 0.0},
        {"name": "stairs",       "height_noise": 0.02, "step_height": 0.20, "slope": 0.0},
        {"name": "rough_stairs", "height_noise": 0.05, "step_height": 0.20, "slope": 0.05},
    ]

    def __init__(self, n_envs: int, success_threshold: float = 0.8):
        self.n_envs = n_envs
        self.success_threshold = success_threshold
        self.levels = np.zeros(n_envs, dtype=int)
        self.success_rates = np.zeros(n_envs)

    def record_episode(self, env_id: int, success: bool):
        alpha = 0.1
        self.success_rates[env_id] = (
            (1 - alpha) * self.success_rates[env_id] + alpha * float(success)
        )
        if self.success_rates[env_id] > self.success_threshold:
            self.levels[env_id] = min(
                self.levels[env_id] + 1, len(self.TERRAIN_LEVELS) - 1
            )

    def get_terrain_config(self, env_id: int) -> dict:
        return self.TERRAIN_LEVELS[self.levels[env_id]]

    def summary(self) -> str:
        avg = self.levels.mean()
        return (f"Curriculum: avg_level={avg:.1f}, max={self.levels.max()}, "
                f"terrain={self.TERRAIN_LEVELS[int(avg)]['name']}")
