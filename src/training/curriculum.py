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


class AdvancedTerrainCurriculum:
    """Curriculum for AdvancedTerrainEnv: manages terrain type and difficulty
    progression, plus skill mode scheduling.

    Progression:  flat→rough→slope→stairs→gaps→stepping_stones→mixed
    Each terrain starts at difficulty 0 and ramps up to 1.
    """

    TERRAIN_SEQUENCE = [
        "flat", "rough", "slope_up", "stairs_up",
        "gaps", "stepping_stones", "random_blocks", "mixed",
    ]

    SKILL_SEQUENCE = [
        "stand", "walk", "trot", "run", "crouch", "jump",
    ]

    def __init__(
        self,
        n_envs: int = 1,
        success_threshold: float = 0.7,
        difficulty_increment: float = 0.1,
    ):
        self.n_envs = n_envs
        self.success_threshold = success_threshold
        self.difficulty_increment = difficulty_increment

        self.terrain_idx = np.zeros(n_envs, dtype=int)
        self.difficulty = np.zeros(n_envs, dtype=np.float32)
        self.skill_idx = np.zeros(n_envs, dtype=int)
        self.success_ema = np.zeros(n_envs, dtype=np.float32)

    def record_episode(self, env_id: int, success: bool, ep_reward: float = 0.0):
        alpha = 0.1
        self.success_ema[env_id] = (
            (1 - alpha) * self.success_ema[env_id] + alpha * float(success)
        )

        if self.success_ema[env_id] > self.success_threshold:
            # Increase difficulty first, then advance terrain
            if self.difficulty[env_id] < 0.95:
                self.difficulty[env_id] = min(
                    1.0, self.difficulty[env_id] + self.difficulty_increment
                )
            else:
                # Advance terrain type
                self.terrain_idx[env_id] = min(
                    self.terrain_idx[env_id] + 1,
                    len(self.TERRAIN_SEQUENCE) - 1,
                )
                self.difficulty[env_id] = 0.0

            # Advance skill periodically
            if (self.terrain_idx[env_id] > 0 and
                    self.skill_idx[env_id] < len(self.SKILL_SEQUENCE) - 1):
                self.skill_idx[env_id] = min(
                    self.terrain_idx[env_id] // 2,
                    len(self.SKILL_SEQUENCE) - 1,
                )

    def get_config(self, env_id: int) -> dict:
        return {
            "terrain_type": self.TERRAIN_SEQUENCE[self.terrain_idx[env_id]],
            "difficulty": float(self.difficulty[env_id]),
            "skill_mode": self.SKILL_SEQUENCE[self.skill_idx[env_id]],
        }

    def summary(self) -> str:
        avg_t = self.terrain_idx.mean()
        avg_d = self.difficulty.mean()
        avg_s = self.skill_idx.mean()
        t_name = self.TERRAIN_SEQUENCE[int(avg_t)]
        s_name = self.SKILL_SEQUENCE[int(avg_s)]
        return (
            f"TerrainCurriculum: terrain={t_name} (idx={avg_t:.1f}), "
            f"difficulty={avg_d:.2f}, skill={s_name} (idx={avg_s:.1f})"
        )
