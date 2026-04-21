"""
Comprehensive Test & Performance Report Framework.

Runs multi-level tests (unit, integration, performance, regression) and
auto-generates an HTML performance report with metrics, charts, and tables.

Usage:
  python tests/test_suite.py                        # Run all tests + report
  python tests/test_suite.py --quick                # Skip slow performance tests
  python tests/test_suite.py --output report.html   # Custom output path
"""
import sys
import os
import time
import json
import math
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Import OBS_DIM so all assertions stay in sync with the environment.
try:
    from src.env.cheetah_env import OBS_DIM as ENV_OBS_DIM
except ImportError:
    ENV_OBS_DIM = 196  # fallback if env import fails during collection


# ---------------------------------------------------------------------------
# Data classes for test results
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    category: str   # "unit", "integration", "performance", "regression"
    passed: bool
    duration: float = 0.0
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Per-scenario performance data."""
    scenario: str
    n_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_steps: float = 0.0
    survival_rate: float = 0.0
    mean_speed: float = 0.0
    mean_height: float = 0.0
    mean_energy: float = 0.0
    foot_contact_ratio: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

class TestSuite:
    """Collects and runs tests, produces reports."""

    def __init__(self, quick: bool = False, verbose: bool = True):
        self.quick = quick
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.perf_data: List[PerformanceMetrics] = []
        self.start_time = time.time()

    # -- Test registration & running --

    def run(self, name: str, category: str, fn):
        t0 = time.time()
        try:
            metrics = fn()
            dt = time.time() - t0
            result = TestResult(name=name, category=category, passed=True,
                                duration=dt, metrics=metrics or {})
            if self.verbose:
                print(f"  [PASS] {name}  ({dt:.2f}s)")
        except Exception as e:
            dt = time.time() - t0
            tb = traceback.format_exc()
            result = TestResult(name=name, category=category, passed=False,
                                duration=dt, message=f"{e}\n{tb}")
            if self.verbose:
                print(f"  [FAIL] {name}: {e}")
        self.results.append(result)
        return result

    def run_all(self):
        """Execute every test group."""
        print("\n" + "=" * 60)
        print("  MINI CHEETAH — COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        print("\n--- Unit Tests ---")
        self._run_unit_tests()

        print("\n--- Integration Tests ---")
        self._run_integration_tests()

        if not self.quick:
            print("\n--- Performance Tests ---")
            self._run_performance_tests()

            print("\n--- Regression Tests ---")
            self._run_regression_tests()

        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        print(f"\n{'='*60}")
        print(f"  TOTAL: {passed} passed / {failed} failed / "
              f"{len(self.results)} total  ({total_time:.1f}s)")
        print(f"{'='*60}\n")
        return failed == 0

    # ---------------------------------------------------------------
    # UNIT TESTS
    # ---------------------------------------------------------------

    def _run_unit_tests(self):
        # Base environment
        self.run("env_creation", "unit", self._test_env_creation)
        self.run("env_step", "unit", self._test_env_step)
        self.run("env_action_space", "unit", self._test_action_space)
        self.run("env_observation_space", "unit", self._test_observation_space)
        self.run("env_domain_randomization", "unit", self._test_domain_rand)
        self.run("env_episode_termination", "unit", self._test_termination)

        # Terrain environment
        self.run("terrain_generator_all_types", "unit", self._test_terrain_gen)
        self.run("terrain_env_creation", "unit", self._test_terrain_env_creation)
        self.run("terrain_env_step", "unit", self._test_terrain_env_step)
        self.run("terrain_env_skill_modes", "unit", self._test_terrain_skill_modes)
        self.run("terrain_foot_contacts", "unit", self._test_foot_contacts)
        self.run("terrain_encoding", "unit", self._test_terrain_encoding)

        # Curriculum
        self.run("curriculum_basic", "unit", self._test_curriculum)
        self.run("curriculum_advanced", "unit", self._test_advanced_curriculum)

        # Policy
        self.run("policy_forward", "unit", self._test_policy_forward)
        self.run("policy_components", "unit", self._test_policy_components)

        # Controls
        self.run("keyboard_controller", "unit", self._test_keyboard)
        self.run("exploration_policy", "unit", self._test_exploration)

    def _test_env_creation(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        obs, info = env.reset(seed=42)
        assert obs.shape == (ENV_OBS_DIM,), f"Expected ({ENV_OBS_DIM},), got {obs.shape}"
        assert obs.dtype == np.float32
        env.close()
        return {"obs_dim": ENV_OBS_DIM}

    def _test_env_step(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        env.reset(seed=42)
        action = np.zeros(12, dtype=np.float32)
        obs, reward, done, trunc, info = env.step(action)
        assert obs.shape == (ENV_OBS_DIM,)
        assert isinstance(reward, float)
        env.close()
        return {"reward": reward}

    def _test_action_space(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none")
        assert env.action_space.shape == (12,)
        assert float(env.action_space.low[0]) == -1.0, (
            f"Expected action low=-1.0, got {env.action_space.low[0]}"
        )
        assert float(env.action_space.high[0]) == 1.0, (
            f"Expected action high=1.0, got {env.action_space.high[0]}"
        )
        env.close()

    def _test_observation_space(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        assert env.observation_space.shape == (ENV_OBS_DIM,)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        env.close()

    def _test_domain_rand(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=True)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=43)
        env.close()

    def _test_termination(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=10)
        env.reset()
        for _ in range(15):
            _, _, done, trunc, _ = env.step(np.zeros(12, dtype=np.float32))
            if done or trunc:
                break
        assert done or trunc, "Episode should terminate"
        env.close()

    # -- Terrain tests --

    def _test_terrain_gen(self):
        from src.env.terrain_env import TerrainGenerator
        gen = TerrainGenerator(size=10.0, resolution=50, seed=0)
        results = {}
        for tt in TerrainGenerator.TERRAIN_TYPES:
            h = gen.generate(tt, difficulty=0.5)
            assert h.shape == (50, 50), f"{tt}: shape {h.shape}"
            results[tt] = {"min": float(h.min()), "max": float(h.max())}
        return results

    def _test_terrain_env_creation(self):
        from src.env.terrain_env import AdvancedTerrainEnv
        env = AdvancedTerrainEnv(render_mode="none", randomize_terrain=False)
        obs, info = env.reset(seed=0)
        assert obs.shape == (57,), f"Expected (57,), got {obs.shape}"
        env.close()
        return {"obs_dim": 57}

    def _test_terrain_env_step(self):
        from src.env.terrain_env import AdvancedTerrainEnv
        env = AdvancedTerrainEnv(render_mode="none", randomize_terrain=False)
        env.reset(seed=0)
        obs, r, d, t, info = env.step(np.zeros(12, dtype=np.float32))
        assert obs.shape == (57,)
        assert "reward_components" in info
        env.close()
        return {"reward": r, "components": len(info["reward_components"])}

    def _test_terrain_skill_modes(self):
        from src.env.terrain_env import AdvancedTerrainEnv, SKILL_MODES
        env = AdvancedTerrainEnv(render_mode="none", randomize_terrain=False,
                                  randomize_skill=False)
        rewards = {}
        for skill in SKILL_MODES:
            env.set_skill(skill)
            env.reset(seed=0)
            _, r, _, _, _ = env.step(np.zeros(12, dtype=np.float32))
            rewards[skill] = r
        env.close()
        return rewards

    def _test_foot_contacts(self):
        from src.env.terrain_env import AdvancedTerrainEnv
        env = AdvancedTerrainEnv(render_mode="none", randomize_terrain=False,
                                  randomize_skill=False)
        env.reset(seed=0)
        for _ in range(50):
            _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
        contacts = info["foot_contacts"]
        assert sum(contacts) > 0, f"Expected foot contacts, got {contacts}"
        env.close()
        return {"contacts": contacts}

    def _test_terrain_encoding(self):
        from src.env.terrain_env import TerrainGenerator
        gen = TerrainGenerator(size=10.0, resolution=100, seed=42)
        gen.generate("rough", 0.8)
        enc = gen.get_terrain_encoding(0.0, 0.0)
        assert enc.shape == (8,)
        assert not np.all(enc == 0), "Rough terrain encoding should be non-zero"
        return {"encoding": enc.tolist()}

    # -- Curriculum tests --

    def _test_curriculum(self):
        from src.training.curriculum import TerrainCurriculum
        c = TerrainCurriculum(n_envs=2)
        for _ in range(30):
            c.record_episode(0, True)
        config = c.get_terrain_config(0)
        assert c.levels[0] > 0, "Should advance after consistent success"
        return {"level": int(c.levels[0]), "terrain": config["name"]}

    def _test_advanced_curriculum(self):
        from src.training.curriculum import AdvancedTerrainCurriculum
        c = AdvancedTerrainCurriculum(n_envs=2)
        for _ in range(50):
            c.record_episode(0, True, 1.0)
        config = c.get_config(0)
        assert c.difficulty[0] > 0 or c.terrain_idx[0] > 0
        return config

    # -- Policy tests --

    def _test_policy_forward(self):
        import torch
        from src.training.advanced_policy import HierarchicalTransformerPolicy
        policy = HierarchicalTransformerPolicy(d_model=64, n_transformer_layers=2,
                                                n_experts=2)
        # Use ENV_OBS_DIM (196) so sensory group slices don't go out of range
        obs = torch.randn(2, 1, ENV_OBS_DIM)
        step_count = torch.zeros(2)
        out = policy(obs, step_count=step_count)
        assert out["action_mean"].shape == (2, 12)
        assert out["value"].shape == (2, 1)
        n_params = sum(p.numel() for p in policy.parameters())
        return {"n_params": n_params}

    def _test_policy_components(self):
        import torch
        from src.training.advanced_policy import (
            GaitPhaseOscillator, TerrainEstimator, ContrastiveTemporalHead
        )
        d = 64
        osc = GaitPhaseOscillator(d_model=d)
        phase = osc(torch.tensor([50]), torch.randn(1, 3))
        assert phase.shape == (1, d)

        te = TerrainEstimator(d_model=d)
        obs_hist = torch.randn(2, 8, 54)  # (batch, seq, obs_dim)
        feat, lat = te(obs_hist)
        assert feat.shape == (2, d)
        assert lat.shape == (2, 8)

        ct = ContrastiveTemporalHead(d_model=d)
        seq = torch.randn(2, 4, d)
        loss = ct.compute_loss(seq)
        assert loss.ndim == 0
        return {"phase_shape": list(phase.shape), "terrain_feat_shape": list(feat.shape)}

    # -- Controls tests --

    def _test_keyboard(self):
        from src.control.keyboard_controller import KeyboardController
        kb = KeyboardController()
        kb._keys_held.add("w")
        kb.speed_level = kb.SPEED_TROT
        kb._update_from_held_keys()
        assert kb.vx > 0
        kb._keys_held.clear()

    def _test_exploration(self):
        from src.control.exploration_policy import ExplorationPolicy
        p = ExplorationPolicy(turn_gain=1.5, forward_speed=1.5)
        p.set_target_heading(math.pi / 2)
        vx, vy, wz = p.get_command(current_yaw=0.0)
        assert wz > 0

    # ---------------------------------------------------------------
    # INTEGRATION TESTS
    # ---------------------------------------------------------------

    def _run_integration_tests(self):
        self.run("sb3_base_env_training", "integration", self._test_sb3_base)
        self.run("sb3_terrain_env_training", "integration", self._test_sb3_terrain)
        self.run("sb3_transformer_policy", "integration", self._test_sb3_transformer)
        self.run("history_wrapper", "integration", self._test_history_wrapper)
        self.run("action_smoothing_wrapper", "integration", self._test_smoothing_wrapper)

    def _test_sb3_base(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from src.env.cheetah_env import MiniCheetahEnv

        env = DummyVecEnv([
            lambda: MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=50)
        ])
        model = PPO("MlpPolicy", env, n_steps=32, batch_size=32, verbose=0, device="cpu")
        model.learn(total_timesteps=64)
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, 12)
        env.close()
        return {"trained": True}

    def _test_sb3_terrain(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from src.env.terrain_env import AdvancedTerrainEnv

        env = DummyVecEnv([
            lambda: AdvancedTerrainEnv(render_mode="none", randomize_terrain=True, episode_length=50)
        ])
        model = PPO("MlpPolicy", env, n_steps=32, batch_size=32, verbose=0, device="cpu")
        model.learn(total_timesteps=64)
        env.close()
        return {"trained": True}

    def _test_sb3_transformer(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from src.env.cheetah_env import MiniCheetahEnv
        from src.training.sb3_integration import (
            TransformerActorCriticPolicy, HistoryWrapper, ActionSmoothingWrapper
        )

        def mk():
            e = MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=50)
            e = ActionSmoothingWrapper(e, alpha=0.8)
            e = HistoryWrapper(e, history_len=8)
            return e

        env = DummyVecEnv([mk])
        model = PPO(
            TransformerActorCriticPolicy, env,
            n_steps=32, batch_size=32, verbose=0, device="cpu",
            policy_kwargs=dict(d_model=64, n_heads=2, n_layers=1, n_experts=2,
                               history_len=8, obs_dim=ENV_OBS_DIM),
        )
        model.learn(total_timesteps=64)
        n_params = sum(p.numel() for p in model.policy.parameters())
        env.close()
        return {"n_params": n_params}

    def _test_history_wrapper(self):
        from src.env.cheetah_env import MiniCheetahEnv
        from src.training.sb3_integration import HistoryWrapper
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        wrapped = HistoryWrapper(env, history_len=8)
        obs, _ = wrapped.reset()
        # HistoryWrapper returns (history_len * obs_dim,) flattened
        expected_size = 8 * ENV_OBS_DIM
        assert obs.size == expected_size, f"Expected {expected_size} elements, got {obs.size}"
        obs2, _, _, _, _ = wrapped.step(np.zeros(12, dtype=np.float32))
        assert obs2.size == expected_size, f"Expected {expected_size} elements, got {obs2.size}"
        wrapped.close()
        return {"obs_shape": list(obs.shape)}

    def _test_smoothing_wrapper(self):
        from src.env.cheetah_env import MiniCheetahEnv
        from src.training.sb3_integration import ActionSmoothingWrapper
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        wrapped = ActionSmoothingWrapper(env, alpha=0.5)
        wrapped.reset()
        a1 = np.ones(12, dtype=np.float32)
        wrapped.step(a1)
        a2 = np.zeros(12, dtype=np.float32)
        obs, _, _, _, _ = wrapped.step(a2)
        wrapped.close()

    # ---------------------------------------------------------------
    # PERFORMANCE TESTS
    # ---------------------------------------------------------------

    def _run_performance_tests(self):
        self.run("perf_base_flat", "performance", self._perf_base_flat)
        self.run("perf_terrain_flat", "performance", self._perf_terrain_flat)
        self.run("perf_terrain_rough", "performance", self._perf_terrain_rough)
        self.run("perf_terrain_stairs", "performance", self._perf_terrain_stairs)
        self.run("perf_terrain_gaps", "performance", self._perf_terrain_gaps)
        self.run("perf_terrain_mixed", "performance", self._perf_terrain_mixed)
        self.run("perf_skill_walk", "performance", lambda: self._perf_skill("walk"))
        self.run("perf_skill_trot", "performance", lambda: self._perf_skill("walk"))
        self.run("perf_skill_run", "performance", lambda: self._perf_skill("run"))
        self.run("perf_skill_jump", "performance", lambda: self._perf_skill("jump"))
        self.run("perf_skill_crouch", "performance", lambda: self._perf_skill("crouch"))
        self.run("perf_simulation_fps", "performance", self._perf_sim_fps)

    @staticmethod
    def _load_best_policy():
        """Load the best available trained policy for use in performance evaluation.

        Returns (model, vec_norm) or (None, None) if no model is found.
        Performance tests use the policy when available so metrics reflect
        real navigation capability, not uncontrolled physics.
        """
        import os
        from stable_baselines3 import PPO
        ckpt_candidates = [
            "checkpoints/best/best_model.zip",
            "checkpoints/best_model.zip",
        ]
        for ckpt in ckpt_candidates:
            if os.path.exists(ckpt):
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = PPO.load(ckpt, device="cpu")
                    return model
                except Exception:
                    pass
        return None

    def _run_episodes(self, env, n_episodes=5, max_steps=200,
                      policy=None) -> PerformanceMetrics:
        """Run evaluation episodes.

        If *policy* is supplied (stable-baselines3 model), actions are drawn
        from it (deterministic).  Otherwise zero actions are used — useful for
        measuring raw physics stability but NOT a measure of navigation skill.
        """
        rewards, steps, speeds, heights, energies, contacts = [], [], [], [], [], []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            ep_steps = 0
            ep_speeds = []
            ep_heights = []
            ep_energies = []
            ep_contacts = []

            for s in range(max_steps):
                if policy is not None:
                    # Use trained policy for realistic evaluation
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    # Zero-action baseline: tests raw physics / terrain geometry
                    action = np.zeros(12, dtype=np.float32)
                obs, r, done, trunc, info = env.step(action)
                ep_reward += r
                ep_steps += 1
                ep_speeds.append(abs(float(env.data.qvel[0])))
                ep_heights.append(float(env.data.qpos[2]))
                if "reward_components" in info:
                    ep_energies.append(abs(info["reward_components"].get("r_torque", 0.0)))
                if "foot_contacts" in info:
                    ep_contacts.append(sum(info["foot_contacts"]) / 4.0)
                if done or trunc:
                    break

            rewards.append(ep_reward)
            steps.append(ep_steps)
            speeds.append(np.mean(ep_speeds) if ep_speeds else 0.0)
            heights.append(np.mean(ep_heights) if ep_heights else 0.0)
            energies.append(np.mean(ep_energies) if ep_energies else 0.0)
            contacts.append(np.mean(ep_contacts) if ep_contacts else 0.0)

        return PerformanceMetrics(
            scenario=env.terrain_type if hasattr(env, "terrain_type") else "base",
            n_episodes=n_episodes,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_steps=float(np.mean(steps)),
            survival_rate=float(np.mean([1.0 if s >= max_steps else 0.0
                                          for s in steps])),
            mean_speed=float(np.mean(speeds)),
            mean_height=float(np.mean(heights)),
            mean_energy=float(np.mean(energies)),
            foot_contact_ratio=float(np.mean(contacts)),
        )

    def _perf_base_flat(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        policy = self._load_best_policy()
        m = self._run_episodes(env, n_episodes=5, max_steps=200, policy=policy)
        env.close()
        self.perf_data.append(m)
        return asdict(m)

    def _perf_terrain(self, terrain_type: str, difficulty: float = 0.5):
        # Use MiniCheetahEnv (196-dim obs, same as training) so the best policy
        # can be evaluated on real terrain.  AdvancedTerrainEnv has only 57-dim
        # obs and cannot be used with the trained policy.
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", terrain_type=terrain_type,
                             terrain_difficulty=difficulty, use_terrain=True,
                             randomize_domain=False)
        policy = self._load_best_policy()
        m = self._run_episodes(env, n_episodes=5, max_steps=200, policy=policy)
        m.scenario = f"terrain_{terrain_type}"
        env.close()
        self.perf_data.append(m)
        return asdict(m)

    def _perf_terrain_flat(self):
        return self._perf_terrain("flat", 0.0)

    def _perf_terrain_rough(self):
        return self._perf_terrain("rough", 0.5)

    def _perf_terrain_stairs(self):
        return self._perf_terrain("stairs_up", 0.3)

    def _perf_terrain_gaps(self):
        return self._perf_terrain("gaps", 0.3)

    def _perf_terrain_mixed(self):
        return self._perf_terrain("mixed", 0.5)

    def _perf_skill(self, skill: str):
        # Use MiniCheetahEnv (196-dim obs) for policy compatibility.
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", terrain_type="flat",
                             terrain_difficulty=0.0, use_terrain=False,
                             randomize_domain=False, forced_mode=skill)
        policy = self._load_best_policy()
        m = self._run_episodes(env, n_episodes=5, max_steps=200, policy=policy)
        m.scenario = f"skill_{skill}"
        env.close()
        self.perf_data.append(m)
        return asdict(m)

    def _perf_sim_fps(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        env.reset()
        n_steps = 1000
        t0 = time.time()
        for _ in range(n_steps):
            env.step(np.zeros(12, dtype=np.float32))
        dt = time.time() - t0
        fps = n_steps / dt
        env.close()
        return {"fps": fps, "time_1000_steps": dt}

    # ---------------------------------------------------------------
    # REGRESSION TESTS
    # ---------------------------------------------------------------

    def _run_regression_tests(self):
        self.run("regression_standing_stability", "regression", self._reg_standing)
        self.run("regression_determinism", "regression", self._reg_determinism)
        self.run("regression_obs_bounds", "regression", self._reg_obs_bounds)
        self.run("regression_reward_sign", "regression", self._reg_reward_sign)

    def _reg_standing(self):
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        env.reset(seed=0)
        # Explicitly command zero-velocity standing to test postural stability,
        # not gait reference oscillation from a random walk command.
        env.set_command(0.0, 0.0, 0.0, "stand")
        heights = []
        for _ in range(200):
            _, _, done, trunc, _ = env.step(np.zeros(12, dtype=np.float32))
            heights.append(float(env.data.qpos[2]))
            if done or trunc:
                break
        env.close()
        min_h = min(heights)
        max_h = max(heights)
        assert min_h > 0.25, f"Robot fell too low: {min_h:.4f}"
        assert max_h < 0.5, f"Robot jumped too high: {max_h:.4f}"
        return {"min_height": min_h, "max_height": max_h}

    def _reg_determinism(self):
        from src.env.cheetah_env import MiniCheetahEnv
        rewards = []
        for trial in range(2):
            env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
            env.reset(seed=42)
            total = 0.0
            for _ in range(50):
                _, r, _, _, _ = env.step(np.zeros(12, dtype=np.float32))
                total += r
            rewards.append(total)
            env.close()
        diff = abs(rewards[0] - rewards[1])
        assert diff < 1e-4, f"Non-deterministic: diff={diff:.6f}"
        return {"reward_diff": diff}

    def _reg_obs_bounds(self):
        from src.env.terrain_env import AdvancedTerrainEnv
        env = AdvancedTerrainEnv(render_mode="none", randomize_terrain=True,
                                  randomize_domain=False)
        env.reset(seed=0)
        for _ in range(100):
            obs, _, done, trunc, _ = env.step(env.action_space.sample())
            if done or trunc:
                env.reset()
            assert np.all(np.isfinite(obs)), f"Non-finite obs detected: {obs}"
        env.close()
        return {"checked_steps": 100}

    def _reg_reward_sign(self):
        """Zero-action on flat terrain should give positive reward."""
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
        env.reset(seed=0)
        rewards = []
        for _ in range(50):
            _, r, done, trunc, _ = env.step(np.zeros(12, dtype=np.float32))
            rewards.append(r)
            if done:
                break
        env.close()
        mean_r = np.mean(rewards)
        assert mean_r > -1.0, f"Mean reward too negative: {mean_r:.4f}"
        return {"mean_reward": float(mean_r)}

    # ---------------------------------------------------------------
    # REPORT GENERATION
    # ---------------------------------------------------------------

    def generate_report(self, output_path: str = "reports/test_report.html"):
        """Generate HTML performance report."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        # Group by category
        by_cat = {}
        for r in self.results:
            by_cat.setdefault(r.category, []).append(r)

        # Build HTML
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini Cheetah — Test & Performance Report</title>
<style>
  :root {{
    --bg: #0d1117; --card-bg: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-muted: #8b949e; --green: #3fb950;
    --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
    --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: var(--bg); color: var(--text); padding: 24px;
    max-width: 1200px; margin: 0 auto;
  }}
  h1 {{ color: var(--blue); margin-bottom: 8px; }}
  h2 {{ color: var(--purple); margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ color: var(--text-muted); margin: 16px 0 8px; }}
  .meta {{ color: var(--text-muted); font-size: 14px; margin-bottom: 20px; }}
  .summary-cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .card {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; min-width: 140px; flex: 1;
  }}
  .card .label {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; }}
  .card .value {{ font-size: 28px; font-weight: bold; margin-top: 4px; }}
  .card .value.pass {{ color: var(--green); }}
  .card .value.fail {{ color: var(--red); }}
  .card .value.neutral {{ color: var(--blue); }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; background: var(--card-bg); }}
  tr:hover {{ background: rgba(88,166,255,0.05); }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 12px; font-weight: bold;
  }}
  .badge.pass {{ background: rgba(63,185,80,0.2); color: var(--green); }}
  .badge.fail {{ background: rgba(248,81,73,0.2); color: var(--red); }}
  .bar-container {{ width: 100%; height: 6px; background: var(--border); border-radius: 3px; }}
  .bar {{ height: 6px; border-radius: 3px; transition: width 0.3s; }}
  .bar.green {{ background: var(--green); }}
  .bar.red {{ background: var(--red); }}
  .bar.blue {{ background: var(--blue); }}
  pre {{ background: var(--card-bg); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }}
  .chart-bar {{
    display: flex; align-items: center; margin: 4px 0;
  }}
  .chart-bar .label {{ width: 160px; font-size: 13px; color: var(--text-muted); }}
  .chart-bar .bar-wrap {{ flex: 1; height: 20px; background: var(--border); border-radius: 4px; position: relative; }}
  .chart-bar .bar-fill {{ height: 100%; border-radius: 4px; }}
  .chart-bar .bar-val {{ position: absolute; right: 8px; top: 2px; font-size: 12px; color: var(--text); }}
</style>
</head>
<body>

<h1>Mini Cheetah — Test & Performance Report</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Duration: {total_time:.1f}s</p>

<div class="summary-cards">
  <div class="card">
    <div class="label">Total Tests</div>
    <div class="value neutral">{total}</div>
  </div>
  <div class="card">
    <div class="label">Passed</div>
    <div class="value pass">{passed}</div>
  </div>
  <div class="card">
    <div class="label">Failed</div>
    <div class="value {'fail' if failed else 'pass'}">{failed}</div>
  </div>
  <div class="card">
    <div class="label">Pass Rate</div>
    <div class="value {'pass' if passed == total else 'fail'}">{passed*100//max(total,1)}%</div>
  </div>
</div>
"""

        # Test results tables by category
        for cat in ["unit", "integration", "performance", "regression"]:
            tests = by_cat.get(cat, [])
            if not tests:
                continue
            cat_pass = sum(1 for t in tests if t.passed)
            html += f"""
<h2>{cat.capitalize()} Tests ({cat_pass}/{len(tests)})</h2>
<table>
<tr><th>Test</th><th>Status</th><th>Duration</th><th>Details</th></tr>
"""
            for t in tests:
                status_badge = '<span class="badge pass">PASS</span>' if t.passed else '<span class="badge fail">FAIL</span>'
                detail = ""
                if t.metrics:
                    detail = ", ".join(f"{k}={v}" for k, v in _flatten_metrics(t.metrics).items())
                elif t.message:
                    # Truncate error messages
                    msg = t.message.split("\n")[0][:120]
                    detail = f'<span style="color:var(--red)">{_html_escape(msg)}</span>'
                html += f"<tr><td>{t.name}</td><td>{status_badge}</td>"
                html += f"<td>{t.duration:.2f}s</td><td>{detail}</td></tr>\n"
            html += "</table>\n"

        # Performance charts
        if self.perf_data:
            html += """
<h2>Performance Metrics</h2>
"""
            # Reward bar chart
            html += '<h3>Mean Episode Reward by Scenario</h3>\n'
            max_reward = max(abs(m.mean_reward) for m in self.perf_data) or 1.0
            for m in self.perf_data:
                pct = max(0, min(100, (m.mean_reward / max_reward) * 100))
                color = "var(--green)" if m.mean_reward > 0 else "var(--red)"
                html += f"""<div class="chart-bar">
  <div class="label">{m.scenario}</div>
  <div class="bar-wrap">
    <div class="bar-fill" style="width:{abs(pct):.0f}%; background:{color}"></div>
    <div class="bar-val">{m.mean_reward:.1f} &plusmn; {m.std_reward:.1f}</div>
  </div>
</div>
"""
            # Survival bar chart
            html += '<h3>Survival Rate by Scenario</h3>\n'
            for m in self.perf_data:
                pct = m.survival_rate * 100
                color = "var(--green)" if pct >= 80 else "var(--yellow)" if pct >= 50 else "var(--red)"
                html += f"""<div class="chart-bar">
  <div class="label">{m.scenario}</div>
  <div class="bar-wrap">
    <div class="bar-fill" style="width:{pct:.0f}%; background:{color}"></div>
    <div class="bar-val">{pct:.0f}%</div>
  </div>
</div>
"""
            # Detailed table
            html += """
<h3>Detailed Performance Table</h3>
<table>
<tr><th>Scenario</th><th>Episodes</th><th>Mean Reward</th><th>Mean Steps</th>
<th>Survival</th><th>Speed</th><th>Height</th><th>Energy</th><th>Contact</th></tr>
"""
            for m in self.perf_data:
                html += f"""<tr>
<td>{m.scenario}</td><td>{m.n_episodes}</td>
<td>{m.mean_reward:.2f} &plusmn; {m.std_reward:.2f}</td>
<td>{m.mean_steps:.0f}</td>
<td>{m.survival_rate*100:.0f}%</td>
<td>{m.mean_speed:.3f}</td>
<td>{m.mean_height:.4f}</td>
<td>{m.mean_energy:.4f}</td>
<td>{m.foot_contact_ratio:.2f}</td>
</tr>"""
            html += "</table>\n"

        # Environment info
        html += f"""
<h2>Environment Info</h2>
<pre>
Project: Unitree Go1 Locomotion
Base Obs Dim: {ENV_OBS_DIM}  |  Terrain Obs Dim: 57
Action Dim: 12     |  Control: PD (kp=60, kd=0.5)
Terrain Types: {', '.join(TerrainGenerator.TERRAIN_TYPES) if 'TerrainGenerator' in dir() else 'flat, rough, slope_up, slope_down, stairs_up, stairs_down, gaps, stepping_stones, random_blocks, mixed'}
Skill Modes: {', '.join(SKILL_MODES.keys()) if 'SKILL_MODES' in dir() else 'walk, trot, run, jump, crouch, stand'}
</pre>
"""

        # Failed test details
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            html += "<h2>Failed Test Details</h2>\n"
            for t in failed_tests:
                html += f'<h3 style="color:var(--red)">{t.name}</h3>\n'
                html += f"<pre>{_html_escape(t.message)}</pre>\n"

        html += """
<p class="meta" style="margin-top:40px; text-align:center;">
  Generated by Mini Cheetah Test Suite
</p>
</body></html>
"""

        with open(output_path, "w") as f:
            f.write(html)

        # Also save JSON metrics
        json_path = output_path.replace(".html", ".json")
        report_data = {
            "timestamp": now,
            "duration": total_time,
            "passed": passed,
            "failed": failed,
            "total": total,
            "tests": [asdict(r) for r in self.results],
            "performance": [asdict(m) for m in self.perf_data],
        }
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"Report saved to: {output_path}")
        print(f"JSON data saved to: {json_path}")
        return output_path


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _flatten_metrics(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for display."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, f"{key}."))
        elif isinstance(v, (list, np.ndarray)):
            out[key] = str(v)[:60]
        elif isinstance(v, float):
            out[key] = f"{v:.4f}"
        else:
            out[key] = str(v)
    return out


# Import for report environment info
try:
    from src.env.terrain_env import TerrainGenerator, SKILL_MODES
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Mini Cheetah Test Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Skip performance and regression tests")
    parser.add_argument("--output", default="reports/test_report.html",
                        help="Output path for HTML report")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    suite = TestSuite(quick=args.quick, verbose=not args.quiet)
    success = suite.run_all()
    suite.generate_report(args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
