"""CMA-ES baseline integrated inside the walker2d package."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from cmaes import CMA

from ..config import CMAESConfig
from ..logging.metrics import plot_training_curves
from ..policies.mlp import two_layer_action, two_layer_parameter_count
from ..utils import debug_log, ensure_dir, seed_all
from ..utils.video import record_policy_rollout, write_rgb_video


def _eval_single_candidate(args: Tuple[np.ndarray, Dict, int, int, int, Optional[np.ndarray]]) -> float:
    """Evaluate a single CMA-ES sample in its own environment."""
    theta, cfg_dict, obs_dim, act_dim, hidden, noise_vec = args
    cfg = CMAESConfig(**cfg_dict)
    env = gym.make(
        cfg.env_id,
        exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
        forward_reward_weight=cfg.forward_reward_weight,
    )
    obs, _ = env.reset(seed=cfg.seed)
    total_reward = 0.0
    steps = 0

    while steps < cfg.max_episode_steps:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = two_layer_action(theta, noisy_obs, obs_dim, hidden, act_dim).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return total_reward


class CMAESTrainer:
    """Train the two-hidden-layer MLP policy with plain CMA-ES."""

    def __init__(self, cfg: CMAESConfig) -> None:
        self.cfg = cfg
        ensure_dir(cfg.out_dir)
        seed_all(cfg.seed)

        probe = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            forward_reward_weight=cfg.forward_reward_weight,
        )
        self.obs_dim = int(np.prod(probe.observation_space.shape))
        self.act_dim = int(np.prod(probe.action_space.shape))
        probe.close()

        self.hidden_size = cfg.hidden_size
        self.total_params = two_layer_parameter_count(self.obs_dim, self.hidden_size, self.act_dim)
        self.optimizer = CMA(
            mean=np.zeros(self.total_params, dtype=np.float64),
            sigma=cfg.sigma0,
            population_size=cfg.pop_size,
            seed=cfg.seed,
        )

        worker_count = cfg.workers if cfg.workers > 0 else (os.cpu_count() or 1)
        debug_log("Starting CMA-ES worker pool", workers=worker_count)
        self.pool = mp.Pool(processes=worker_count)
        print(f"[CMA-ES] active workers: {worker_count}")

        self.history: Dict[str, List[float]] = {"gen_max": [], "gen_avg": []}
        self.best_theta: Optional[np.ndarray] = None
        self.best_reward: float = float("-inf")

    def close(self) -> None:
        """Tear down the multiprocessing pool."""
        debug_log("Closing CMA-ES worker pool")
        self.pool.close()
        self.pool.join()

    def _phase_noise(self, phase_idx: int) -> np.ndarray:
        noise_vec = np.random.normal(0.0, self.cfg.noise_std, size=(self.obs_dim,)).astype(np.float32)
        np.save(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), noise_vec)
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w", encoding="utf-8") as fh:
            json.dump({"phase_noise_norm": float(np.linalg.norm(noise_vec))}, fh, indent=2)
        debug_log("CMA-ES phase noise generated", phase=phase_idx, noise_vector=noise_vec.tolist())
        return noise_vec

    def _save_checkpoint(self, phase_idx: int, gen_global: int) -> None:
        ckpt_dir = os.path.join(self.cfg.out_dir, f"checkpoint_phase{phase_idx:02d}_iter{gen_global:04d}")
        ensure_dir(ckpt_dir)
        state = {
            "config": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_size": self.hidden_size,
            "total_params": self.total_params,
            "cma_mean": self.optimizer._mean.tolist(),
            "cma_sigma": float(self.optimizer._sigma),
            "pop_size": self.optimizer.population_size,
            "seed": self.cfg.seed,
            "history": self.history,
            "best_reward": float(self.best_reward),
        }
        with open(os.path.join(ckpt_dir, "state.json"), "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        if self.best_theta is not None:
            np.save(os.path.join(ckpt_dir, self.cfg.best_params_filename), self.best_theta)

        plot_training_curves(
            self.history,
            os.path.join(ckpt_dir, self.cfg.reward_curve_filename),
            series=[("gen_max", "Max reward"), ("gen_avg", "Avg reward")],
            x_label="Generation",
            left_label="Reward",
            title=f"CMA-ES Walker2d (seed={self.cfg.seed})",
            figsize=(7, 4),
        )

        if self.cfg.allow_save_best_model_video and self.best_theta is not None:
            noise_path = os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy")
            noise_vec = np.load(noise_path) if os.path.exists(noise_path) else None
            try:
                frames = record_policy_rollout(
                    env_id=self.cfg.env_id,
                    seed=self.cfg.seed,
                    policy_fn=lambda obs: two_layer_action(
                        self.best_theta, obs, self.obs_dim, self.hidden_size, self.act_dim
                    ).astype(np.float32),
                    max_episode_steps=self.cfg.max_episode_steps,
                    video_seconds=self.cfg.video_seconds,
                    render_fps=self.cfg.render_fps,
                    noise_vec=noise_vec,
                    env_kwargs={
                        "exclude_current_positions_from_observation": self.cfg.exclude_current_positions_from_observation,
                        "forward_reward_weight": self.cfg.forward_reward_weight,
                    },
                )
                write_rgb_video(frames, os.path.join(ckpt_dir, self.cfg.best_video_filename), self.cfg.render_fps)
            except Exception:
                debug_log("Video export failed for CMA-ES checkpoint", phase=phase_idx)

    def train(self) -> Tuple[Optional[np.ndarray], int, Dict[str, List[float]]]:
        total_gens = self.cfg.iterations_per_phase * self.cfg.phases
        gen_global = 0
        cfg_dict = asdict(self.cfg)

        for phase_idx in range(1, self.cfg.phases + 1):
            noise_vec = self._phase_noise(phase_idx)
            print(
                f"[Phase {phase_idx}/{self.cfg.phases}] noise vector "
                f"{np.array2string(noise_vec, precision=3, separator=', ')}"
            )

            for gen in range(1, self.cfg.iterations_per_phase + 1):
                gen_global += 1
                thetas = [self.optimizer.ask() for _ in range(self.optimizer.population_size)]
                tasks = [(theta, cfg_dict, self.obs_dim, self.act_dim, self.hidden_size, noise_vec) for theta in thetas]
                rewards = self.pool.map(_eval_single_candidate, tasks)

                solutions = []
                for theta, reward in zip(thetas, rewards):
                    solutions.append((theta, -reward))
                    if reward > self.best_reward or self.best_theta is None:
                        self.best_reward = reward
                        self.best_theta = theta.copy()

                self.optimizer.tell(solutions)

                gen_max = float(np.max(rewards))
                gen_avg = float(np.mean(rewards))
                self.history["gen_max"].append(gen_max)
                self.history["gen_avg"].append(gen_avg)

                print(
                    f"[Phase {phase_idx}/{self.cfg.phases}] "
                    f"Gen {gen}/{self.cfg.iterations_per_phase} "
                    f"(global {gen_global}/{total_gens}) | "
                    f"max={gen_max:.2f} avg={gen_avg:.2f} best={self.best_reward:.2f}"
                )

            self._save_checkpoint(phase_idx, gen_global)

        self._save_checkpoint(self.cfg.phases, total_gens)
        self.close()
        return self.best_theta, gen_global, self.history
