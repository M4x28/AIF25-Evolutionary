from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

try:
    import cv2
except Exception:
    cv2 = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class TrainConfig:
    env_id: str = "Walker2d-v5"
    seed: int = 42
    iterations_per_phase: int = 100
    phases: int = 6
    workers: int = 8
    num_emitters: int = 8 
    emitter_batch: int = 40
    sigma0: float = 0.5
    archive_bins: Tuple[int, int] = (48, 32)
    forward_velocity_range: Tuple[float, float] = (0.0, 10.0)
    torso_height_range: Tuple[float, float] = (0.8, 1.7)
    qd_score_offset: float = -500.0
    noise_std: float = 0.05
    out_dir: str = "runs_mapelites_2_walker2d"
    render_fps: int = 30
    video_seconds: int = 20
    heatmap_vmin: float = -500.0
    heatmap_vmax: float = 1500.0


def simulate_policy(
    model: np.ndarray,
    env_id: str,
    seed: int,
    obs_dim: int,
    act_dim: int,
    noise_vec: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    env = gym.make(env_id)
    weights = model.reshape((act_dim, obs_dim))
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    forward_vel_samples: List[float] = []
    torso_height_samples: List[float] = []
    prev_x = float(env.unwrapped.data.qpos[0])
    dt = float(getattr(env.unwrapped, "dt", 0.02))

    while not done:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        raw_action = weights @ noisy_obs
        action = np.tanh(raw_action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        
        x_now = float(env.unwrapped.data.qpos[0])
        forward_vel_samples.append((x_now - prev_x) / dt)
        prev_x = x_now

        if hasattr(env.unwrapped, "data"):
            torso_height_samples.append(float(env.unwrapped.data.qpos[1]))
        elif info and "z_distance_from_origin" in info:
            torso_height_samples.append(float(info["z_distance_from_origin"]) + 1.0)

    env.close()
    avg_vel = float(np.mean(forward_vel_samples)) if forward_vel_samples else 0.0
    avg_height = float(np.mean(torso_height_samples)) if torso_height_samples else 0.0
    return total_reward, avg_vel, avg_height


def record_video(
    cfg: TrainConfig,
    solution: np.ndarray,
    obs_dim: int,
    act_dim: int,
    noise_vec: Optional[np.ndarray],
    out_path: str,
) -> None:
    if cv2 is None:
        return
    env = gym.make(cfg.env_id, render_mode="rgb_array")
    weights = solution.reshape((act_dim, obs_dim))
    obs, _ = env.reset(seed=cfg.seed)
    frames: List[np.ndarray] = []
    max_frames = cfg.video_seconds * cfg.render_fps
    steps = 0
    max_steps = getattr(env.spec, "max_episode_steps", 1000)
    while steps < max_steps and len(frames) < max_frames:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        raw_action = weights @ noisy_obs
        action = np.tanh(raw_action)
        obs, _, terminated, truncated, _ = env.step(action)
        rgb = env.render()
        if rgb is not None:
            frames.append(rgb)
        if terminated or truncated:
            break
        steps += 1
    env.close()
    if not frames:
        return
    h, w, _ = frames[0].shape
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.render_fps, (w, h))
    for frame in frames:
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()


class WalkerCMAMeTrainer:
    def __init__(self, cfg: TrainConfig):
        np.random.seed(cfg.seed)
        self.cfg = cfg
        ensure_dir(cfg.out_dir)
        env = gym.make(cfg.env_id)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        env.close()

        self.solution_dim = self.obs_dim * self.act_dim
        self.archive = GridArchive(
            solution_dim=self.solution_dim,
            dims=list(cfg.archive_bins),
            ranges=[cfg.forward_velocity_range, cfg.torso_height_range],
            qd_score_offset=cfg.qd_score_offset,
            seed=cfg.seed,
            dtype="f",
        )
        initial = np.zeros(self.solution_dim, dtype=np.float64)
        self.emitters = [
            EvolutionStrategyEmitter(
                archive=self.archive,
                x0=initial,
                sigma0=cfg.sigma0,
                batch_size=cfg.emitter_batch,
                ranker="2imp",
                seed=cfg.seed + i
            )
            for i in range(cfg.num_emitters)
        ]
        self.scheduler = Scheduler(self.archive, self.emitters)
        self.history: Dict[str, List[float]] = {
            "obj_max": [],
            "obj_mean": [],
            "coverage": [],
            "qd_score": [],
            "norm_qd_score": [],
        }
        self.best_solution: Optional[np.ndarray] = None
        self.best_objective: float = -np.inf

    def _phase_noise(self, phase_idx: int) -> np.ndarray:
        noise_vec = np.random.normal(0.0, self.cfg.noise_std, size=(self.obs_dim,)).astype(np.float32)
        np.save(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), noise_vec)
        print(f"[Phase {phase_idx}] Noise vector: {noise_vec}")
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w") as fh:
            json.dump({"noise_std": self.cfg.noise_std}, fh, indent=2)
        return noise_vec

    def _save_checkpoint(
        self,
        phase_idx: int,
        iter_global: int,
        phase_best_solution: Optional[np.ndarray],
        noise_vec: np.ndarray,
    ) -> None:
        ckpt_dir = os.path.join(self.cfg.out_dir, f"checkpoint_phase{phase_idx:02d}_iter{iter_global:04d}")
        ensure_dir(ckpt_dir)
        state = {
            "config": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "history": self.history,
            "best_objective": float(self.best_objective),
        }
        with open(os.path.join(ckpt_dir, "state.json"), "w") as fh:
            json.dump(state, fh, indent=2)

        np.savez(os.path.join(ckpt_dir, "archive.npz"), **self.archive.data())
        with open(os.path.join(ckpt_dir, "scheduler.pkl"), "wb") as fh:
            pickle.dump(self.scheduler, fh)

        if self.best_solution is not None:
            np.save(os.path.join(ckpt_dir, "best_solution.npy"), self.best_solution)
        if phase_best_solution is not None:
            np.save(os.path.join(ckpt_dir, "phase_best_solution.npy"), phase_best_solution)

        def _normalize(series: List[float]) -> List[float]:
            if not series:
                return []
            arr = np.array(series, dtype=np.float32)
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            if arr_max - arr_min < 1e-8:
                return [0.0 for _ in arr]
            return ((arr - arr_min) / (arr_max - arr_min)).tolist()

        fig, ax1 = plt.subplots(figsize=(8, 5))
        line1, = ax1.plot(self.history["obj_max"], label="Max reward", color="tab:blue")
        line2, = ax1.plot(self.history["obj_mean"], label="Mean reward", color="tab:orange")
        ax1.set_xlabel("Iterazione")
        ax1.set_ylabel("Reward")

        ax2 = ax1.twinx()
        line3, = ax2.plot(self.history["coverage"], label="Coverage", color="tab:green")
        norm_qd = _normalize(self.history["qd_score"])
        line4 = None
        if norm_qd:
            line4, = ax2.plot(norm_qd, label="QD score (norm)", color="tab:purple", linestyle="--")
        ax2.set_ylabel("Coverage / Normalized QD")
        ax2.set_ylim(0.0, 1.0)

        lines = [line1, line2, line3] + ([line4] if line4 is not None else [])
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left")
        ax1.set_title("CMA-ME Walker2d")
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "reward_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(
            self.archive,
            vmin=self.cfg.heatmap_vmin,
            vmax=self.cfg.heatmap_vmax,
            cmap="viridis",
            transpose_measures=False,
        )

        plt.gca().invert_yaxis()
        plt.ylabel("Torso height")
        plt.xlabel("Forward velocity")
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "archive_heatmap.png"))
        plt.close()

        if phase_best_solution is not None:
            record_video(
                self.cfg,
                phase_best_solution,
                self.obs_dim,
                self.act_dim,
                noise_vec,
                os.path.join(ckpt_dir, "best_phase.mp4"),
            )

    def _run_phases(self, start_phase: int, iter_global: int) -> int:
        client = Client(n_workers=self.cfg.workers, threads_per_worker=1)
        evaluate = partial(
            simulate_policy,
            env_id=self.cfg.env_id,
            seed=self.cfg.seed,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        for phase_idx in range(start_phase, self.cfg.phases + 1):
            noise_vec = self._phase_noise(phase_idx)
            phase_best_solution = None
            phase_best_obj = -np.inf

            for itr in range(1, self.cfg.iterations_per_phase + 1):
                solutions = self.scheduler.ask()
                futures = client.map(evaluate, solutions, noise_vec=noise_vec)
                results = client.gather(futures)

                objectives = []
                measures = []
                for reward, vel, height in results:
                    objectives.append(reward)
                    measures.append([vel, height])

                self.scheduler.tell(objectives, measures)

                iter_global += 1
                stats = self.archive.stats

                qd_score = float(stats.qd_score) if stats.qd_score is not None else 0.0
                norm_qd_score_attr = getattr(stats, "normalized_qd_score", None)
                norm_qd_score = float(norm_qd_score_attr) if norm_qd_score_attr is not None else 0.0
                obj_max = float(stats.obj_max) if stats.obj_max is not None else 0.0
                obj_mean = float(stats.obj_mean) if stats.obj_mean is not None else 0.0
                coverage = float(stats.coverage) if stats.coverage is not None else 0.0

                self.history["qd_score"].append(qd_score)
                self.history["norm_qd_score"].append(norm_qd_score)
                self.history["obj_max"].append(obj_max)
                self.history["obj_mean"].append(obj_mean)
                self.history["coverage"].append(coverage)

                best_elite = self.archive.best_elite
                if best_elite is not None:
                    best_obj = float(best_elite["objective"])
                    if best_obj > self.best_objective or self.best_solution is None:
                        self.best_objective = best_obj
                        self.best_solution = best_elite["solution"].copy()
                    if best_obj > phase_best_obj or phase_best_solution is None:
                        phase_best_obj = best_obj
                        phase_best_solution = best_elite["solution"].copy()

                if iter_global % 25 == 0 or itr == self.cfg.iterations_per_phase:
                    print(
                        f"[Phase {phase_idx}/{self.cfg.phases}] Iter {itr}/{self.cfg.iterations_per_phase} "
                        f"(global {iter_global}) | max={obj_max:.2f} mean={obj_mean:.2f} "
                        f"cov={coverage:.3f} best={self.best_objective:.2f}"
                    )

            self._save_checkpoint(phase_idx, iter_global, phase_best_solution, noise_vec)

        client.close()
        return iter_global

    def train(self) -> None:
        final_iter = self._run_phases(start_phase=1, iter_global=0)
        if self.best_solution is not None:
            np.save(os.path.join(self.cfg.out_dir, "best_solution.npy"), self.best_solution)


def main() -> None:
    cfg = TrainConfig()
    trainer = WalkerCMAMeTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
