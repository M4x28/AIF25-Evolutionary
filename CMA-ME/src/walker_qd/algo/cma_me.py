"""Implementation of the CMA-ME quality-diversity trainer."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
from dask.distributed import Client
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from ..config import TrainConfig
from ..eval.simulator import record_episode_video, simulate_policy
from ..logging import debug_log
from ..logging.metrics import History
from ..persistence.checkpoint import save_checkpoint_bundle
from ..utils.io import ensure_dir, save_json
from ..utils.video import write_rgb_video

@dataclass
class CMAMETrainer:
    """Train a linear policy on Walker2d with CMA-ME and MAP-Elites."""

    cfg: TrainConfig
    obs_dim: int
    act_dim: int

    archive: GridArchive = field(init=False)
    emitters: list = field(init=False)
    scheduler: Scheduler = field(init=False)
    history: History = field(default_factory=History)
    best_solution: Optional[np.ndarray] = None
    best_objective: float = float("-inf")

    def __post_init__(self) -> None:
        ensure_dir(self.cfg.out_dir)
        solution_dim = self.obs_dim * self.act_dim
        debug_log(
            "Initializing CMA-ME trainer",
            solution_dim=solution_dim,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            emitters=self.cfg.num_emitters,
        )
        self.archive = GridArchive(
            solution_dim=solution_dim,
            dims=list(self.cfg.archive_bins),
            ranges=[self.cfg.forward_velocity_range, self.cfg.torso_height_range],
            qd_score_offset=self.cfg.qd_score_offset,
            seed=self.cfg.seed,
            dtype="f",
        )
        initial = np.zeros(solution_dim, dtype=np.float64)
        self.emitters = [
            EvolutionStrategyEmitter(
                archive=self.archive,
                x0=initial,
                sigma0=self.cfg.sigma0,
                batch_size=self.cfg.emitter_batch,
                ranker="2imp",
                seed=self.cfg.seed + i,
            ) for i in range(self.cfg.num_emitters)
        ]
        self.scheduler = Scheduler(self.archive, self.emitters)
        debug_log("Scheduler initialized", archive_bins=self.cfg.archive_bins)

    def _phase_noise(self, phase_idx: int) -> np.ndarray:
        """Generate and persist the noise vector used during the phase."""
        noise_vec = np.random.normal(0.0, self.cfg.noise_std, size=(self.obs_dim,)).astype(np.float32)
        noise_path = os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy")
        json_path = os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json")
        np.save(noise_path, noise_vec)
        save_json(json_path, {"noise_std": self.cfg.noise_std})
        debug_log("Phase noise generated", phase=phase_idx, path=noise_path)
        return noise_vec

    def _checkpoint(
        self,
        phase_idx: int,
        iter_global: int,
        phase_best_solution: Optional[np.ndarray],
        noise_vec: np.ndarray,
    ) -> None:
        """Snapshot training artifacts and optionally render video for the phase best."""
        ckpt_dir = save_checkpoint_bundle(
            out_dir=self.cfg.out_dir,
            phase_idx=phase_idx,
            iter_global=iter_global,
            cfg_dict=self.cfg.__dict__,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            history=self.history,
            archive=self.archive,
            scheduler=self.scheduler,
            best_solution=self.best_solution,
            phase_best_solution=phase_best_solution,
            heatmap_bounds=(self.cfg.heatmap_vmin, self.cfg.heatmap_vmax),
        )
        # Video best phase
        if phase_best_solution is not None:
            frames = record_episode_video(
                env_id=self.cfg.env_id,
                seed=self.cfg.seed,
                solution=phase_best_solution,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                noise_vec=noise_vec,
                max_seconds=self.cfg.video_seconds,
                fps=self.cfg.render_fps,
            )
            write_rgb_video(frames, os.path.join(ckpt_dir, "best_phase.mp4"), self.cfg.render_fps)
            debug_log("Phase video exported", phase=phase_idx, frames=len(frames))
        debug_log("Checkpoint completed", phase=phase_idx, iteration=iter_global, path=ckpt_dir)

    def _update_history_and_bests(self) -> None:
        """Track archive stats while updating the global best solution."""
        stats = self.archive.stats
        qd_score = float(stats.qd_score) if stats.qd_score is not None else 0.0
        obj_max = float(stats.obj_max) if stats.obj_max is not None else 0.0
        obj_mean = float(stats.obj_mean) if stats.obj_mean is not None else 0.0
        coverage = float(stats.coverage) if stats.coverage is not None else 0.0

        self.history.qd_score.append(qd_score)
        # Alcune versioni di ribs hanno normalized_qd_score; se non c'è, lasciamo 0
        norm_qd_score_attr = getattr(stats, "normalized_qd_score", None)
        self.history.norm_qd_score.append(float(norm_qd_score_attr) if norm_qd_score_attr is not None else 0.0)
        self.history.obj_max.append(obj_max)
        self.history.obj_mean.append(obj_mean)
        self.history.coverage.append(coverage)

        best_elite = self.archive.best_elite
        if best_elite is not None:
            best_obj = float(best_elite["objective"])
            if best_obj > self.best_objective or self.best_solution is None:
                self.best_objective = best_obj
                self.best_solution = best_elite["solution"].copy()
        debug_log(
            "Archive stats updated",
            qd_score=qd_score,
            obj_max=obj_max,
            obj_mean=obj_mean,
            coverage=coverage,
        )

    def train(self) -> tuple[Optional[np.ndarray], int, History]:
        """Execute the CMA-ME training loop."""
        debug_log("Starting Dask client", workers=self.cfg.workers)
        client = Client(n_workers=self.cfg.workers, threads_per_worker=1)
        evaluate = partial(
            simulate_policy,
            env_id=self.cfg.env_id,
            seed=self.cfg.seed,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )

        iter_global = 0
        for phase_idx in range(1, self.cfg.phases + 1):
            noise_vec = self._phase_noise(phase_idx)
            phase_best_solution = None
            phase_best_obj = float("-inf")
            debug_log("Phase started", phase=phase_idx, iterations=self.cfg.iterations_per_phase)

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

                self._update_history_and_bests()

                # Aggiorna best di fase
                be = self.archive.best_elite
                if be is not None:
                    best_obj = float(be["objective"])
                    if best_obj > phase_best_obj or phase_best_solution is None:
                        phase_best_obj = best_obj
                        phase_best_solution = be["solution"].copy()

                debug_log(
                    "Iteration completed",
                    phase=phase_idx,
                    iter_in_phase=itr,
                    iter_global=iter_global,
                    best_phase=phase_best_obj,
                    best_global=self.best_objective,
                )

            self._checkpoint(phase_idx, iter_global, phase_best_solution, noise_vec)
            debug_log("Phase completed", phase=phase_idx, iter_global=iter_global)
            if self.history.obj_max:
                print(
                    f"[Phase {phase_idx}/{self.cfg.phases}] completed | "
                    f"iters={iter_global} | max={self.history.obj_max[-1]:.2f} "
                    f"mean={self.history.obj_mean[-1]:.2f} cov={self.history.coverage[-1]:.3f} "
                    f"best={self.best_objective:.2f}"
                )

        client.close()
        debug_log("Training loop finished", iterations=iter_global, has_solution=self.best_solution is not None)
        if self.history.obj_max:
            print(
                f"[Training Completed] total_iters={iter_global} | "
                f"best_reward={self.best_objective:.2f} | "
                f"final_cov={self.history.coverage[-1]:.3f}"
            )
        return self.best_solution, iter_global, self.history
