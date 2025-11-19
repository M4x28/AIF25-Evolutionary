"""NEAT baseline integrated inside the walker2d package."""
from __future__ import annotations

import json
import os
import pickle
import multiprocessing as mp
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import neat
import numpy as np

from ..config import NEATConfig, ensure_neat_config_file
from ..eval.neat_runner import evaluate_single_genome, record_genome_video
from ..logging.metrics import plot_training_curves
from ..utils import debug_log, ensure_dir, seed_all


class NEATTrainer:
    """Parallel NEAT training with per-phase observation noise."""

    def __init__(self, cfg: NEATConfig) -> None:
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

        cfg_path = ensure_neat_config_file(cfg, self.obs_dim, self.act_dim)
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(cfg_path),
        )
        self.neat_config.pop_size = cfg.pop_size
        self.neat_config.population_size = cfg.pop_size

        self.population = neat.Population(self.neat_config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

        worker_count = cfg.workers if cfg.workers > 0 else (os.cpu_count() or 1)
        debug_log("Starting NEAT worker pool", workers=worker_count)
        self.pool = mp.Pool(processes=worker_count)
        print(f"[NEAT] active workers: {worker_count}")

        self.history: Dict[str, List[float]] = {"gen_max": [], "gen_avg": []}
        self.best_genome: Optional[neat.DefaultGenome] = None
        self.best_fitness: float = float("-inf")

    def close(self) -> None:
        debug_log("Closing NEAT worker pool")
        self.pool.close()
        self.pool.join()

    def _phase_noise(self, phase_idx: int) -> np.ndarray:
        noise_vec = np.random.normal(0.0, self.cfg.noise_std, size=(self.obs_dim,)).astype(np.float32)
        np.save(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), noise_vec)
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w", encoding="utf-8") as fh:
            json.dump({"phase_noise_norm": float(np.linalg.norm(noise_vec))}, fh, indent=2)
        debug_log("NEAT phase noise generated", phase=phase_idx, noise_vector=noise_vec.tolist())
        return noise_vec

    def _save_checkpoint(self, phase_idx: int, gen_global: int) -> None:
        ckpt_dir = os.path.join(self.cfg.out_dir, f"checkpoint_phase{phase_idx:02d}_iter{gen_global:04d}")
        ensure_dir(ckpt_dir)
        state = {
            "config": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "history": self.history,
            "best_fitness": float(self.best_fitness),
            "neat_generation": self.population.generation,
        }
        with open(os.path.join(ckpt_dir, "state.json"), "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

        if self.best_genome is not None:
            with open(os.path.join(ckpt_dir, self.cfg.best_params_filename), "wb") as fh:
                pickle.dump(self.best_genome, fh)

        checkpointer = neat.Checkpointer(
            generation_interval=None,
            time_interval_seconds=None,
            filename_prefix=os.path.join(ckpt_dir, "neat_state-"),
        )
        checkpointer.save_checkpoint(
            self.neat_config,
            self.population.population,
            self.population.species,
            self.population.generation,
        )

        plot_training_curves(
            self.history,
            os.path.join(ckpt_dir, self.cfg.reward_curve_filename),
            series=[("gen_max", "Max fitness"), ("gen_avg", "Avg fitness")],
            x_label="Generation",
            left_label="Fitness",
            title=f"NEAT Walker2d (seed={self.cfg.seed})",
            figsize=(7, 4),
        )

        if self.cfg.allow_save_best_model_video and self.best_genome is not None:
            noise_path = os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy")
            noise_vec = np.load(noise_path) if os.path.exists(noise_path) else None
            try:
                record_genome_video(
                    self.cfg,
                    self.best_genome,
                    self.neat_config,
                    noise_vec,
                    os.path.join(ckpt_dir, self.cfg.best_video_filename),
                )
            except Exception:
                debug_log("Video export failed for NEAT checkpoint", phase=phase_idx)

    def _evaluate_current_generation(self, noise_vec: Optional[np.ndarray]) -> None:
        eval_seed = self.cfg.seed

        def eval_fn(genomes: List[Tuple[int, neat.DefaultGenome]], neat_cfg: neat.Config) -> None:
            tasks = [(genome, neat_cfg, self.cfg, noise_vec, eval_seed) for _, genome in genomes]
            try:
                fitness_list = self.pool.map(evaluate_single_genome, tasks)
            except Exception as exc:  # pragma: no cover - defensive fallback
                debug_log("Parallel NEAT evaluation failed", error=str(exc))
                fitness_list = [0.0] * len(genomes)

            for (_, genome), fitness in zip(genomes, fitness_list):
                genome.fitness = fitness

        self.population.run(eval_fn, 1)

    def _update_history(self) -> Tuple[float, float, float]:
        if not self.stats.most_fit_genomes:
            return 0.0, 0.0, self.best_fitness

        current_best = self.stats.most_fit_genomes[-1]
        gen_max = float(current_best.fitness)
        means = self.stats.get_fitness_mean()
        gen_avg = float(means[-1]) if means else 0.0
        self.history["gen_max"].append(gen_max)
        self.history["gen_avg"].append(gen_avg)

        if gen_max > self.best_fitness or self.best_genome is None:
            self.best_fitness = gen_max
            self.best_genome = pickle.loads(pickle.dumps(current_best))

        return gen_max, gen_avg, self.best_fitness

    def train(self) -> Tuple[Optional[neat.DefaultGenome], int, Dict[str, List[float]]]:
        total_generations = self.cfg.iterations_per_phase * self.cfg.phases
        gen_global = 0

        for phase_idx in range(1, self.cfg.phases + 1):
            noise_vec = self._phase_noise(phase_idx)
            print(
                f"[Phase {phase_idx}/{self.cfg.phases}] noise vector "
                f"{np.array2string(noise_vec, precision=3, separator=', ')}"
            )

            for gen_idx in range(1, self.cfg.iterations_per_phase + 1):
                self._evaluate_current_generation(noise_vec)
                gen_global += 1
                gen_max, gen_avg, best = self._update_history()
                print(
                    f"[Phase {phase_idx}/{self.cfg.phases}] "
                    f"Gen {gen_idx}/{self.cfg.iterations_per_phase} "
                    f"(global {gen_global}/{total_generations}) | "
                    f"max={gen_max:.2f} avg={gen_avg:.2f} best={best:.2f}"
                )

            self._save_checkpoint(phase_idx, gen_global)

        self._save_checkpoint(self.cfg.phases, total_generations)
        self.close()
        return self.best_genome, gen_global, self.history
