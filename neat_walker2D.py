from __future__ import annotations

import json
import os
import pathlib
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import neat
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None  # Video opzionale


# ======================================================================================
# Utility
# ======================================================================================


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


DEFAULT_NEAT_CONFIG_TEMPLATE = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 9999999
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.4
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob           = 0.1
conn_delete_prob        = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.1
node_delete_prob        = 0.1
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.4
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 10
species_elitism      = 1

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.3
"""


def ensure_neat_config(path: pathlib.Path, pop_size: int, num_inputs: int, num_outputs: int) -> None:
    if path.exists():
        return
    ensure_dir(str(path.parent))
    config_text = DEFAULT_NEAT_CONFIG_TEMPLATE.format(
        pop_size=pop_size, num_inputs=num_inputs, num_outputs=num_outputs
    )
    path.write_text(config_text)


# ======================================================================================
# Configurazione
# ======================================================================================


@dataclass
class TrainConfig:
    env_id: str = "Walker2d-v5"
    exclude_current_positions_from_observation: bool = False
    max_episode_steps: int = 10_000
    render_fps: int = 30
    forward_reward_weight: float = 2.0

    pop_size: int = 40
    max_generations_per_phase: int = 100
    phases: int = 3

    noise_std: float = 0.05

    out_dir: str = "runs_neat_2_walker2d"
    video_seconds: int = 20

    seed: int = 42

    neat_config_path: str = "configs/neat_walker2d.cfg"


# ======================================================================================
# NEAT Evaluation Helpers
# ======================================================================================


class WalkerEvaluator:
    """Valuta una genome NEAT su Walker2d con rumore costante per fase."""

    def __init__(self, cfg: TrainConfig, obs_dim: int, act_dim: int):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            forward_reward_weight=cfg.forward_reward_weight,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def evaluate_genomes(
        self,
        genomes,
        neat_config: neat.Config,
        noise_vec: Optional[np.ndarray],
    ) -> None:
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            obs, _ = self.env.reset(seed=self.cfg.seed)

            cum_reward = 0.0
            steps = 0

            while steps < self.cfg.max_episode_steps:
                action_raw = np.array(net.activate(obs.tolist()), dtype=np.float32)
                action = np.tanh(action_raw if noise_vec is None else (action_raw + noise_vec))

                obs, reward, terminated, truncated, _ = self.env.step(action)
                cum_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break

            genome.fitness = cum_reward


# ======================================================================================
# Video helper
# ======================================================================================


def record_video(
    cfg: TrainConfig,
    genome,
    neat_config: neat.Config,
    obs_dim: int,
    act_dim: int,
    noise_vec: Optional[np.ndarray],
    out_path: str,
) -> None:
    if cv2 is None:
        return

    env = gym.make(
        cfg.env_id,
        render_mode="rgb_array",
        exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
        forward_reward_weight=cfg.forward_reward_weight,
    )
    obs, _ = env.reset(seed=cfg.seed)
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    frames: List[np.ndarray] = []
    max_frames = cfg.video_seconds * cfg.render_fps
    steps = 0

    while steps < cfg.max_episode_steps and len(frames) < max_frames:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = np.tanh(np.array(net.activate(noisy_obs.tolist()), dtype=np.float32))
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, cfg.render_fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()


# ======================================================================================
# Trainer NEAT
# ======================================================================================


class NEATTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        ensure_dir(cfg.out_dir)
        set_global_seed(cfg.seed)

        probe = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
            forward_reward_weight=cfg.forward_reward_weight,
        )
        self.obs_dim = int(np.prod(probe.observation_space.shape)) 
        self.act_dim = int(np.prod(probe.action_space.shape))
        probe.close()

        config_path = pathlib.Path(cfg.neat_config_path)
        ensure_neat_config(config_path, cfg.pop_size, self.obs_dim, self.act_dim)
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        # Garantiamo pop_size coerente con TrainConfig
        self.neat_config.pop_size = cfg.pop_size
        self.neat_config.population_size = cfg.pop_size

        self.population = neat.Population(self.neat_config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

        self.evaluator = WalkerEvaluator(cfg, self.obs_dim, self.act_dim)

        self.history: Dict[str, List[float]] = {"gen_max": [], "gen_avg": []}
        self.best_genome = None
        self.best_fitness: float = -np.inf
        self.current_phase_noise_value: float = 0.0

    def close(self) -> None:
        self.evaluator.close()

    # Sostituzione in _phase_noise
    def _phase_noise(self, phase_idx: int) -> Tuple[float, np.ndarray]:
        # Genera un vettore di rumore, non un bias scalare
        noise_vec = np.random.normal(loc=0.0, scale=self.cfg.noise_std, size=(self.obs_dim,)).astype(np.float32)
        
        # Salva il vettore (l'idea di salvare il "valore" scalare non è più valida)
        np.save(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), noise_vec)
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w") as fh:
            # Salva una statistica, es. la norma del vettore
            json.dump({"phase_noise_norm": float(np.linalg.norm(noise_vec))}, fh, indent=2)

        return noise_vec

    def _save_checkpoint(self, phase_idx: int, gen_global: int) -> None:
        ckpt_dir = os.path.join(self.cfg.out_dir, f"checkpoint_phase{phase_idx:02d}_gen{gen_global:04d}")
        ensure_dir(ckpt_dir)

        state = {
            "config": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "history": self.history,
            "best_fitness": float(self.best_fitness),
            "current_phase_noise_value": self.current_phase_noise_value,
            "neat_generation": self.population.generation,
        }
        with open(os.path.join(ckpt_dir, "state.json"), "w") as fh:
            json.dump(state, fh, indent=2)

        if self.best_genome is not None:
            with open(os.path.join(ckpt_dir, "best_genome.pkl"), "wb") as fh:
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

        plt.figure(figsize=(7, 4))
        plt.plot(self.history["gen_max"], label="Max fitness")
        plt.plot(self.history["gen_avg"], label="Avg fitness")
        plt.xlabel("Generazione")
        plt.ylabel("Fitness")
        plt.title("NEAT Walker2d (seed=42)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "fitness_curve.png"))
        plt.close()

        if self.best_genome is not None:
            noise_path = os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy")
            noise_vec = np.load(noise_path) if os.path.exists(noise_path) else None
            try:
                record_video(
                    self.cfg,
                    self.best_genome,
                    self.neat_config,
                    self.obs_dim,
                    self.act_dim,
                    noise_vec,
                    os.path.join(ckpt_dir, "best.mp4"),
                )
            except Exception:
                pass

    def _evaluate_current_generation(self, noise_vec: Optional[np.ndarray]) -> None:
        def eval_fn(genomes, neat_cfg):
            self.evaluator.evaluate_genomes(genomes, neat_cfg, noise_vec)

        self.population.run(eval_fn, 1)

    def _update_history(self) -> Tuple[float, float, float]:
        if not self.stats.most_fit_genomes:
            return 0.0, 0.0, self.best_fitness

        current_best = self.stats.most_fit_genomes[-1]
        gen_max = float(current_best.fitness)

        fitness_means = self.stats.get_fitness_mean()
        gen_avg = float(fitness_means[-1]) if fitness_means else 0.0

        self.history["gen_max"].append(gen_max)
        self.history["gen_avg"].append(gen_avg)

        if gen_max > self.best_fitness or self.best_genome is None:
            self.best_fitness = gen_max
            self.best_genome = pickle.loads(pickle.dumps(current_best))

        return gen_max, gen_avg, self.best_fitness

    def train(self) -> None:
        total_generations = self.cfg.max_generations_per_phase * self.cfg.phases
        gen_global = 0

        for phase_idx in range(1, self.cfg.phases + 1):
            noise_vec = self._phase_noise(phase_idx)
            print(
                f"[Phase {phase_idx}/{self.cfg.phases}] Noise vector kept constant this phase: {noise_vec}"
            )

            for gen_idx in range(1, self.cfg.max_generations_per_phase + 1):
                self._evaluate_current_generation(noise_vec=noise_vec)
                gen_global += 1

                gen_max, gen_avg, best = self._update_history()
                print(
                    f"[Phase {phase_idx}/{self.cfg.phases}] "
                    f"Gen {gen_idx}/{self.cfg.max_generations_per_phase} "
                    f"(global {gen_global}/{total_generations}) | "
                    f"max={gen_max:.2f} avg={gen_avg:.2f} best={best:.2f}"
                )

            self._save_checkpoint(phase_idx, gen_global)

        self._save_checkpoint(self.cfg.phases, total_generations)
        self.close()


# ======================================================================================
# Entry point
# ======================================================================================


def main() -> None:
    cfg = TrainConfig()
    trainer = NEATTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
