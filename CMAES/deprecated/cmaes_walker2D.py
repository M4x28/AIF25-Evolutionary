from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from cmaes import CMA

try:
    import cv2
except Exception:
    cv2 = None  # OpenCV opzionale per il video


# ======================================================================================
# Utility
# ======================================================================================


def ensure_dir(path: str) -> None:
    """Crea la directory se non esiste."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Imposta seed globale per la riproducibilità."""
    np.random.seed(seed)


# ======================================================================================
# Configurazione
# ======================================================================================


@dataclass
class TrainConfig:
    # Ambiente
    env_id: str = "Walker2d-v5"
    exclude_current_positions_from_observation: bool = False
    max_episode_steps: int = 10_000
    render_fps: int = 30

    # Rete neurale (hidden layer size fissata a 32)
    hidden_size: int = 32

    # Parametri ambiente
    forward_reward_weight: float = 2.0

    # CMA-ES
    pop_size: int = 40
    sigma_init: float = 0.1
    max_generations_per_phase: int = 100  # N
    phases: int = 3  # 3N generazioni totali

    # Noise osservazioni (costante per fase)
    noise_std: float = 0.05

    # Logging / Checkpoint
    out_dir: str = "runs_cmaes_walker2d"
    video_seconds: int = 20

    # Seed
    seed: int = 42


# ======================================================================================
# Rete neurale: funzioni helper
# ======================================================================================


def total_parameters(obs_dim: int, hidden_size: int, act_dim: int) -> int:
    return (obs_dim * hidden_size) + (hidden_size * act_dim) + hidden_size + act_dim


def get_action(
    model: np.ndarray,
    observation: np.ndarray,
    obs_dim: int,
    hidden_size: int,
    act_dim: int,
) -> np.ndarray:
    """
    Calcola l'azione a partire dai parametri flat della rete:
      - input -> hidden (ReLU)
      - hidden -> output (Tanh per restare in [-1, 1])
    """
    expected = total_parameters(obs_dim, hidden_size, act_dim)
    if model.shape[0] != expected:
        raise ValueError(f"Model size {model.shape[0]} != expected {expected}")

    i2h_end = obs_dim * hidden_size
    h2o_end = i2h_end + hidden_size * act_dim
    b1_end = h2o_end + hidden_size

    w1 = model[:i2h_end].reshape(obs_dim, hidden_size)
    w2 = model[i2h_end:h2o_end].reshape(hidden_size, act_dim)
    b1 = model[h2o_end:b1_end].reshape(1, hidden_size)
    b2 = model[b1_end:].reshape(1, act_dim)

    if observation.ndim == 1:
        observation = observation.reshape(1, -1)

    hidden = np.tanh(observation @ w1 + b1)
    outputs = np.tanh(hidden @ w2 + b2)
    return outputs if outputs.shape[0] > 1 else outputs.flatten()


# ======================================================================================
# Runner a singolo environment
# ======================================================================================


class EnvRunner:
    """
    Valuta una policy in un singolo episodio su un singolo environment.
    Il rumore è aggiunto a tutte le feature d'ingresso e resta costante per l'intera fase.
    """

    def __init__(self, cfg: TrainConfig, obs_dim: int, act_dim: int, hidden_size: int):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
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

    def evaluate(self, model: np.ndarray, noise_vec: Optional[np.ndarray]) -> float:
        obs, _ = self.env.reset(seed=self.cfg.seed)
        cum_reward = 0.0
        steps = 0

        while steps < self.cfg.max_episode_steps:
            noisy_obs = obs if noise_vec is None else (obs + noise_vec)
            action = get_action(model, noisy_obs, self.obs_dim, self.hidden_size, self.act_dim).astype(np.float32)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            cum_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break

        return cum_reward


# ======================================================================================
# Video best model
# ======================================================================================


def record_video(
    cfg: TrainConfig,
    model: np.ndarray,
    obs_dim: int,
    hidden_size: int,
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

    frames: List[np.ndarray] = []
    max_frames = cfg.video_seconds * cfg.render_fps
    steps = 0

    while steps < cfg.max_episode_steps and len(frames) < max_frames:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = get_action(model, noisy_obs, obs_dim, hidden_size, act_dim).astype(np.float32)
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
# Trainer CMA-ES
# ======================================================================================


class CMAESTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        ensure_dir(cfg.out_dir)

        set_global_seed(cfg.seed)

        probe = gym.make(
            cfg.env_id,
            exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
        )
        self.obs_dim = int(np.prod(probe.observation_space.shape))
        self.act_dim = int(np.prod(probe.action_space.shape))
        probe.close()

        self.hidden_size = cfg.hidden_size
        self.total_params = total_parameters(self.obs_dim, self.hidden_size, self.act_dim)

        self.runner = EnvRunner(cfg, self.obs_dim, self.act_dim, self.hidden_size)

        self.optimizer = CMA(
            mean=np.zeros(self.total_params, dtype=np.float64),
            sigma=cfg.sigma_init,
            population_size=cfg.pop_size,
            seed=cfg.seed,
        )

        self.history: Dict[str, List[float]] = {"gen_max": [], "gen_avg": []}
        self.best_theta: Optional[np.ndarray] = None
        self.best_reward: float = -np.inf
        self.current_phase_noise_value: float = 0.0

    def close(self) -> None:
        self.runner.close()

    def _phase_noise(self, phase_idx: int) -> Tuple[float, np.ndarray]:
        phase_noise_value = float(np.random.standard_normal()) * self.cfg.noise_std
        noise_vec = np.full((self.obs_dim,), phase_noise_value, dtype=np.float32)
        self.current_phase_noise_value = phase_noise_value

        np.save(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), noise_vec)
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w") as f:
            json.dump({"phase_noise_value": phase_noise_value}, f, indent=2)

        return phase_noise_value, noise_vec

    def _save_checkpoint(self, phase_idx: int, gen_global: int) -> None:
        ckpt_dir = os.path.join(self.cfg.out_dir, f"checkpoint_phase{phase_idx:02d}_gen{gen_global:04d}")
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
            "current_phase_noise_value": self.current_phase_noise_value,
        }
        if self.best_theta is not None:
            np.save(os.path.join(ckpt_dir, "best_theta.npy"), self.best_theta)
        with open(os.path.join(ckpt_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

        plt.figure(figsize=(7, 4))
        plt.plot(self.history["gen_max"], label="Max reward")
        plt.plot(self.history["gen_avg"], label="Avg reward")
        plt.xlabel("Generazione")
        plt.ylabel("Reward")
        plt.title("CMA-ES Walker2d (seed=42)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "reward_curve.png"))
        plt.close()

        if self.best_theta is not None:
            noise_path = os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy")
            noise_vec = np.load(noise_path) if os.path.exists(noise_path) else None
            try:
                record_video(
                    self.cfg,
                    self.best_theta,
                    self.obs_dim,
                    self.hidden_size,
                    self.act_dim,
                    noise_vec,
                    os.path.join(ckpt_dir, "best.mp4"),
                )
            except Exception:
                pass

    def train(self) -> None:
        total_gens = self.cfg.max_generations_per_phase * self.cfg.phases
        gen_global = 0

        for phase_idx in range(1, self.cfg.phases + 1):
            phase_noise_value, noise_vec = self._phase_noise(phase_idx)
            print(
                f"[Phase {phase_idx}/{self.cfg.phases}] "
                f"Noise value kept constant this phase: {phase_noise_value:.5f}"
            )

            for gen in range(1, self.cfg.max_generations_per_phase + 1):
                gen_global += 1
                rewards_this_gen: List[float] = []
                solutions = []

                for _ in range(self.optimizer.population_size):
                    theta = self.optimizer.ask()
                    reward = self.runner.evaluate(theta, noise_vec=noise_vec)
                    rewards_this_gen.append(reward)
                    solutions.append((theta, -reward))

                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.best_theta = theta.copy()

                self.optimizer.tell(solutions)

                gen_max = float(np.max(rewards_this_gen))
                gen_avg = float(np.mean(rewards_this_gen))
                self.history["gen_max"].append(gen_max)
                self.history["gen_avg"].append(gen_avg)

                print(
                    f"[Phase {phase_idx}/{self.cfg.phases}] "
                    f"Gen {gen}/{self.cfg.max_generations_per_phase} "
                    f"(global {gen_global}/{total_gens}) | "
                    f"max={gen_max:.2f} avg={gen_avg:.2f} best={self.best_reward:.2f}"
                )

            self._save_checkpoint(phase_idx, gen_global)

        self._save_checkpoint(self.cfg.phases, total_gens)
        self.close()


# ======================================================================================
# Entry point
# ======================================================================================


def main() -> None:
    cfg = TrainConfig()
    trainer = CMAESTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
