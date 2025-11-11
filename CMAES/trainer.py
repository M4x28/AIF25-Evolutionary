from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import multiprocessing 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from cmaes import CMA

try:
    import cv2
except Exception:
    cv2 = None  # OpenCV opzionale per il video

from utils import ensure_dir, set_global_seed
from config import TrainConfig
from visualization import record_video
from neuralnetwork import total_parameters
from evaluation import eval_single_genome

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

        # --- MODIFICA: Rimosso self.runner, aggiunto Pool ---
        # self.runner = EnvRunner(cfg, self.obs_dim, self.act_dim, self.hidden_size)
        self.num_cores = multiprocessing.cpu_count()
        print(f"--- Creazione di un Pool con {self.num_cores} worker paralleli ---")
        self.pool = multiprocessing.Pool(processes=self.num_cores)
        # --- Fine Modifica ---

        self.optimizer = CMA(
            mean=np.zeros(self.total_params, dtype=np.float64),
            sigma=cfg.sigma_init,
            population_size=cfg.pop_size,
            seed=cfg.seed,
        )

        self.history: Dict[str, List[float]] = {"gen_max": [], "gen_avg": []}
        self.best_theta: Optional[np.ndarray] = None
        self.best_reward: float = -np.inf
        self.current_phase_noise_vector: Optional[np.ndarray] = None

    def close(self) -> None:
        # --- MODIFICA: Chiusura del Pool ---
        print("\n--- Chiusura del Pool di worker ---")
        try:
            self.pool.close()
            self.pool.join()
        except Exception:
            pass
        # self.runner.close() # Rimosso
        # --- Fine Modifica ---

    def _phase_noise(self, phase_idx: int) -> Tuple[float, np.ndarray]:
        # Genera un vettore di rumore Gaussiano (media 0)
        noise_vec = np.random.normal(
            loc=0.0, 
            scale=self.cfg.noise_std, 
            size=(self.obs_dim,)
        ).astype(np.float32)
        
        # Salva il vettore di rumore in un file binario NumPy
        np.save(
            os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.npy"), 
            noise_vec
        )
        
        # Salva la norma L2 (grandezza) del vettore in un file JSON
        # per un rapido controllo
        with open(os.path.join(self.cfg.out_dir, f"phase_{phase_idx:02d}_noise.json"), "w") as fh:
            json.dump(
                {"phase_noise_norm": float(np.linalg.norm(noise_vec))}, 
                fh, 
                indent=2
            )

        # Restituisce il vettore generato
        return noise_vec


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
            "current_phase_noise_vector": self.current_phase_noise_vector,
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
        plt.title(f"CMA-ES Walker2d (seed={self.cfg.seed})")
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
        
        # --- MODIFICA: Passa la config come dizionario per il pickling ---
        cfg_dict = asdict(self.cfg)

        for phase_idx in range(1, self.cfg.phases + 1):
            noise_vec = self._phase_noise(phase_idx)
            print(
                f"[Phase {phase_idx}/{self.cfg.phases}] "
                f"Noise vector kept constant this phase: {noise_vec:.5f}"
            )

            for gen in range(1, self.cfg.max_generations_per_phase + 1):
                gen_global += 1

                # --- MODIFICA: Logica di valutazione parallela ---
                
                # 1. Chiedi tutti i genomi (theta) all'ottimizzatore
                thetas = [self.optimizer.ask() for _ in range(self.optimizer.population_size)]

                # 2. Prepara i parametri per ogni task del pool
                tasks = [
                    (theta, cfg_dict, self.obs_dim, self.act_dim, self.hidden_size, noise_vec)
                    for theta in thetas
                ]
                
                # 3. Esegui le valutazioni in parallelo
                # pool.map mantiene l'ordine, quindi rewards_this_gen[i] 
                # corrisponde a thetas[i]
                rewards_this_gen: List[float] = self.pool.map(eval_single_genome, tasks)
                
                # 4. Prepara le soluzioni per l'ottimizzatore e aggiorna il best
                solutions = []
                for theta, reward in zip(thetas, rewards_this_gen):
                    # CMA-ES minimizza, quindi passiamo il negativo del reward
                    solutions.append((theta, -reward))
                    
                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.best_theta = theta.copy()
                
                # --- Fine Modifica ---

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
        self.close()  # Chiude il pool