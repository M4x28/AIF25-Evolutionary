# evaluation.py
import gymnasium as gym
import numpy as np
from typing import Tuple, List, Optional

# Importa la definizione della configurazione
from config import TrainConfig
from neuralnetwork import get_action

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


def eval_single_genome(params: Tuple) -> float:
    """
    Funzione eseguita da un processo worker.
    Valuta un singolo genoma in un ambiente dedicato.

    I parametri sono passati come tupla per compatibilità con pool.map.
    """
    theta, cfg_dict, obs_dim, act_dim, hidden_size, noise_vec = params

    # Ricrea la configurazione e il runner nel worker
    # È FONDAMENTALE creare un'istanza di Env (e quindi EnvRunner)
    # separata per ogni processo worker.
    cfg = TrainConfig(**cfg_dict)
    runner = EnvRunner(cfg, obs_dim, act_dim, hidden_size)

    try:
        reward = runner.evaluate(theta, noise_vec)
    finally:
        runner.close()  # Assicurati che l'ambiente del worker sia chiuso

    return float(reward)
