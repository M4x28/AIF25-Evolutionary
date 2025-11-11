# visualization.py
import gymnasium as gym
import numpy as np
from typing import Optional, List

try:
    import cv2
except Exception:
    cv2 = None # Video opzionale

# Importa la definizione della configurazione
from config import TrainConfig 

from neuralnetwork import get_action

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
