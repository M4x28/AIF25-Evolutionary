"""Evaluation helpers specific to NEAT genomes."""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import neat
import numpy as np

from ..config import NEATConfig
from ..utils.video import record_policy_rollout, write_rgb_video


def evaluate_single_genome(args: Tuple[neat.DefaultGenome, neat.Config, NEATConfig, Optional[np.ndarray], int]) -> float:
    """Run a single genome rollout inside its own environment instance."""
    genome, neat_config, cfg, noise_vec, eval_seed = args
    env = gym.make(
        cfg.env_id,
        exclude_current_positions_from_observation=cfg.exclude_current_positions_from_observation,
        forward_reward_weight=cfg.forward_reward_weight,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    obs, _ = env.reset(seed=eval_seed)
    cum_reward = 0.0
    steps = 0

    while steps < cfg.max_episode_steps:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = np.tanh(np.array(net.activate(noisy_obs.tolist()), dtype=np.float32))
        obs, reward, terminated, truncated, _ = env.step(action)
        cum_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return cum_reward


def record_genome_video(
    cfg: NEATConfig,
    genome: neat.DefaultGenome,
    neat_config: neat.Config,
    noise_vec: Optional[np.ndarray],
    out_path: str,
) -> None:
    """Render a video showcasing the provided genome."""
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    frames = record_policy_rollout(
        env_id=cfg.env_id,
        seed=cfg.seed,
        policy_fn=lambda obs: np.tanh(np.array(net.activate(obs.tolist()), dtype=np.float32)),
        max_episode_steps=cfg.max_episode_steps,
        video_seconds=cfg.video_seconds,
        render_fps=cfg.render_fps,
        noise_vec=noise_vec,
        env_kwargs={
            "exclude_current_positions_from_observation": cfg.exclude_current_positions_from_observation,
            "forward_reward_weight": cfg.forward_reward_weight,
        },
    )
    write_rgb_video(frames, out_path, cfg.render_fps)
