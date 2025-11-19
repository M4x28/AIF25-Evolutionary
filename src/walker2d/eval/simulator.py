"""Simulation utilities used by the CMA-ME trainer."""
from __future__ import annotations

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

from ..policies.policy import LinearTanhPolicy
from ..logging import debug_log
from ..utils.video import record_policy_rollout


def simulate_policy(
    model: np.ndarray,
    env_id: str,
    seed: int,
    obs_dim: int,
    act_dim: int,
    noise_vec: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    """Run a single episode and return reward, velocity and torso height."""
    debug_log("Simulating policy", env=env_id, seed=seed)
    env = gym.make(env_id)
    policy = LinearTanhPolicy(obs_dim=obs_dim, act_dim=act_dim)
    weights = policy.weights_from_vector(model)
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    vel_samples: List[float] = []
    h_samples: List[float] = []

    prev_x = float(env.unwrapped.data.qpos[0])
    dt = float(getattr(env.unwrapped, "dt", 0.02))

    done = False
    while not done:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = policy.act(weights, noisy_obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        x_now = float(env.unwrapped.data.qpos[0])
        vel_samples.append((x_now - prev_x) / dt)
        prev_x = x_now

        if hasattr(env.unwrapped, "data"):
            h_samples.append(float(env.unwrapped.data.qpos[1]))
        elif info and "z_distance_from_origin" in info:
            h_samples.append(float(info["z_distance_from_origin"]) + 1.0)

    env.close()
    avg_vel = float(np.mean(vel_samples)) if vel_samples else 0.0
    avg_h = float(np.mean(h_samples)) if h_samples else 0.0
    debug_log("Simulation finished", reward=total_reward, avg_vel=avg_vel, avg_h=avg_h)
    return total_reward, avg_vel, avg_h


def record_episode_video(
    env_id: str,
    seed: int,
    solution: np.ndarray,
    obs_dim: int,
    act_dim: int,
    noise_vec: Optional[np.ndarray],
    max_seconds: int,
    fps: int,
    max_episode_steps: int,
) -> list:
    """Collect RGB frames for a policy rollout (if supported by the env)."""
    debug_log("Recording episode video", env=env_id, fps=fps, duration=max_seconds)
    policy = LinearTanhPolicy(obs_dim=obs_dim, act_dim=act_dim)
    weights = policy.weights_from_vector(solution)
    frames = record_policy_rollout(
        env_id=env_id,
        seed=seed,
        policy_fn=lambda obs: policy.act(weights, obs),
        max_episode_steps=max_episode_steps,
        video_seconds=max_seconds,
        render_fps=fps,
        noise_vec=noise_vec,
    )
    debug_log("Episode recording finished", frames=len(frames))
    return frames
