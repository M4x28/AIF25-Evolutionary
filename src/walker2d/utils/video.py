"""Video helpers used when exporting rollouts."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np

from ..logging import debug_log

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def record_policy_rollout(
    env_id: str,
    seed: int,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    max_episode_steps: int,
    video_seconds: int,
    render_fps: int,
    noise_vec: Optional[np.ndarray],
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> List[np.ndarray]:
    """Run a policy inside an RGB environment and collect frames."""
    env_kwargs = env_kwargs or {}
    try:
        env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    except TypeError:
        env = gym.make(env_id, **env_kwargs)
    obs, _ = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    max_frames = video_seconds * render_fps
    steps = 0
    while steps < max_episode_steps and len(frames) < max_frames:
        noisy_obs = obs if noise_vec is None else (obs + noise_vec)
        action = policy_fn(noisy_obs)
        obs, _, terminated, truncated, _ = env.step(action)
        rgb = env.render()
        if rgb is not None:
            frames.append(rgb)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return frames


def write_rgb_video(frames: List[np.ndarray], out_path: str, fps: int) -> None:
    """Serialize RGB frames to disk using OpenCV if it is available."""
    if not frames:
        debug_log("Skipping video export because no frames were produced", path=out_path)
        return
    if cv2 is None:
        debug_log("OpenCV missing, cannot write video", path=out_path)
        return

    h, w, _ = frames[0].shape
    debug_log("Writing RGB video", path=out_path, fps=fps, frames=len(frames), resolution=f"{w}x{h}")
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
