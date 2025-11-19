"""Utility CLI to resume a checkpoint and export the best model video."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Callable, Dict, Tuple

import neat
import numpy as np

from ..config import CMAMEConfig, CMAESConfig, NEATConfig, ensure_neat_config_file
from ..eval.neat_runner import record_genome_video
from ..eval.simulator import record_episode_video
from ..policies.mlp import two_layer_action
from ..utils.video import record_policy_rollout, write_rgb_video


def _ensure_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_state(checkpoint_dir: str) -> Tuple[Dict, Dict]:
    state_path = os.path.join(checkpoint_dir, "state.json")
    _ensure_file(state_path)
    with open(state_path, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    cfg_dict = state.get("config")
    if cfg_dict is None:
        raise ValueError("state.json does not contain the serialized config.")
    return cfg_dict, state


def _resume_cma_me(cfg_dict: Dict, state: Dict, ckpt_dir: str, out_path: str) -> None:
    cfg = CMAMEConfig(**cfg_dict)
    best_path = os.path.join(ckpt_dir, cfg.best_params_filename)
    _ensure_file(best_path)
    solution = np.load(best_path)
    frames = record_episode_video(
        env_id=cfg.env_id,
        seed=cfg.seed,
        solution=solution,
        obs_dim=state["obs_dim"],
        act_dim=state["act_dim"],
        noise_vec=None,
        max_seconds=cfg.video_seconds,
        fps=cfg.render_fps,
        max_episode_steps=cfg.max_episode_steps,
    )
    write_rgb_video(frames, out_path, cfg.render_fps)


def _resume_cma_es(cfg_dict: Dict, state: Dict, ckpt_dir: str, out_path: str) -> None:
    cfg = CMAESConfig(**cfg_dict)
    best_path = os.path.join(ckpt_dir, cfg.best_params_filename)
    _ensure_file(best_path)
    theta = np.load(best_path)

    def policy(obs: np.ndarray) -> np.ndarray:
        return two_layer_action(theta, obs, state["obs_dim"], cfg.hidden_size, state["act_dim"]).astype(np.float32)

    frames = record_policy_rollout(
        env_id=cfg.env_id,
        seed=cfg.seed,
        policy_fn=policy,
        max_episode_steps=cfg.max_episode_steps,
        video_seconds=cfg.video_seconds,
        render_fps=cfg.render_fps,
        noise_vec=None,
        env_kwargs={
            "exclude_current_positions_from_observation": cfg.exclude_current_positions_from_observation,
            "forward_reward_weight": cfg.forward_reward_weight,
        },
    )
    write_rgb_video(frames, out_path, cfg.render_fps)


def _resume_neat(cfg_dict: Dict, state: Dict, ckpt_dir: str, out_path: str) -> None:
    cfg = NEATConfig(**cfg_dict)
    best_path = os.path.join(ckpt_dir, cfg.best_params_filename)
    _ensure_file(best_path)
    with open(best_path, "rb") as fh:
        best_genome = pickle.load(fh)

    cfg_path = ensure_neat_config_file(cfg, state["obs_dim"], state["act_dim"])
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(cfg_path),
    )
    record_genome_video(cfg, best_genome, neat_config, noise_vec=None, out_path=out_path)


HANDLERS: Dict[str, Callable[[Dict, Dict, str, str], None]] = {
    "cma-me": _resume_cma_me,
    "cma-es": _resume_cma_es,
    "neat": _resume_neat,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume a checkpoint and export the best model video.")
    parser.add_argument(
        "--algo",
        choices=HANDLERS.keys(),
        required=True,
        help="Algorithm associated with the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory (e.g., runs_*/checkpoint_phaseXX_iterYYYY).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Defaults to <checkpoint>/best_model_resume.mp4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt_dir):
        raise NotADirectoryError(f"Checkpoint directory not found: {ckpt_dir}")

    cfg_dict, state = _load_state(ckpt_dir)
    output_path = args.output or os.path.join(ckpt_dir, "best_model_resume.mp4")

    handler = HANDLERS[args.algo]
    handler(cfg_dict, state, ckpt_dir, output_path)
    print(f"[Resume] Video exported to {output_path}")


if __name__ == "__main__":
    main()
