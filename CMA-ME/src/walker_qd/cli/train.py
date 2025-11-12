"""Command line entrypoint used to launch training runs."""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np

from ..algo.cma_me import CMAMETrainer
from ..config import TrainConfig
from ..logging import debug_log
from ..utils.seeding import seed_all

ALGOS = {
    "cma-me": CMAMETrainer,
}


def _infer_dims(env_id: str) -> tuple[int, int]:
    """Instantiate the env once to infer observation/action dimensions."""
    debug_log("Inferring environment dimensions", env=env_id)
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()
    debug_log("Environment dimensions inferred", obs_dim=obs_dim, act_dim=act_dim)
    return obs_dim, act_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="QD Training CLI")
    parser.add_argument("--algo", type=str, default="cma-me", choices=list(ALGOS.keys()))
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg_dict = TrainConfig().__dict__.copy()
    if args.out_dir is not None:
        cfg_dict["out_dir"] = args.out_dir
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    cfg = TrainConfig(**cfg_dict)
    debug_log("CLI configuration built", algo=args.algo, out_dir=cfg.out_dir, seed=cfg.seed)

    seed_all(cfg.seed)

    obs_dim, act_dim = _infer_dims(cfg.env_id)
    TrainerCls = ALGOS[args.algo]
    trainer = TrainerCls(cfg=cfg, obs_dim=obs_dim, act_dim=act_dim)
    debug_log("Trainer created", algo=args.algo, obs_dim=obs_dim, act_dim=act_dim)

    best_solution, total_iter, history = trainer.train()
    debug_log("Training finished", iterations=total_iter, best_found=best_solution is not None)
    if best_solution is not None:
        os.makedirs(cfg.out_dir, exist_ok=True)
        out_path = os.path.join(cfg.out_dir, "best_solution.npy")
        np.save(out_path, best_solution)
        debug_log("Best solution persisted", path=out_path)


if __name__ == "__main__":
    main()
