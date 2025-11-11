from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional, Tuple

import numpy as np

from CMAES.cmaes_walker2D import CMAESTrainer, TrainConfig, set_global_seed


def _parse_phase_gen(name: str) -> Tuple[int, int]:
    parts = name.split("_")
    phase = int(parts[1].replace("phase", ""))
    gen = int(parts[2].replace("gen", ""))
    return phase, gen


def _find_latest_checkpoint(out_dir: pathlib.Path) -> pathlib.Path:
    candidates = []
    for path in out_dir.glob("checkpoint_phase*_gen*"):
        try:
            phase, gen = _parse_phase_gen(path.name)
        except Exception:
            continue
        candidates.append((phase, gen, path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")
    candidates.sort()
    return candidates[-1][2]


def _load_state(ckpt_dir: pathlib.Path) -> Tuple[TrainConfig, dict]:
    state_path = ckpt_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json in {ckpt_dir}")
    with state_path.open("r") as fh:
        state = json.load(fh)
    cfg = TrainConfig(**state["config"])
    return cfg, state


def _resolve_checkpoint_path(
    checkpoint: Optional[str],
    base_dir: pathlib.Path,
) -> pathlib.Path:
    if checkpoint is None:
        return _find_latest_checkpoint(base_dir)

    candidate = pathlib.Path(checkpoint)
    if not candidate.is_absolute():
        candidate = base_dir / candidate

    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint directory {candidate} does not exist")
    return candidate


def resume_training(
    checkpoint: Optional[str],
    extra_phases: int,
    generations_per_phase: int,
    out_dir: Optional[str],
) -> None:
    base_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path(TrainConfig().out_dir)
    ckpt_dir = _resolve_checkpoint_path(checkpoint, base_dir)

    cfg, state = _load_state(ckpt_dir)
    if out_dir:
        cfg.out_dir = str(base_dir)
    cfg.max_generations_per_phase = generations_per_phase
    cfg.phases = state["config"]["phases"] + extra_phases

    trainer = CMAESTrainer(cfg)
    trainer.history = state["history"]
    trainer.best_reward = state["best_reward"]
    best_theta_path = ckpt_dir / "best_theta.npy"
    trainer.best_theta = np.load(best_theta_path) if best_theta_path.exists() else None
    trainer.optimizer._mean = np.array(state["cma_mean"], dtype=np.float64)
    trainer.optimizer._sigma = float(state["cma_sigma"])
    trainer.current_phase_noise_value = state.get("current_phase_noise_value", 0.0)

    completed_phase, _ = _parse_phase_gen(ckpt_dir.name)
    completed_generations = len(trainer.history["gen_max"])

    set_global_seed(cfg.seed)
    for _ in range(completed_phase):
        np.random.standard_normal()

    gen_global = completed_generations
    total_phases = cfg.phases

    for offset in range(extra_phases):
        phase_number = completed_phase + offset + 1
        phase_noise_value, noise_vec = trainer._phase_noise(phase_number)
        print(
            f"[Resume] Phase {phase_number}/{total_phases} | "
            f"Noise value: {phase_noise_value:.5f}"
        )

        for gen in range(1, cfg.max_generations_per_phase + 1):
            gen_global += 1
            rewards_this_gen = []
            solutions = []

            for _ in range(trainer.optimizer.population_size):
                theta = trainer.optimizer.ask()
                reward = trainer.runner.evaluate(theta, noise_vec=noise_vec)
                rewards_this_gen.append(reward)
                solutions.append((theta, -reward))

                if reward > trainer.best_reward:
                    trainer.best_reward = reward
                    trainer.best_theta = theta.copy()

            trainer.optimizer.tell(solutions)

            gen_max = float(np.max(rewards_this_gen))
            gen_avg = float(np.mean(rewards_this_gen))
            trainer.history["gen_max"].append(gen_max)
            trainer.history["gen_avg"].append(gen_avg)

            print(
                f"[Resume] Phase {phase_number}/{total_phases} "
                f"Gen {gen}/{cfg.max_generations_per_phase} "
                f"(global {gen_global}) | "
                f"max={gen_max:.2f} avg={gen_avg:.2f} best={trainer.best_reward:.2f}"
            )

        trainer._save_checkpoint(phase_number, gen_global)

    trainer._save_checkpoint(cfg.phases, gen_global)
    trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume CMA-ES Walker2d training")
    parser.add_argument(
        "--extra-phases",
        type=int,
        default=1,
        help="Number of additional phases to train",
    )
    parser.add_argument(
        "--generations-per-phase",
        type=int,
        default=100,
        help="Number of generations (N) to run in each resumed phase",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory name or path to resume from (defaults to latest in out-dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Root directory containing checkpoints (used when --checkpoint is relative or omitted)",
    )
    args = parser.parse_args()

    resume_training(
        checkpoint=args.checkpoint,
        extra_phases=args.extra_phases,
        generations_per_phase=args.generations_per_phase,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
