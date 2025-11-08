from __future__ import annotations

import argparse
import json
import pathlib
import pickle
from typing import Optional, Tuple

import numpy as np

from mapelites_walker2D import TrainConfig, WalkerCMAMeTrainer


def _parse_phase_iter(name: str) -> Tuple[int, int]:
    parts = name.split("_")
    phase = int(parts[1].replace("phase", ""))
    iteration = int(parts[2].replace("iter", ""))
    return phase, iteration


def _find_latest_checkpoint(out_dir: pathlib.Path) -> pathlib.Path:
    candidates = []
    for path in out_dir.glob("checkpoint_phase*_iter*"):
        try:
            phase, iteration = _parse_phase_iter(path.name)
        except Exception:
            continue
        candidates.append((phase, iteration, path))
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


def _resolve_checkpoint_path(checkpoint: Optional[str], base_dir: pathlib.Path) -> pathlib.Path:
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
    iterations_per_phase: int,
    out_dir: Optional[str],
) -> None:
    base_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path(TrainConfig().out_dir)
    ckpt_dir = _resolve_checkpoint_path(checkpoint, base_dir)

    cfg, state = _load_state(ckpt_dir)
    if out_dir:
        cfg.out_dir = str(base_dir)
    cfg.iterations_per_phase = iterations_per_phase
    cfg.phases = state["config"]["phases"] + extra_phases

    trainer = WalkerCMAMeTrainer(cfg)
    trainer.history = state["history"]
    trainer.best_objective = state["best_objective"]

    best_solution_path = ckpt_dir / "best_solution.npy"
    trainer.best_solution = np.load(best_solution_path) if best_solution_path.exists() else None

    with open(ckpt_dir / "scheduler.pkl", "rb") as fh:
        trainer.scheduler = pickle.load(fh)
    trainer.archive = trainer.scheduler.archive
    trainer.emitters = trainer.scheduler.emitters

    best_elite = trainer.archive.best_elite
    if best_elite is not None:
        trainer.best_objective = float(best_elite["objective"])
        trainer.best_solution = best_elite["solution"].copy()

    completed_phase, _ = _parse_phase_iter(ckpt_dir.name)
    iter_global = len(trainer.history["obj_max"])
    total_phases = cfg.phases

    final_iter = trainer._run_phases(start_phase=completed_phase + 1, iter_global=iter_global)

    if trainer.best_solution is not None:
        np.save(pathlib.Path(cfg.out_dir) / "best_solution.npy", trainer.best_solution)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume CMA-ME Walker2d training")
    parser.add_argument(
        "--extra-phases",
        type=int,
        default=1,
        help="Number of additional phases to train",
    )
    parser.add_argument(
        "--iterations-per-phase",
        type=int,
        default=100,
        help="Iterations per resumed phase",
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
        iterations_per_phase=args.iterations_per_phase,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
