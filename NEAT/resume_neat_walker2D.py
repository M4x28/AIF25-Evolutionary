from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import pickle
from typing import Optional, Tuple

import neat
import random


from utils import set_global_seed
from trainer import NEATTrainer
from config import TrainConfig


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


def _resolve_checkpoint_path(checkpoint: Optional[str], base_dir: pathlib.Path) -> pathlib.Path:
    if checkpoint is None:
        return _find_latest_checkpoint(base_dir)
    candidate = pathlib.Path(checkpoint)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint directory {candidate} does not exist")
    return candidate


def _load_neat_state(ckpt_dir: pathlib.Path):
    neat_state_files = list(ckpt_dir.glob("neat_state-*"))
    if not neat_state_files:
        raise FileNotFoundError(f"No NEAT state file found in {ckpt_dir}")
    latest = sorted(neat_state_files)[-1]
    with gzip.open(latest, "rb") as fh:
        generation, config, population, species_set, random_state = pickle.load(fh)
    return generation, config, population, species_set, random_state


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

    trainer = NEATTrainer(cfg)
    trainer.history = state["history"]
    trainer.best_fitness = state["best_fitness"]
    best_genome_path = ckpt_dir / "best_genome.pkl"
    trainer.best_genome = pickle.load(open(best_genome_path, "rb")) if best_genome_path.exists() else None
    trainer.current_phase_noise_value = state.get("current_phase_noise_value", 0.0)

    generation, neat_config, population, species_set, random_state = _load_neat_state(ckpt_dir)
    trainer.population.population = population
    trainer.population.species = species_set
    trainer.population.generation = generation
    trainer.neat_config = neat_config
    trainer.population.config = neat_config
    trainer.stats.most_fit_genomes = []
    set_global_seed(cfg.seed)
    random.setstate(random_state)

    completed_phase, _ = _parse_phase_gen(ckpt_dir.name)
    completed_generations = len(trainer.history["gen_max"])

    gen_global = completed_generations
    total_phases = cfg.phases

    for offset in range(extra_phases):
        phase_number = completed_phase + offset + 1
        noise_vec = trainer._phase_noise(phase_number)
        print(
            f"[Resume] Phase {phase_number}/{total_phases} | "
            f"Noise vector: {noise_vec}"
        )

        for gen in range(1, cfg.max_generations_per_phase + 1):
            trainer._evaluate_current_generation(noise_vec=noise_vec)
            gen_global += 1
            gen_max, gen_avg, best = trainer._update_history()

            print(
                f"[Resume] Phase {phase_number}/{total_phases} "
                f"Gen {gen}/{cfg.max_generations_per_phase} "
                f"(global {gen_global}) | "
                f"max={gen_max:.2f} avg={gen_avg:.2f} best={best:.2f}"
            )

        trainer._save_checkpoint(phase_number, gen_global)

    trainer._save_checkpoint(cfg.phases, gen_global)
    trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume NEAT Walker2d training")
    parser.add_argument(
        "--extra-phases",
        type=int,
        default=1,
        help="Number of additional phases to train",
    )
    parser.add_argument(
        "--generations-per-phase",
        type=int,
        default=1,
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
