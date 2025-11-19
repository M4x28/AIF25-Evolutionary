"""Checkpoint helpers for persisting training progress."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ribs.visualize import grid_archive_heatmap

from ..logging import debug_log
from ..logging.metrics import History, plot_training_curves
from ..utils.io import ensure_dir


@dataclass(frozen=True)
class CheckpointPaths:
    """Encapsulates the on-disk layout for a checkpoint bundle."""

    ckpt_dir: str
    state_json: str
    archive_npz: str
    scheduler_pkl: str
    best_solution: str
    phase_solution: str
    reward_curve: str
    heatmap: str

    @classmethod
    def build(
        cls,
        out_dir: str,
        phase_idx: int,
        iter_global: int,
        best_params_filename: str,
        phase_params_filename: str,
        reward_curve_filename: str,
    ) -> "CheckpointPaths":
        ckpt_dir = os.path.join(out_dir, f"checkpoint_phase{phase_idx:02d}_iter{iter_global:04d}")
        ensure_dir(ckpt_dir)
        return cls(
            ckpt_dir=ckpt_dir,
            state_json=os.path.join(ckpt_dir, "state.json"),
            archive_npz=os.path.join(ckpt_dir, "archive.npz"),
            scheduler_pkl=os.path.join(ckpt_dir, "scheduler.pkl"),
            best_solution=os.path.join(ckpt_dir, best_params_filename),
            phase_solution=os.path.join(ckpt_dir, phase_params_filename),
            reward_curve=os.path.join(ckpt_dir, reward_curve_filename),
            heatmap=os.path.join(ckpt_dir, "archive_heatmap.png"),
        )


def _serialize_state(state_path: str, cfg_dict: Dict[str, Any], obs_dim: int, act_dim: int, history: History) -> None:
    import json

    """Dump the minimal metadata required to reproduce training plots."""
    payload = {
        "config": cfg_dict,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "history": history.as_dict(),
    }
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    debug_log("Serialized state.json", path=state_path)


def _serialize_archive(archive_path: str, archive) -> None:
    """Persist the archive grid (solutions + metadata) as a NumPy .npz."""
    np.savez(archive_path, **archive.data())
    debug_log("Serialized archive grid", path=archive_path)


def _serialize_scheduler(scheduler_path: str, scheduler) -> None:
    """Persist the scheduler, which contains emitter / optimizer state."""
    with open(scheduler_path, "wb") as fh:
        pickle.dump(scheduler, fh)
    debug_log("Serialized scheduler state", path=scheduler_path)


def _serialize_solutions(paths: CheckpointPaths, best_solution: Optional[np.ndarray], phase_solution: Optional[np.ndarray]) -> None:
    """Save best solutions discovered so far (global + current phase)."""
    if best_solution is not None:
        np.save(paths.best_solution, best_solution)
        debug_log("Saved global best solution", path=paths.best_solution)
    if phase_solution is not None:
        np.save(paths.phase_solution, phase_solution)
        debug_log("Saved phase best solution", path=paths.phase_solution)


def save_archive_heatmap(
    archive,
    out_path: str,
    vmin: float,
    vmax: float,
    measure_labels: Optional[Tuple[str, str]] = None,
) -> None:
    """Render a MAP-Elites heatmap with fully configurable measure labels."""
    labels = measure_labels or ("Measure 1", "Measure 2")
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=vmin, vmax=vmax, cmap="viridis", transpose_measures=False)
    plt.gca().invert_yaxis()
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    debug_log("Archive heatmap saved", path=out_path, vmin=vmin, vmax=vmax, labels=labels)


def save_checkpoint_bundle(
    out_dir: str,
    phase_idx: int,
    iter_global: int,
    cfg_dict: Dict[str, Any],
    obs_dim: int,
    act_dim: int,
    history: History,
    archive,
    scheduler,
    best_solution: Optional[np.ndarray],
    phase_best_solution: Optional[np.ndarray],
    heatmap_bounds: Tuple[float, float],
    measure_labels: Optional[Tuple[str, str]] = None,
    best_params_filename: str = "best_solution.npy",
    phase_params_filename: str = "phase_best_solution.npy",
    reward_curve_filename: str = "reward_curve.png",
) -> str:
    """Persist all artifacts for the current training phase.

    Args:
        out_dir: Root directory where checkpoints are written.
        phase_idx: Current phase number (1-indexed).
        iter_global: Total training iterations completed so far.
        cfg_dict: Serialized configuration dataclass.
        obs_dim: Observation dimensionality of the controller.
        act_dim: Action dimensionality of the controller.
        history: Aggregated training metrics.
        archive: MAP-Elites archive instance to snapshot.
        scheduler: Scheduler / emitter state to pickle.
        best_solution: Best solution discovered throughout training.
        phase_best_solution: Best solution for the current phase.
        heatmap_bounds: (vmin, vmax) pair controlling the heatmap palette.
        measure_labels: Optional tuple describing archive axes; keeps API
            flexible in case the archive uses different measures in the future.
        best_params_filename: Filename used to persist the global best solution.
        phase_params_filename: Filename used for the per-phase best solution.
        reward_curve_filename: Filename used for the training curve image.
    """
    paths = CheckpointPaths.build(
        out_dir,
        phase_idx,
        iter_global,
        best_params_filename=best_params_filename,
        phase_params_filename=phase_params_filename,
        reward_curve_filename=reward_curve_filename,
    )
    debug_log(
        "Saving checkpoint bundle",
        phase=phase_idx,
        iteration=iter_global,
        out_dir=paths.ckpt_dir,
    )

    _serialize_state(paths.state_json, cfg_dict, obs_dim, act_dim, history)
    _serialize_archive(paths.archive_npz, archive)
    _serialize_scheduler(paths.scheduler_pkl, scheduler)
    _serialize_solutions(paths, best_solution, phase_best_solution)

    plot_training_curves(history, paths.reward_curve)
    debug_log("Saved training curve plot", path=paths.reward_curve)

    vmin, vmax = heatmap_bounds
    save_archive_heatmap(archive, paths.heatmap, vmin, vmax, measure_labels)

    return paths.ckpt_dir
