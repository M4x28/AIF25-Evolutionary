"""Checkpoint helpers for persisting training progress."""
from __future__ import annotations

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ribs.visualize import grid_archive_heatmap

from ..logging import debug_log
from ..logging.metrics import History, plot_training_curves
from ..utils.io import ensure_dir

def save_archive_heatmap(archive, out_path: str, vmin: float, vmax: float) -> None:
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(
        archive, vmin=vmin, vmax=vmax, cmap="viridis", transpose_measures=False
    )
    plt.gca().invert_yaxis()
    plt.ylabel("Torso height")
    plt.xlabel("Forward velocity")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    debug_log("Archive heatmap saved", path=out_path, vmin=vmin, vmax=vmax)

def save_checkpoint_bundle(
    out_dir: str,
    phase_idx: int,
    iter_global: int,
    cfg_dict,
    obs_dim: int,
    act_dim: int,
    history: History,
    archive,
    scheduler,
    best_solution: np.ndarray | None,
    phase_best_solution: np.ndarray | None,
    heatmap_bounds: tuple[float, float],
) -> str:
    ckpt_dir = os.path.join(out_dir, f"checkpoint_phase{phase_idx:02d}_iter{iter_global:04d}")
    ensure_dir(ckpt_dir)
    debug_log(
        "Saving checkpoint bundle",
        phase=phase_idx,
        iteration=iter_global,
        out_dir=ckpt_dir,
    )

    # State JSON with the minimal metadata required to resume analysis.
    import json

    state = {
        "config": cfg_dict,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "history": history.as_dict(),
    }
    with open(os.path.join(ckpt_dir, "state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    debug_log("Serialized state.json", path=os.path.join(ckpt_dir, "state.json"))

    # Archive (npz)
    archive_path = os.path.join(ckpt_dir, "archive.npz")
    np.savez(archive_path, **archive.data())
    debug_log("Serialized archive grid", path=archive_path)

    # Scheduler (pickle)
    scheduler_path = os.path.join(ckpt_dir, "scheduler.pkl")
    with open(scheduler_path, "wb") as fh:
        pickle.dump(scheduler, fh)
    debug_log("Serialized scheduler state", path=scheduler_path)

    # Best solutions per phase / global
    if best_solution is not None:
        path_best = os.path.join(ckpt_dir, "best_solution.npy")
        np.save(path_best, best_solution)
        debug_log("Saved global best solution", path=path_best)
    if phase_best_solution is not None:
        path_phase = os.path.join(ckpt_dir, "phase_best_solution.npy")
        np.save(path_phase, phase_best_solution)
        debug_log("Saved phase best solution", path=path_phase)

    # Curves and heatmaps act as quick diagnostics.
    reward_curve_path = os.path.join(ckpt_dir, "reward_curve.png")
    plot_training_curves(history, reward_curve_path)
    debug_log("Saved training curve plot", path=reward_curve_path)

    vmin, vmax = heatmap_bounds
    heatmap_path = os.path.join(ckpt_dir, "archive_heatmap.png")
    save_archive_heatmap(archive, heatmap_path, vmin, vmax)

    return ckpt_dir
