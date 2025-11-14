"""Centralized configuration for Walker QD training jobs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .logging import debug_log


@dataclass(frozen=True)
class TrainConfig:
    """Immutable hyper-parameters used by CMA-ME training."""

    # Environment configuration
    env_id: str = "Walker2d-v5"
    seed: int = 42

    # Phase schedule
    iterations_per_phase: int = 100
    phases: int = 12

    # Parallelism
    workers: int = 8

    # CMA-ME emitters
    num_emitters: int = 12
    emitter_batch: int = 50
    sigma0: float = 0.5

    # Archive (MAP-Elites) ranges
    archive_bins: Tuple[int, int] = (128, 128)
    forward_velocity_range: Tuple[float, float] = (0.0, 10.0)
    torso_height_range: Tuple[float, float] = (0.8, 1.7)
    qd_score_offset: float = -500.0

    # Observation noise injection
    noise_std: float = 0.2

    # Output directories
    out_dir: str = "runs_mapelites_walker2d"

    # Video export
    render_fps: int = 30
    video_seconds: int = 20

    # Heatmap visualization
    heatmap_vmin: float = -500.0
    heatmap_vmax: float = 1500.0

    def __post_init__(self) -> None:
        debug_log(
            "TrainConfig instantiated",
            env=self.env_id,
            out_dir=self.out_dir,
            emitters=self.num_emitters,
        )
