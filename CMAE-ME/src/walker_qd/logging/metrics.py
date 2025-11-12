"""Utility types and plots to track QD metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import debug_log


@dataclass
class History:
    """Container that accumulates scalar metrics during training."""

    obj_max: List[float] = field(default_factory=list)
    obj_mean: List[float] = field(default_factory=list)
    coverage: List[float] = field(default_factory=list)
    qd_score: List[float] = field(default_factory=list)
    norm_qd_score: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        """Return a plain dictionary for JSON serialization."""
        payload = {
            "obj_max": self.obj_max,
            "obj_mean": self.obj_mean,
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "norm_qd_score": self.norm_qd_score,
        }
        debug_log("History exported to dict", lengths={k: len(v) for k, v in payload.items()})
        return payload


def _normalize(series: List[float]) -> List[float]:
    """Scale a series to [0, 1] for plotting. Empty -> empty."""
    if not series:
        return []
    arr = np.array(series, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return [0.0 for _ in arr]
    return ((arr - mn) / (mx - mn)).tolist()


def plot_training_curves(history: History, out_path: str) -> None:
    """Plot reward/coverage curves and persist them to *out_path*."""
    fig, ax_obj = plt.subplots(figsize=(8, 5))
    (line_max,) = ax_obj.plot(history.obj_max, label="Max reward")
    (line_mean,) = ax_obj.plot(history.obj_mean, label="Mean reward")
    ax_obj.set_xlabel("Iteration")
    ax_obj.set_ylabel("Reward")

    ax_cov = ax_obj.twinx()
    (line_cov,) = ax_cov.plot(history.coverage, label="Coverage")
    norm_qd = _normalize(history.qd_score)
    line_norm = None
    if norm_qd:
        (line_norm,) = ax_cov.plot(norm_qd, label="QD score (norm)", linestyle="--")
    ax_cov.set_ylabel("Coverage / Normalized QD")
    ax_cov.set_ylim(0.0, 1.0)

    lines = [line_max, line_mean, line_cov] + ([line_norm] if line_norm is not None else [])
    labels = [line.get_label() for line in lines]
    ax_obj.legend(lines, labels, loc="upper left")
    ax_obj.set_title("Training")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    debug_log("Training curves saved", path=out_path, iterations=len(history.obj_max))
