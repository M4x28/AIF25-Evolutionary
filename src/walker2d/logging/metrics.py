"""Utility types and plots to track QD metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

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


def _resolve_series(source: History | Dict[str, List[float]], entries: Iterable[Tuple[object, str]]) -> List[Tuple[List[float], str]]:
    resolved = []
    for key_or_series, label in entries:
        if isinstance(key_or_series, str):
            if isinstance(source, dict):
                values = source[key_or_series]
            else:
                values = getattr(source, key_or_series)
        else:
            values = key_or_series
        resolved.append((values, label))
    return resolved


def plot_training_curves(
    source: History | Dict[str, List[float]],
    out_path: str,
    *,
    series: Optional[Iterable[Tuple[object, str]]] = None,
    right_series: Optional[Iterable[Tuple[object, str]]] = None,
    x_label: str = "Iteration",
    left_label: str = "Reward",
    right_label: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
    right_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Single plotting utility used by every algorithm."""
    if series is None:
        if not isinstance(source, History):
            raise ValueError("Series must be provided when plotting custom metrics.")
        left_series = [(source.obj_max, "Max reward"), (source.obj_mean, "Mean reward")]
        right_series = [(source.coverage, "Coverage")]
        norm_qd = _normalize(source.qd_score)
        if norm_qd:
            right_series.append((norm_qd, "QD score (norm)"))
        x_label = "Iteration"
        left_label = "Reward"
        right_label = "Coverage / Normalized QD"
        title = title or "Training"
        figsize = (8, 5)
        right_ylim = (0.0, 1.0)
    else:
        left_series = _resolve_series(source, series)
        right_series = _resolve_series(source, right_series) if right_series is not None else None

    fig, ax_left = plt.subplots(figsize=figsize)
    for values, label in left_series:
        ax_left.plot(values, label=label)
    ax_left.set_xlabel(x_label)
    ax_left.set_ylabel(left_label)

    lines = list(ax_left.lines)
    if right_series:
        ax_right = ax_left.twinx()
        for values, label in right_series:
            ax_right.plot(values, label=label, linestyle="--")
        if right_label:
            ax_right.set_ylabel(right_label)
        if right_ylim is not None:
            ax_right.set_ylim(*right_ylim)
        lines += ax_right.lines

    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="upper left")
    if title:
        ax_left.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    debug_log("Plot saved", path=out_path, title=title)

