"""Abstractions shared by algorithm implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..logging.metrics import History


@dataclass
class TrainerResult:
    """Bundle returned by any trainer implementation."""

    best_solution: Optional[np.ndarray]
    total_iterations: int
    history: History


class TrainerStrategy:
    """Interface implemented by every training strategy."""

    def train(self) -> TrainerResult:  # pragma: no cover - interface only
        raise NotImplementedError
