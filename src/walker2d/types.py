"""Common type aliases shared across modules."""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np

Reward = float
Velocity = float
Height = float
Objective = float
Measure = Tuple[float, float]


class EvaluateFn(Protocol):
    """Protocol describing the evaluator callable used by schedulers."""

    def __call__(
        self,
        model: np.ndarray,
        noise_vec: Optional[np.ndarray],
    ) -> Tuple[Reward, Velocity, Height]: ...
