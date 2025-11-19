"""Simple policy parameterization used during evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging import debug_log


@dataclass(frozen=True)
class LinearTanhPolicy:
    """Linear policy followed by tanh squashing."""

    obs_dim: int
    act_dim: int

    def __post_init__(self) -> None:
        debug_log("LinearTanhPolicy constructed", obs_dim=self.obs_dim, act_dim=self.act_dim)

    def weights_from_vector(self, vec: np.ndarray) -> np.ndarray:
        """Reshape a flat vector into the expected (act_dim, obs_dim) matrix."""
        weights = vec.reshape((self.act_dim, self.obs_dim))
        debug_log("Policy weights reshaped", obs_dim=self.obs_dim, act_dim=self.act_dim)
        return weights

    def act(self, weights: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Propagate an observation through the linear layer and tanh."""
        raw = weights @ obs
        action = np.tanh(raw)
        return action
