"""Randomness helpers to make experiments reproducible."""
import numpy as np

from ..logging import debug_log


def seed_all(seed: int) -> None:
    """Seed numpy (and future RNG backends) from a single entry point."""
    np.random.seed(seed)
    debug_log("Global RNG seeded", seed=seed)
