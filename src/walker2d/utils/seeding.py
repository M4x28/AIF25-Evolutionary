"""Randomness helpers to make experiments reproducible."""
import random

import numpy as np

from ..logging import debug_log


def seed_all(seed: int) -> None:
    """Seed numpy and the Python stdlib RNG from a single entry point."""
    np.random.seed(seed)
    random.seed(seed)
    debug_log("Global RNG seeded", seed=seed)
