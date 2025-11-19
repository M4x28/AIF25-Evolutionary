"""Lightweight IO helpers shared across the training pipeline."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from ..logging import debug_log


def ensure_dir(path: str) -> None:
    """Create *path* (recursively) and emit a debug line so checkpoints are traceable."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)
    debug_log("Ensured directory exists", path=path)


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Persist dictionaries to disk with indentation for easy inspection."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    debug_log("Saved JSON payload", path=path, keys=list(data.keys()))
