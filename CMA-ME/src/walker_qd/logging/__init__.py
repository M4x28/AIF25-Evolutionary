"""Utility helpers for simple debug logging.

A tiny logger is used instead of the stdlib logging module so that we can
sprinkle prints across the codebase without any configuration hassle.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

_DEBUG_ENABLED = os.getenv("WALKER_QD_DEBUG", "1").strip().lower() not in {"0", "false", "no"}


def debug_log(message: str, **context: Any) -> None:
    """Print a timestamped debug line if WALKER_QD_DEBUG is enabled."""
    if not _DEBUG_ENABLED:
        return
    context_bits = [f"{key}={value}" for key, value in context.items()]
    context_suffix = f" | {' '.join(context_bits)}" if context_bits else ""
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[DEBUG {timestamp}] {message}{context_suffix}")


__all__ = ["debug_log"]
