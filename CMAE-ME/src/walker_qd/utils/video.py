"""Video helpers used when exporting rollouts."""
from __future__ import annotations

from typing import List

import numpy as np

from ..logging import debug_log

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def write_rgb_video(frames: List[np.ndarray], out_path: str, fps: int) -> None:
    """Serialize RGB frames to disk using OpenCV if it is available."""
    if not frames:
        debug_log("Skipping video export because no frames were produced", path=out_path)
        return
    if cv2 is None:
        debug_log("OpenCV missing, cannot write video", path=out_path)
        return

    h, w, _ = frames[0].shape
    debug_log("Writing RGB video", path=out_path, fps=fps, frames=len(frames), resolution=f"{w}x{h}")
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
