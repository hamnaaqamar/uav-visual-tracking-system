from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StageProfiler:
    """Collect direct wall-clock stage timings after an initial warm-up."""

    warmup_frames: int
    _stages: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, processed_frame: int, stage: str, milliseconds: float) -> None:
        if processed_frame > self.warmup_frames:
            self._stages[stage].append(milliseconds)

    def report(
        self,
        *,
        frames_processed: int,
        wall_clock_seconds: float,
    ) -> dict[str, Any]:
        measured_frames = max(0, frames_processed - self.warmup_frames)
        end_to_end = self._stages.get("end_to_end", [])
        measured_seconds = sum(end_to_end) / 1000.0
        return {
            "enabled": True,
            "warmup_frames": min(self.warmup_frames, frames_processed),
            "measured_frames": measured_frames,
            "frames_processed": frames_processed,
            "total_wall_clock_seconds": wall_clock_seconds,
            "end_to_end_fps": (
                measured_frames / measured_seconds if measured_seconds > 0.0 else 0.0
            ),
            "overall_fps": (
                frames_processed / wall_clock_seconds if wall_clock_seconds > 0.0 else 0.0
            ),
            "stages_ms": {
                name: _distribution(values) for name, values in sorted(self._stages.items())
            },
            "notes": {
                "end_to_end_fps": (
                    "Measured directly from post-warm-up per-frame end-to-end latency; "
                    "stage FPS values are not added."
                ),
                "detector_total": (
                    "YOLO preprocessing, inference, NMS/postprocessing, result conversion, "
                    "and CPU transfer."
                ),
                "pipeline": (
                    "Legacy compute-path timer retained for compatibility; excludes frame "
                    "read/decode, rendering, video encoding, and sample image writes."
                ),
                "wall_clock": (
                    "Covers all frame reads and processing plus final video-writer release; "
                    "excludes dataset, model, and tracker construction."
                ),
            },
        }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    data = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }
