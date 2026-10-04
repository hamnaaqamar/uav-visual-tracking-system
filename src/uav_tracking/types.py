from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruth:
    frame_index: int
    target_id: int
    bbox: BBox
    class_id: int
    class_name: str
    out_of_view: int
    occlusion: int


@dataclass(frozen=True)
class FrameData:
    image: np.ndarray
    frame_index: int
    sequence: str
    image_path: Path
    ground_truth: tuple[GroundTruth, ...]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    class_name: str
    confidence: float | None
    confirmed: bool
    age: int
    hits: int
    time_since_update: int
    state: str
