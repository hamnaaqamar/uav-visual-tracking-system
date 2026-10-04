from __future__ import annotations

import numpy as np

from uav_tracking.config import TrackerConfig
from uav_tracking.types import Detection, Track


class DeepSortTracker:
    """Small adapter around deep-sort-realtime, preserving lifecycle fields."""

    def __init__(self, config: TrackerConfig) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as error:
            raise RuntimeError(
                "deep-sort-realtime is not installed. Install requirements.txt."
            ) from error
        self._tracker = DeepSort(
            max_age=config.max_age,
            n_init=config.n_init,
            max_cosine_distance=config.max_cosine_distance,
            embedder=config.embedder,
            embedder_gpu=config.embedder_gpu,
            half=config.half,
            bgr=True,
        )

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        raw = []
        for detection in detections:
            left, top, right, bottom = detection.bbox
            raw.append(
                (
                    [left, top, right - left, bottom - top],
                    detection.confidence,
                    detection.class_name,
                )
            )
        active = self._tracker.update_tracks(raw, frame=frame)
        output: list[Track] = []
        for track in active:
            bbox = track.to_ltrb(orig=True, orig_strict=True)
            if bbox is None:
                bbox = track.to_ltrb()
            confidence = getattr(track, "det_conf", None)
            class_name = getattr(track, "det_class", None) or "unknown"
            confirmed = bool(track.is_confirmed())
            time_since_update = int(track.time_since_update)
            state = "tentative"
            if confirmed:
                state = "predicted" if time_since_update > 0 else "confirmed"
            output.append(
                Track(
                    track_id=int(track.track_id),
                    bbox=tuple(float(value) for value in bbox),
                    class_name=str(class_name),
                    confidence=float(confidence) if confidence is not None else None,
                    confirmed=confirmed,
                    age=int(getattr(track, "age", 0)),
                    hits=int(getattr(track, "hits", 0)),
                    time_since_update=time_since_update,
                    state=state,
                )
            )
        return output
