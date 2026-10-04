from __future__ import annotations

import colorsys

import numpy as np

from uav_tracking.types import Detection, GroundTruth, Track


def _color(identifier: int) -> tuple[int, int, int]:
    red, green, blue = colorsys.hsv_to_rgb((identifier * 0.61803398875) % 1.0, 0.75, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_ground_truth(image: np.ndarray, objects: tuple[GroundTruth, ...]) -> np.ndarray:
    import cv2

    output = image.copy()
    for obj in objects:
        left, top, right, bottom = (int(value) for value in obj.bbox)
        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"GT {obj.class_name} #{obj.target_id}",
            (left, max(15, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    import cv2

    output = image.copy()
    for detection in detections:
        left, top, right, bottom = (int(value) for value in detection.bbox)
        cv2.rectangle(output, (left, top), (right, bottom), (0, 180, 255), 2)
        cv2.putText(
            output,
            f"{detection.class_name} {detection.confidence:.2f}",
            (left, max(15, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def draw_tracks(
    image: np.ndarray,
    tracks: list[Track],
    frame_index: int,
    fps: float | None = None,
) -> np.ndarray:
    import cv2

    output = image.copy()
    for track in tracks:
        if not track.confirmed:
            continue
        left, top, right, bottom = (int(value) for value in track.bbox)
        color = _color(track.track_id)
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        confidence = f" {track.confidence:.2f}" if track.confidence is not None else ""
        label = f"{track.class_name} #{track.track_id}{confidence} [{track.state}]"
        cv2.putText(
            output,
            label,
            (left, max(15, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    status = f"frame {frame_index}"
    if fps is not None:
        status += f" | {fps:.1f} FPS"
    cv2.putText(
        output,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
