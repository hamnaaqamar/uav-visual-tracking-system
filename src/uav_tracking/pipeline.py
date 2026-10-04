from __future__ import annotations

import csv
import json
import logging
import platform
import statistics
import time
from importlib import metadata
from typing import Any

import numpy as np

from uav_tracking.config import AppConfig
from uav_tracking.data.uavdt import UAVDTSequence
from uav_tracking.detection.yolo import YoloDetector
from uav_tracking.evaluation.metrics import DetectionMetrics, TrackingMetrics
from uav_tracking.tracking.deep_sort import DeepSortTracker
from uav_tracking.visualization.drawing import draw_tracks

LOGGER = logging.getLogger(__name__)


class TrackingPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        import cv2

        sequence = UAVDTSequence(
            self.config.dataset.root,
            self.config.dataset.sequence,
            self.config.dataset.start_frame,
            self.config.dataset.end_frame,
        )
        detector = YoloDetector(self.config.detector, self.config.runtime.device)
        tracker = DeepSortTracker(self.config.tracker)
        output = self.config.output.directory
        output.mkdir(parents=True, exist_ok=True)

        detection_metrics = DetectionMetrics()
        tracking_metrics = TrackingMetrics()
        detector_ms: list[float] = []
        tracker_ms: list[float] = []
        pipeline_ms: list[float] = []
        confidences: list[float] = []
        active_ids: set[int] = set()
        deletion_events: list[dict[str, int]] = []
        pre_outage_ids: set[int] | None = None
        end_outage_ids: set[int] | None = None
        first_post_outage: dict[str, Any] | None = None
        video_writer = None

        with (
            (output / "detections.csv").open("w", newline="", encoding="utf-8") as detection_file,
            (output / "tracks.csv").open("w", newline="", encoding="utf-8") as track_file,
        ):
            detection_log = csv.writer(detection_file)
            track_log = csv.writer(track_file)
            detection_log.writerow(
                ["frame", "class", "confidence", "x1", "y1", "x2", "y2", "suppressed"]
            )
            track_log.writerow(
                [
                    "frame",
                    "track_id",
                    "class",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "state",
                    "confirmed",
                    "age",
                    "hits",
                    "time_since_update",
                ]
            )

            for processed, frame in enumerate(sequence, start=1):
                started = time.perf_counter()
                detection_started = time.perf_counter()
                detections = detector.detect(frame.image)
                detector_ms.append((time.perf_counter() - detection_started) * 1000.0)
                suppressed = self.config.experiment.suppresses(frame.frame_index)
                if suppressed:
                    detections = []
                confidences.extend(detection.confidence for detection in detections)
                for detection in detections:
                    detection_log.writerow(
                        [
                            frame.frame_index,
                            detection.class_name,
                            detection.confidence,
                            *detection.bbox,
                            suppressed,
                        ]
                    )
                detection_metrics.update(frame.ground_truth, detections)

                tracking_started = time.perf_counter()
                tracks = tracker.update(detections, frame.image)
                tracker_ms.append((time.perf_counter() - tracking_started) * 1000.0)
                tracking_metrics.update(frame.ground_truth, tracks)
                current_ids = {track.track_id for track in tracks}
                confirmed_ids = {track.track_id for track in tracks if track.confirmed}
                suppression_start = self.config.experiment.suppress_detections_start
                suppression_end = self.config.experiment.suppress_detections_end
                if suppression_start is not None and frame.frame_index == suppression_start - 1:
                    pre_outage_ids = confirmed_ids
                if suppression_end is not None and frame.frame_index == suppression_end:
                    end_outage_ids = confirmed_ids
                if (
                    suppression_end is not None
                    and frame.frame_index > suppression_end
                    and first_post_outage is None
                    and confirmed_ids
                ):
                    first_post_outage = {
                        "frame": frame.frame_index,
                        "confirmed_track_ids": sorted(confirmed_ids),
                    }
                for deleted_id in sorted(active_ids.difference(current_ids)):
                    deletion_events.append({"frame": frame.frame_index, "track_id": deleted_id})
                active_ids = current_ids
                for track in tracks:
                    track_log.writerow(
                        [
                            frame.frame_index,
                            track.track_id,
                            track.class_name,
                            track.confidence,
                            *track.bbox,
                            track.state,
                            track.confirmed,
                            track.age,
                            track.hits,
                            track.time_since_update,
                        ]
                    )

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                pipeline_ms.append(elapsed_ms)
                instantaneous_fps = 1000.0 / elapsed_ms if elapsed_ms else None
                rendered = draw_tracks(frame.image, tracks, frame.frame_index, instantaneous_fps)
                if self.config.output.save_video:
                    if video_writer is None:
                        height, width = rendered.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(output / f"{sequence.sequence}_tracked.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            self.config.output.video_fps,
                            (width, height),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError("Could not initialize MP4 output writer")
                    video_writer.write(rendered)
                if processed <= self.config.output.save_sample_frames:
                    cv2.imwrite(str(output / f"sample_{frame.frame_index:06d}.jpg"), rendered)
                if processed % 100 == 0:
                    LOGGER.info("Processed %d/%d frames", processed, len(sequence))

        if video_writer is not None:
            video_writer.release()
        summary = {
            "status": "completed",
            "sequence": sequence.sequence,
            "frames": len(sequence),
            "configuration": _serializable_config(self.config),
            "environment": _environment(),
            "detection": {
                **detection_metrics.summary(),
                "confidence": _distribution(confidences),
            },
            "tracking": {
                **tracking_metrics.summary(),
                "deletion_events": deletion_events,
            },
            "timing_ms": {
                "detector": _distribution(detector_ms),
                "tracker": _distribution(tracker_ms),
                "pipeline": _distribution(pipeline_ms),
                "mean_fps": 1000.0 / statistics.mean(pipeline_ms) if pipeline_ms else 0.0,
            },
            "controlled_failure": {
                "enabled": self.config.experiment.suppress_detections_start is not None,
                "suppression_start": self.config.experiment.suppress_detections_start,
                "suppression_end": self.config.experiment.suppress_detections_end,
                "confirmed_ids_before_outage": sorted(pre_outage_ids or ()),
                "confirmed_ids_at_outage_end": sorted(end_outage_ids or ()),
                "ids_surviving_outage": sorted(
                    (pre_outage_ids or set()) & (end_outage_ids or set())
                ),
                "deletions_during_outage": _deletions_in_range(
                    deletion_events,
                    self.config.experiment.suppress_detections_start,
                    self.config.experiment.suppress_detections_end,
                ),
                "first_confirmed_post_outage": first_post_outage,
                "note": (
                    "This reports aggregate track continuity around a controlled detector outage. "
                    "Use ground-truth matching in tracks.csv for target-specific "
                    "identity analysis. "
                    "No re-detection mechanism is implemented."
                ),
            },
        }
        with (output / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        LOGGER.info("Experiment summary written to %s", output / "summary.json")
        return summary


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def _serializable_config(config: AppConfig) -> dict[str, Any]:
    from dataclasses import asdict

    result = asdict(config)
    result["dataset"]["root"] = str(config.dataset.root)
    result["output"]["directory"] = str(config.output.directory)
    return result


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("ultralytics", "deep-sort-realtime", "torch", "opencv-python"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu: dict[str, Any] = {"cuda_available": False, "name": None}
    try:
        import torch

        gpu["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu["name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "hardware": gpu,
    }


def _deletions_in_range(
    events: list[dict[str, int]],
    start: int | None,
    end: int | None,
) -> list[dict[str, int]]:
    if start is None or end is None:
        return []
    return [event for event in events if start <= event["frame"] <= end + 1]
