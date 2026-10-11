from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uav_tracking.config import DetectorConfig
from uav_tracking.types import Detection


@dataclass(frozen=True)
class DetectorProfile:
    """Per-frame detector timings reported in milliseconds."""

    total: float
    preprocess: float
    inference: float
    postprocess: float


class YoloDetector:
    """Ultralytics YOLO adapter with explicit class-name filtering."""

    def __init__(self, config: DetectorConfig, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Install dependencies from requirements.txt."
            ) from error
        self.config = config
        self.device = device
        self.model = YOLO(config.model)
        names = self.model.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        requested = set(config.classes)
        self.class_ids = [class_id for class_id, name in self.names.items() if name in requested]
        missing = requested.difference(self.names.values())
        if missing:
            raise ValueError(
                f"Model '{config.model}' does not expose requested classes: {sorted(missing)}"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections, _ = self._detect(frame, profile=False)
        return detections

    def detect_profiled(self, frame: np.ndarray) -> tuple[list[Detection], DetectorProfile]:
        """Run detection and expose Ultralytics' synchronized CUDA timings."""
        detections, profile = self._detect(frame, profile=True)
        if profile is None:
            raise RuntimeError("Detector profiling did not produce timing data")
        return detections, profile

    def _detect(
        self, frame: np.ndarray, profile: bool
    ) -> tuple[list[Detection], DetectorProfile | None]:
        import time

        started = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )
        if not results:
            detector_profile = (
                DetectorProfile(
                    total=(time.perf_counter() - started) * 1000.0,
                    preprocess=0.0,
                    inference=0.0,
                    postprocess=0.0,
                )
                if profile
                else None
            )
            return [], detector_profile
        result = results[0]
        postprocess_started = time.perf_counter()
        boxes = result.boxes
        if boxes is None:
            detections: list[Detection] = []
        else:
            # These transfers synchronize the result tensors with the CPU. They
            # remain part of detection postprocessing in the profiling report.
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            detections = [
                Detection(
                    bbox=tuple(float(value) for value in box),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=self.names[int(class_id)],
                )
                for box, confidence, class_id in zip(
                    xyxy, confidences, classes, strict=True
                )
            ]
        adapter_postprocess = (time.perf_counter() - postprocess_started) * 1000.0
        if not profile:
            return detections, None

        total = (time.perf_counter() - started) * 1000.0
        speed = result.speed
        preprocess = float(speed.get("preprocess") or 0.0)
        inference = float(speed.get("inference") or 0.0)
        ultralytics_postprocess = float(speed.get("postprocess") or 0.0)
        # Ultralytics reports its NMS/postprocessing separately. Add the adapter's
        # result conversion and GPU-to-CPU transfers measured above.
        postprocess = ultralytics_postprocess + adapter_postprocess
        return detections, DetectorProfile(
            total=total,
            preprocess=preprocess,
            inference=inference,
            postprocess=postprocess,
        )

    @property
    def supported_names(self) -> Sequence[str]:
        return tuple(self.names.values())
