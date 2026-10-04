from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from uav_tracking.config import DetectorConfig
from uav_tracking.types import Detection


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
            return []
        boxes = results[0].boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        return [
            Detection(
                bbox=tuple(float(value) for value in box),
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=self.names[int(class_id)],
            )
            for box, confidence, class_id in zip(xyxy, confidences, classes, strict=True)
        ]

    @property
    def supported_names(self) -> Sequence[str]:
        return tuple(self.names.values())
