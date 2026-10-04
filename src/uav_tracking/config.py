from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    sequence: str
    start_frame: int = 1
    end_frame: int | None = None


@dataclass(frozen=True)
class DetectorConfig:
    model: str
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    classes: tuple[str, ...]


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int
    n_init: int
    max_cosine_distance: float
    embedder: str
    embedder_gpu: bool
    half: bool


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    save_video: bool
    video_fps: float
    save_sample_frames: int


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    log_level: str


@dataclass(frozen=True)
class ExperimentConfig:
    suppress_detections_start: int | None
    suppress_detections_end: int | None

    def suppresses(self, frame_index: int) -> bool:
        return (
            self.suppress_detections_start is not None
            and self.suppress_detections_end is not None
            and self.suppress_detections_start <= frame_index <= self.suppress_detections_end
        )


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    output: OutputConfig
    runtime: RuntimeConfig
    experiment: ExperimentConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' is required and must be a mapping")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    dataset = _section(raw, "dataset")
    detector = _section(raw, "detector")
    tracker = _section(raw, "tracker")
    output = _section(raw, "output")
    runtime = _section(raw, "runtime")
    experiment = _section(raw, "experiment")

    dataset_root = Path(str(dataset["root"])).expanduser()
    output_dir = Path(str(output["directory"])).expanduser()
    # Relative paths are resolved from the process working directory, making
    # command-line overrides predictable rather than dependent on config location.
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()

    config = AppConfig(
        dataset=DatasetConfig(
            root=dataset_root,
            sequence=str(dataset["sequence"]),
            start_frame=int(dataset.get("start_frame", 1)),
            end_frame=int(dataset["end_frame"]) if dataset.get("end_frame") is not None else None,
        ),
        detector=DetectorConfig(
            model=str(detector["model"]),
            confidence_threshold=float(detector["confidence_threshold"]),
            iou_threshold=float(detector["iou_threshold"]),
            image_size=int(detector["image_size"]),
            classes=tuple(str(item) for item in detector["classes"]),
        ),
        tracker=TrackerConfig(
            max_age=int(tracker["max_age"]),
            n_init=int(tracker["n_init"]),
            max_cosine_distance=float(tracker["max_cosine_distance"]),
            embedder=str(tracker["embedder"]),
            embedder_gpu=bool(tracker["embedder_gpu"]),
            half=bool(tracker["half"]),
        ),
        output=OutputConfig(
            directory=output_dir,
            save_video=bool(output["save_video"]),
            video_fps=float(output["video_fps"]),
            save_sample_frames=int(output["save_sample_frames"]),
        ),
        runtime=RuntimeConfig(device=str(runtime["device"]), log_level=str(runtime["log_level"])),
        experiment=ExperimentConfig(
            suppress_detections_start=(
                int(experiment["suppress_detections_start"])
                if experiment.get("suppress_detections_start") is not None
                else None
            ),
            suppress_detections_end=(
                int(experiment["suppress_detections_end"])
                if experiment.get("suppress_detections_end") is not None
                else None
            ),
        ),
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.dataset.start_frame < 1:
        raise ValueError("dataset.start_frame must be at least 1")
    if (
        config.dataset.end_frame is not None
        and config.dataset.end_frame < config.dataset.start_frame
    ):
        raise ValueError("dataset.end_frame must not precede dataset.start_frame")
    if not 0.0 <= config.detector.confidence_threshold <= 1.0:
        raise ValueError("detector.confidence_threshold must be between 0 and 1")
    if not 0.0 <= config.detector.iou_threshold <= 1.0:
        raise ValueError("detector.iou_threshold must be between 0 and 1")
    if config.tracker.max_age < 1 or config.tracker.n_init < 1:
        raise ValueError("tracker.max_age and tracker.n_init must be positive")
    starts = config.experiment.suppress_detections_start
    ends = config.experiment.suppress_detections_end
    if (starts is None) != (ends is None):
        raise ValueError("Both detection suppression boundaries must be set together")
    if starts is not None and (starts < 1 or ends is None or ends < starts):
        raise ValueError("Detection suppression frame range is invalid")
