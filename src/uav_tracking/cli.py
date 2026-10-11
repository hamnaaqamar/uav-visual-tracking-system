from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from uav_tracking.config import AppConfig, load_config
from uav_tracking.data.uavdt import UAVDTSequence
from uav_tracking.pipeline import TrackingPipeline
from uav_tracking.visualization.drawing import draw_ground_truth


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UAVDT YOLO + DeepSORT baseline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--sequence")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--validate-dataset", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect post-warm-up stage timings and write profiling.json",
    )
    parser.add_argument(
        "--profile-warmup-frames",
        type=int,
        default=30,
        help="Frames processed normally but excluded from profiling statistics (default: 30)",
    )
    return parser


def _overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    dataset = replace(
        config.dataset,
        root=args.dataset_root.expanduser().resolve() if args.dataset_root else config.dataset.root,
        sequence=args.sequence or config.dataset.sequence,
        start_frame=args.start_frame or config.dataset.start_frame,
        end_frame=args.end_frame if args.end_frame is not None else config.dataset.end_frame,
    )
    output = replace(
        config.output,
        directory=args.output.expanduser().resolve() if args.output else config.output.directory,
    )
    runtime = replace(config.runtime, device=args.device or config.runtime.device)
    return replace(config, dataset=dataset, output=output, runtime=runtime)


def validate_dataset(config: AppConfig) -> dict[str, object]:
    import cv2

    sequence = UAVDTSequence(
        config.dataset.root,
        config.dataset.sequence,
        config.dataset.start_frame,
        config.dataset.end_frame,
    )
    sample_dir = config.output.directory / "dataset_validation"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, frame in enumerate(sequence):
        if index >= config.output.save_sample_frames:
            break
        rendered = draw_ground_truth(frame.image, frame.ground_truth)
        path = sample_dir / f"gt_{frame.frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), rendered):
            raise RuntimeError(f"Failed to write validation image: {path}")
        samples.append(str(path))
    result = {
        "sequence": sequence.sequence,
        "frames": len(sequence),
        "annotations": sequence.annotation_count,
        "image_directory": str(sequence.image_dir),
        "annotation_file": str(sequence.annotation_path),
        "sample_visualizations": samples,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = _parser().parse_args()
    config = _overrides(load_config(args.config), args)
    logging.basicConfig(
        level=getattr(logging, config.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.validate_dataset:
        validate_dataset(config)
        return
    summary = TrackingPipeline(
        config,
        profile=args.profile,
        warmup_frames=args.profile_warmup_frames,
    ).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
