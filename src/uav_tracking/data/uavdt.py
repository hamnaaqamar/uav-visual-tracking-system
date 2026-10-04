from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from uav_tracking.types import FrameData, GroundTruth

UAVDT_CLASSES = {1: "car", 2: "truck", 3: "bus"}
_IMAGE_ROOT_NAMES = ("UAVDT_Benchmark_M", "UAV-benchmark-M")
_GT_ROOT_NAMES = ("UAV-benchmark-MOTD_v1.0/GT", "GT")


class UAVDTSequence:
    """Reader for one sequence in the original UAVDT DET/MOT release."""

    def __init__(
        self,
        root: str | Path,
        sequence: str,
        start_frame: int = 1,
        end_frame: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sequence = sequence
        self.start_frame = start_frame
        self.end_frame = end_frame
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"UAVDT root does not exist: {self.root}. "
                "See data/README.md for the expected layout."
            )
        self.image_dir = self._find_image_dir()
        self.annotation_path = self._find_annotation()
        self.annotations = self._read_annotations()
        self.image_paths = self._find_images()

    def _find_image_dir(self) -> Path:
        candidates = [self.root / name / self.sequence for name in _IMAGE_ROOT_NAMES]
        # A sequence directory may also be supplied as root for small subsets.
        candidates.append(self.root / self.sequence)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Sequence '{self.sequence}' not found. Checked: {checked}")

    def _find_annotation(self) -> Path:
        filename = f"{self.sequence}_gt_whole.txt"
        candidates = [self.root / name / filename for name in _GT_ROOT_NAMES]
        candidates.append(self.root / filename)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"UAVDT whole ground truth not found. Checked: {checked}")

    def _read_annotations(self) -> dict[int, tuple[GroundTruth, ...]]:
        grouped: dict[int, list[GroundTruth]] = defaultdict(list)
        with self.annotation_path.open(encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != 9:
                    raise ValueError(
                        f"{self.annotation_path}:{line_number}: expected 9 comma-separated "
                        f"UAVDT fields, received {len(row)}"
                    )
                try:
                    frame, target, left, top, width, height, out, occ, category = (
                        int(value.strip()) for value in row
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{self.annotation_path}:{line_number}: all UAVDT fields must be integers"
                    ) from error
                if frame < 1 or target < 1 or width <= 0 or height <= 0:
                    raise ValueError(
                        f"{self.annotation_path}:{line_number}: invalid frame, target, or box size"
                    )
                if category not in UAVDT_CLASSES:
                    raise ValueError(
                        f"{self.annotation_path}:{line_number}: unknown UAVDT category {category}; "
                        f"expected one of {sorted(UAVDT_CLASSES)}"
                    )
                grouped[frame].append(
                    GroundTruth(
                        frame_index=frame,
                        target_id=target,
                        bbox=(float(left), float(top), float(left + width), float(top + height)),
                        class_id=category,
                        class_name=UAVDT_CLASSES[category],
                        out_of_view=out,
                        occlusion=occ,
                    )
                )
        return {frame: tuple(objects) for frame, objects in grouped.items()}

    def _find_images(self) -> tuple[Path, ...]:
        images: list[tuple[int, Path]] = []
        for path in self.image_dir.glob("img*.jpg"):
            digits = path.stem.removeprefix("img")
            if not digits.isdigit():
                continue
            frame = int(digits)
            if frame < self.start_frame:
                continue
            if self.end_frame is not None and frame > self.end_frame:
                continue
            images.append((frame, path))
        images.sort(key=lambda item: item[0])
        if not images:
            raise FileNotFoundError(
                f"No img######.jpg frames in requested range under {self.image_dir}"
            )
        return tuple(path for _, path in images)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[FrameData]:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to decode UAVDT frames") from error
        for image_path in self.image_paths:
            frame_index = int(image_path.stem.removeprefix("img"))
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"OpenCV could not decode frame: {image_path}")
            yield FrameData(
                image=image,
                frame_index=frame_index,
                sequence=self.sequence,
                image_path=image_path,
                ground_truth=self.annotations.get(frame_index, ()),
            )

    @property
    def annotation_count(self) -> int:
        return sum(len(objects) for objects in self.annotations.values())
