from __future__ import annotations

from dataclasses import asdict, dataclass

from uav_tracking.types import BBox, Detection, GroundTruth, Track


def intersection_over_union(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _greedy_matches(
    ground_truth: tuple[GroundTruth, ...],
    predictions: list[Detection] | list[Track],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            if gt.class_name != prediction.class_name:
                continue
            iou = intersection_over_union(gt.bbox, prediction.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, prediction_index))
    candidates.sort(reverse=True)
    matches: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    for _, gt_index, prediction_index in candidates:
        if gt_index not in used_gt and prediction_index not in used_predictions:
            used_gt.add(gt_index)
            used_predictions.add(prediction_index)
            matches.append((gt_index, prediction_index))
    return matches


@dataclass
class DetectionMetrics:
    ground_truth: int = 0
    predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def update(
        self,
        ground_truth: tuple[GroundTruth, ...],
        detections: list[Detection],
        iou_threshold: float = 0.5,
    ) -> None:
        matches = _greedy_matches(ground_truth, detections, iou_threshold)
        self.ground_truth += len(ground_truth)
        self.predictions += len(detections)
        self.true_positives += len(matches)
        self.false_positives += len(detections) - len(matches)
        self.false_negatives += len(ground_truth) - len(matches)

    def summary(self) -> dict[str, int | float]:
        precision_denominator = self.true_positives + self.false_positives
        recall_denominator = self.true_positives + self.false_negatives
        return {
            **asdict(self),
            "precision": (
                self.true_positives / precision_denominator if precision_denominator else 0.0
            ),
            "recall": self.true_positives / recall_denominator if recall_denominator else 0.0,
        }


@dataclass
class TrackingMetrics:
    matched_objects: int = 0
    id_switches: int = 0
    fragmentations: int = 0

    def __post_init__(self) -> None:
        self._last_track_by_target: dict[int, int] = {}
        self._was_matched: dict[int, bool] = {}
        self._ever_matched: set[int] = set()

    def update(
        self,
        ground_truth: tuple[GroundTruth, ...],
        tracks: list[Track],
        iou_threshold: float = 0.5,
    ) -> None:
        confirmed = [track for track in tracks if track.confirmed and track.time_since_update == 0]
        matches = _greedy_matches(ground_truth, confirmed, iou_threshold)
        matched_targets: set[int] = set()
        for gt_index, track_index in matches:
            target_id = ground_truth[gt_index].target_id
            track_id = confirmed[track_index].track_id
            matched_targets.add(target_id)
            previous = self._last_track_by_target.get(target_id)
            if previous is not None and previous != track_id:
                self.id_switches += 1
            if target_id in self._ever_matched and not self._was_matched.get(target_id, False):
                self.fragmentations += 1
            self._last_track_by_target[target_id] = track_id
            self._ever_matched.add(target_id)
            self.matched_objects += 1
        for gt in ground_truth:
            self._was_matched[gt.target_id] = gt.target_id in matched_targets

    def summary(self) -> dict[str, int]:
        return {
            "matched_objects": self.matched_objects,
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
        }
