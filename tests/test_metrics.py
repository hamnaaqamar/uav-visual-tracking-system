from uav_tracking.evaluation.metrics import (
    DetectionMetrics,
    TrackingMetrics,
    intersection_over_union,
)
from uav_tracking.types import Detection, GroundTruth, Track


def _ground_truth(target_id: int = 1) -> GroundTruth:
    return GroundTruth(1, target_id, (0.0, 0.0, 10.0, 10.0), 1, "car", 1, 1)


def _track(track_id: int) -> Track:
    return Track(track_id, (0.0, 0.0, 10.0, 10.0), "car", 0.9, True, 3, 3, 0, "confirmed")


def test_iou() -> None:
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_detection_metrics_separate_misses_and_false_positives() -> None:
    metrics = DetectionMetrics()
    metrics.update(
        (_ground_truth(),),
        [
            Detection((0, 0, 10, 10), 0.9, 2, "car"),
            Detection((20, 20, 30, 30), 0.8, 2, "car"),
        ],
    )
    assert metrics.summary()["true_positives"] == 1
    assert metrics.summary()["false_positives"] == 1
    assert metrics.summary()["recall"] == 1.0


def test_tracking_metrics_count_id_switch_after_identity_change() -> None:
    metrics = TrackingMetrics()
    metrics.update((_ground_truth(),), [_track(10)])
    metrics.update((_ground_truth(),), [_track(11)])
    assert metrics.summary()["id_switches"] == 1


def test_tracking_metrics_count_fragment_after_unmatched_frame() -> None:
    metrics = TrackingMetrics()
    metrics.update((_ground_truth(),), [_track(10)])
    metrics.update((_ground_truth(),), [])
    metrics.update((_ground_truth(),), [_track(10)])
    assert metrics.summary()["fragmentations"] == 1
