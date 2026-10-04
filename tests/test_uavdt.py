from pathlib import Path

import pytest

from uav_tracking.data.uavdt import UAVDTSequence


def _dataset(tmp_path: Path, annotation: str) -> Path:
    image_dir = tmp_path / "UAVDT_Benchmark_M" / "M0101"
    image_dir.mkdir(parents=True)
    (image_dir / "img000001.jpg").touch()
    ground_truth = tmp_path / "UAV-benchmark-MOTD_v1.0" / "GT"
    ground_truth.mkdir(parents=True)
    (ground_truth / "M0101_gt_whole.txt").write_text(annotation, encoding="utf-8")
    return tmp_path


def test_parses_official_whole_annotation_fields(tmp_path: Path) -> None:
    root = _dataset(tmp_path, "1,7,10,20,30,40,1,2,3\n")
    sequence = UAVDTSequence(root, "M0101")

    obj = sequence.annotations[1][0]
    assert obj.target_id == 7
    assert obj.bbox == (10.0, 20.0, 40.0, 60.0)
    assert obj.class_name == "bus"
    assert obj.out_of_view == 1
    assert obj.occlusion == 2
    assert sequence.annotation_count == 1


def test_rejects_unknown_category(tmp_path: Path) -> None:
    root = _dataset(tmp_path, "1,7,10,20,30,40,1,2,9\n")
    with pytest.raises(ValueError, match="unknown UAVDT category"):
        UAVDTSequence(root, "M0101")


def test_rejects_malformed_row(tmp_path: Path) -> None:
    root = _dataset(tmp_path, "1,7,10,20\n")
    with pytest.raises(ValueError, match="expected 9"):
        UAVDTSequence(root, "M0101")
