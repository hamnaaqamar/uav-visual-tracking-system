import pytest

from uav_tracking.profiling import StageProfiler


def test_profiler_excludes_warmup_and_uses_direct_end_to_end_latency() -> None:
    profiler = StageProfiler(warmup_frames=2)
    for frame, latency in enumerate((100.0, 80.0, 50.0, 25.0), start=1):
        profiler.record(frame, "end_to_end", latency)
        profiler.record(frame, "tracker", latency / 2)

    report = profiler.report(frames_processed=4, wall_clock_seconds=0.3)

    assert report["warmup_frames"] == 2
    assert report["measured_frames"] == 2
    assert report["stages_ms"]["end_to_end"]["count"] == 2
    assert report["stages_ms"]["end_to_end"]["mean"] == 37.5
    assert report["stages_ms"]["end_to_end"]["median"] == 37.5
    assert report["end_to_end_fps"] == pytest.approx(2 / 0.075)
    assert report["overall_fps"] == pytest.approx(4 / 0.3)


def test_profiler_reports_p95_for_each_stage() -> None:
    profiler = StageProfiler(warmup_frames=0)
    for frame, latency in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), start=1):
        profiler.record(frame, "rendering", latency)

    report = profiler.report(frames_processed=5, wall_clock_seconds=0.1)

    assert report["stages_ms"]["rendering"]["p95"] == pytest.approx(4.8)
