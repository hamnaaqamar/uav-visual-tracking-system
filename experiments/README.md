# Experiments

Generated logs, videos, and model weights are excluded from Git. Record a
concise report for each retained experiment with:

- date, commit, command, and configuration
- hardware and package versions (also captured in `summary.json`)
- UAVDT sequence and frame range
- detection precision/recall at IoU 0.5
- ID switches and fragmentation count
- detector, tracker, and end-to-end latency
- detector misses versus association/track-management failures

For a controlled detector-loss experiment, copy the baseline configuration and
set `experiment.suppress_detections_start` and
`experiment.suppress_detections_end`. Compare `tracks.csv` before, during, and
after the interval, and use the deletion events in `summary.json`. This tests
DeepSORT's existing lifecycle behavior; it does not add re-detection.

## Week 1 environment inspection

- Repository at inspection: README-only initial commit.
- Runtime: Python 3.12.3, CPU only; no CUDA-visible GPU.
- Initially installed CV dependencies: none.
- UAVDT availability: absent, so no measured sequence results are claimed.
- Selected detector: lightweight COCO-pretrained `yolo11n.pt`.
- Selected tracker: `deep-sort-realtime`, chosen for its maintained adapter API,
  appearance embeddings, and accessible lifecycle fields.

UAVDT has 100 sequences and uses 30 training sequences plus 70 test sequences;
the DET/MOT test set contains 20 of those test sequences. Object categories are
car, truck, and bus. COCO YOLO exposes matching names, but this does not imply
domain adequacy: small aerial vehicles and motion blur are expected detector
failure sources.
