# UAV visual tracking baseline

A reproducible Week 1 baseline for multi-object vehicle tracking in UAV
imagery. It deliberately separates detector quality, association behavior,
track lifecycle, and runtime:

```text
UAVDT frames → COCO-pretrained YOLO → DeepSORT → tracks, video, logs, metrics
```

This is an experimental baseline, not a claim of deployment robustness.

## Architecture

- `data/uavdt.py`: strict original-format UAVDT frame and annotation reader
- `detection/yolo.py`: Ultralytics YOLO adapter and explicit class filtering
- `tracking/deep_sort.py`: DeepSORT adapter and lifecycle state extraction
- `evaluation/metrics.py`: IoU detection metrics and basic identity diagnostics
- `visualization/drawing.py`: consistent ground-truth, detection, and track overlays
- `pipeline.py`: orchestration, timing, CSV logs, video, and JSON summary

The tracker receives only detector output. Consequently, a missed YOLO
detection is logged as a detector miss rather than attributed to DeepSORT.

## Setup

Python 3.10 or newer is required. Python 3.12 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For CUDA, install the PyTorch wheel matching the host CUDA runtime first, then
install this project. Verify `torch.cuda.is_available()` and set
`runtime.device` to `0`. CPU is the safe default.

The first run downloads the configured Ultralytics weight file to its standard
cache. Model weights are not committed.

## Dataset

UAVDT is the primary dataset because it contains multi-object aerial vehicle
sequences. It defines the categories `car`, `truck`, and `bus`; these names
exist in COCO, although a COCO detector is not adapted to aerial scale or
viewpoint. UAV123 is intentionally deferred to later single-object work.

Dataset files are not redistributed. See [data/README.md](data/README.md) for
the original layout and annotation fields. Put a representative subset under
`data/UAVDT` or pass `--dataset-root`. Always verify it first:

```bash
uav-track --config configs/baseline.yaml \
  --dataset-root /path/to/UAVDT \
  --sequence M0101 \
  --validate-dataset
```

This saves ground-truth overlays to
`outputs/baseline/dataset_validation/`. Inspect box alignment before running
the model.

## Run

```bash
uav-track --config configs/baseline.yaml \
  --dataset-root /path/to/UAVDT \
  --sequence M0101
```

The source wrapper is equivalent after editable installation:

```bash
python scripts/run_tracking.py --config configs/baseline.yaml
```

Useful reproducible overrides include `--start-frame`, `--end-frame`,
`--output`, and `--device`. Persistent experiment changes should be made in a
copied YAML configuration.

## Configuration

`configs/baseline.yaml` records:

- YOLO weight, confidence/NMS thresholds, input size, and allowed classes
- DeepSORT `max_age`, `n_init`, cosine distance, appearance embedder, and its device
- dataset sequence/frame range and output behavior
- runtime device
- optional inclusive detector-suppression frame range

The default `yolo11n.pt` is lightweight. It is not trained in this project.
DeepSORT uses its MobileNet appearance embedder. Set `embedder_gpu: true` only
when CUDA is available; this is independent of YOLO's `runtime.device`.
`max_age` controls how long a track can survive without an associated detection.

## Outputs

Each run writes beneath `output.directory`:

- `<sequence>_tracked.mp4`: confirmed tracks, states, frame number, and FPS
- `detections.csv`: detector output per frame
- `tracks.csv`: track box, identity, confidence, state, age, hits, and update age
- `summary.json`: exact config, package/hardware details, metrics, latency, FPS,
  confidence distribution, and inferred deletion events
- `sample_*.jpg`: a few inspectable tracked frames

Generated files and weights are ignored by Git.

## Evaluation

Detection precision and recall use class-aware greedy IoU matching at 0.5.
Tracking diagnostics count class-aware GT/track matches, ID switches, and
fragmentations. These intentionally small metrics do not replace an official
UAVDT evaluator or HOTA implementation. Predictions marked as DeepSORT
Kalman-only (`predicted`) are visualized and logged but excluded from GT
identity matching.

Latency distributions are recorded separately for YOLO, DeepSORT, and the
pipeline. Pipeline timing currently excludes video rendering/encoding so the
compute path can be compared independently; output encoding should be profiled
separately for deployment.

### Controlled target-loss baseline

Set both values in a copied config:

```yaml
experiment:
  suppress_detections_start: 100
  suppress_detections_end: 140
```

This controlled detector outage shows whether tracks persist for `max_age`,
when they are deleted, and whether reappearance receives a new ID. Inspect
`tracks.csv` and `summary.json`. It does not implement re-detection or alter
DeepSORT internals.

## Reproducibility checklist

Retain the config, command, Git commit, UAVDT sequence/frame range, and
`summary.json`. The summary captures Python, package versions, device, model,
thresholds, metrics, and timing. Experiment-note guidance is in
[experiments/README.md](experiments/README.md).

## Current limitations

- Generic COCO-pretrained YOLO may miss small, blurred, or occluded aerial vehicles.
- Matching COCO/UAVDT class names do not establish cross-domain detector accuracy.
- No dedicated re-detection, target search, or long-term identity store.
- No UAV-specific tracker or camera-motion compensation.
- No 3D/depth information.
- No Jetson, TensorRT, FP16, power, or memory optimization.
- Basic identity diagnostics are not full MOTA/IDF1/HOTA evaluation.

## Next steps

Run representative UAVDT subsets and classify failures as detector misses,
association errors, lifecycle deletion, or runtime limitations. Week 2 should
use those measurements to select one targeted improvement—likely detector
domain adaptation analysis, camera-motion robustness, or explicit
re-acquisition—without conflating the failure sources.
# uav-visual-tracking-system
