# UAVDT data

Dataset files are intentionally excluded from Git. Download the original UAVDT
DET/MOT benchmark and toolkit from the links published by the dataset authors.
The loader expects this original structure:

```text
data/UAVDT/
├── UAVDT_Benchmark_M/          # UAV-benchmark-M is also accepted
│   ├── M0101/
│   │   ├── img000001.jpg
│   │   └── ...
│   └── ...
└── UAV-benchmark-MOTD_v1.0/
    └── GT/
        ├── M0101_gt_whole.txt
        └── ...
```

Each `*_gt_whole.txt` row has nine integer fields:

```text
frame,target_id,left,top,width,height,out_of_view,occlusion,category
```

UAVDT category IDs are `1=car`, `2=truck`, and `3=bus`. The loader rejects
unknown IDs and malformed rows rather than silently relabeling them. Attribute
files under `M_attr/{train,test}` define the official split and sequence-level
conditions; they are not required for frame-level baseline inference.

Validate parsing and save a few ground-truth overlays before inference:

```bash
uav-track --config configs/baseline.yaml --validate-dataset
```
