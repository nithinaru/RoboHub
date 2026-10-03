"""Compare MediaPipe (Hand Landmarker + Pose heavy) and RTMW whole-body (rtmlib) on the same Runway clips.

Usage: .venv/bin/python scripts/compare_extractors.py data/work/pose/v01_mediapipe.json data/work/pose/v01_rtmw.json ...
Prints detections, detections with all 21 hand points inside the image, and pinch-point jitter (second difference,
px per frame^2; median and 95th percentile) for each file.
"""

import json
import sys

import numpy as np

for path in sys.argv[1:]:
    d = json.load(open(path))
    F = d["frames"]
    det = sum(1 for f in F if f["hand"])
    inb = sum(1 for f in F if f["hand"] and all(0 <= x < d["width"] and 0 <= y < d["height"] for x, y in f["hand"]["pts"]))
    pin = np.array([f["hand"]["pinch"] if f["hand"] else [np.nan, np.nan] for f in F])
    acc = np.linalg.norm(pin[2:] - 2 * pin[1:-1] + pin[:-2], axis=1)
    print(f"{path}: frames {len(F)}, detected {det}, in-image {inb}, "
          f"jitter median {np.nanmedian(acc):.1f} px, p95 {np.nanpercentile(acc, 95):.1f} px")
