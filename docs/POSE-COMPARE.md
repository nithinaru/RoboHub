# Which tracker: MediaPipe or RTMW

Measured 2026-09-23 on two real Runway gen4_turbo clips (1280x720, 24 fps, 121 frames each), CPU only (M-series Mac).
Reproduce: `.venv-pose/bin/python src/pose/extract.py CLIP OUT --extractor {mediapipe,rtmw}`, then
`.venv/bin/python scripts/compare_extractors.py OUT...`.

| clip | tracker | frames with a hand | all 21 points in image | pinch jitter median / p95 (px) | wall time |
|---|---|---|---|---|---|
| v01 (hand leaves frame at ~3.8 s) | MediaPipe Hand Landmarker + Pose heavy | 92 | 85 | 3.6 / 13.8 | 19 s |
| v01 | RTMW-x whole-body (rtmlib, onnxruntime) | 121 | 104 | 3.5 / 355.1 | not clean (includes weight download) |
| v00 | MediaPipe | 115 | 106 | 2.5 / 29.4 | about 20 s |
| v00 | RTMW-x | 121 | 121 | 5.0 / 122.4 | 138 s |

Choice: **MediaPipe**, for three measured reasons.

1. It says when the hand is gone. In v01 the hand leaves the frame at about frame 92; MediaPipe stops reporting
   it, RTMW keeps returning a hand on every frame (its SimCC scores stay around 1 to 2.8 after the hand exits,
   against about 4.3 while it is visible). The visibility gate needs "no hand" to mean no hand.
2. Its jitter tail is far smaller (p95 13.8 px vs 355 px on v01), which matters because the pinch point becomes
   the robot's grasp point.
3. It is about 7 times faster on this Mac (about 20 s vs 138 s per 5 s clip).

RTMW's median jitter is as good, and it gives the whole body in one pass, so it stays in the extractor as an
option (`--extractor rtmw`).

Finger aperture (thumb tip to index tip over hand size) was also measured and is not used to switch the gripper:
on v01 it is 0.13 to 0.25 before the grasp,
0.12 to 0.40 while the block is held and up to 0.39 after the release, so the ranges overlap completely. The grasp and release are read from the block's motion instead (see src/rohub/track.py).
