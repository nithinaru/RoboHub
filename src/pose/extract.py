"""Stage 2: whole-arm + hand landmarks and the red block, per frame, from one generated clip.

Runs in .venv-pose (Python 3.11; mediapipe 0.10.14 is the last release that works on this Mac, newer ones abort
with "graph_service.h Service is unavailable").

Extractors (pick with --extractor, compare with scripts/compare_extractors.py):
  mediapipe  MediaPipe Tasks Hand Landmarker (21 image + 21 metric world landmarks) + Pose Landmarker heavy (33)
  rtmw       RTMW whole-body (133 keypoints, 21 per hand) through rtmlib / onnxruntime

Output JSON (one per clip): fps, size, per-frame hand landmarks (pixels), hand score, the pinch point (midpoint of
thumb tip and index tip), the grip aperture (thumb tip to index tip over wrist to index knuckle, scale free),
the arm (shoulder, elbow, wrist) when the pose model sees it, and the red block's pixel centroid and area.

Usage: .venv-pose/bin/python src/pose/extract.py CLIP.mp4 OUT.json [--extractor mediapipe|rtmw]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "data" / "models"

VERSION = 2  # bump when the output changes; the pipeline cache key includes it
THUMB_TIP, INDEX_TIP, WRIST, INDEX_MCP, MIDDLE_MCP = 4, 8, 0, 5, 9


def red_block(frame_bgr: np.ndarray, color: str = "red") -> dict | None:
    """Largest saturated red blob: centroid (px), area (px^2), bbox, and the runner-up blob. Hue wraps at 0/180."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if color != "red":
        raise ValueError("only the red block is tracked")
    m = cv2.inRange(hsv, (0, 120, 70), (8, 255, 255)) | cv2.inRange(
        hsv, (170, 120, 70), (180, 255, 255)
    )
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    i = 1 + int(order[0])
    area = int(stats[i, cv2.CC_STAT_AREA])
    second = int(areas[order[1]]) if len(order) > 1 else 0
    second_c = [float(v) for v in cents[1 + int(order[1])]] if len(order) > 1 else None
    if area < 60:
        return None
    x, y, w, h = (
        int(stats[i, k])
        for k in (
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
        )
    )
    return {
        "c": [float(cents[i][0]), float(cents[i][1])],
        "area": area,
        "bbox": [x, y, w, h],
        "second_ratio": round(second / area, 3),
        "second_area": second,
        "second_c": second_c,
    }


def hand_summary(pts: np.ndarray) -> dict:
    """pts: [21, 2] pixels (any hand model using the 21-point MediaPipe/COCO-hand order)."""
    pinch = (pts[THUMB_TIP] + pts[INDEX_TIP]) / 2
    scale = float(np.linalg.norm(pts[WRIST] - pts[INDEX_MCP])) + 1e-6
    ap = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])) / scale
    return {"pinch": pinch.tolist(), "aperture_2d": ap, "hand_px": scale}


def run_mediapipe(cap, fps, w, h, frames_bgr):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(MODELS / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(MODELS / "pose_landmarker_heavy.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
        )
    )
    out = []
    for i, bgr in enumerate(frames_bgr):
        ts = int(round(i * 1000 / fps))
        img = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        )
        hr = hand.detect_for_video(img, ts)
        pr = pose.detect_for_video(img, ts)
        rec: dict = {"hand": None, "arm": None}
        if hr.hand_landmarks:
            # the demonstrating hand = the one with the best presence (only one hand is in these clips)
            best = max(
                range(len(hr.hand_landmarks)), key=lambda k: hr.handedness[k][0].score
            )
            lm = hr.hand_landmarks[best]
            pts = np.array([[p.x * w, p.y * h] for p in lm])
            world = np.array([[p.x, p.y, p.z] for p in hr.hand_world_landmarks[best]])
            s = hand_summary(pts)
            ap3 = float(
                np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP])
                / (np.linalg.norm(world[WRIST] - world[INDEX_MCP]) + 1e-9)
            )
            rec["hand"] = {
                "pts": np.round(pts, 1).tolist(),
                "score": float(hr.handedness[best][0].score),
                "label": hr.handedness[best][0].category_name,
                "aperture": ap3,
                "pinch_mm": float(
                    np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP]) * 1000
                ),
                **s,
            }
        if pr.pose_landmarks:
            lm = pr.pose_landmarks[0]
            arm = {}
            for name, idx in (
                ("shoulder", 12),
                ("elbow", 14),
                ("wrist", 16),
                ("l_shoulder", 11),
                ("l_elbow", 13),
                ("l_wrist", 15),
            ):
                arm[name] = [lm[idx].x * w, lm[idx].y * h, float(lm[idx].visibility)]
            rec["arm"] = arm
        out.append(rec)
    return out


def run_rtmw(cap, fps, w, h, frames_bgr):
    from rtmlib import Wholebody

    model = Wholebody(
        mode="performance", backend="onnxruntime", device="cpu"
    )  # RTMW-x wholebody, 133 kp
    out = []
    for bgr in frames_bgr:
        kps, scores = model(bgr)
        rec: dict = {"hand": None, "arm": None}
        if len(kps):
            best = None
            for p in range(len(kps)):
                for lo in (
                    91,
                    112,
                ):  # left hand 91-111, right hand 112-132 (COCO-WholeBody)
                    sc = float(np.mean(scores[p][lo : lo + 21]))
                    if best is None or sc > best[0]:
                        best = (sc, p, lo)
            sc, p, lo = best
            pts = np.asarray(kps[p][lo : lo + 21], dtype=float)
            s = hand_summary(pts)
            rec["hand"] = {
                "pts": np.round(pts, 1).tolist(),
                "score": sc,
                "label": "right" if lo == 112 else "left",
                "aperture": s["aperture_2d"],
                **s,
            }
            k, sc_ = kps[p], scores[p]
            rec["arm"] = {
                n: [float(k[i][0]), float(k[i][1]), float(sc_[i])]
                for n, i in (
                    ("shoulder", 6),
                    ("elbow", 8),
                    ("wrist", 10),
                    ("l_shoulder", 5),
                    ("l_elbow", 7),
                    ("l_wrist", 9),
                )
            }
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--extractor", default="mediapipe", choices=["mediapipe", "rtmw"])
    a = ap.parse_args()
    cap = cv2.VideoCapture(str(a.clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fn = run_mediapipe if a.extractor == "mediapipe" else run_rtmw
    recs = fn(cap, fps, w, h, frames)
    for rec, f in zip(recs, frames):
        rec["block"] = red_block(f)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "clip": a.clip.name,
                "extractor": a.extractor,
                "version": VERSION,
                "fps": fps,
                "width": w,
                "height": h,
                "n_frames": len(frames),
                "frames": recs,
            }
        )
    )
    found = sum(r["hand"] is not None for r in recs)
    print(
        f"{a.clip.name} {a.extractor}: {len(frames)} frames at {fps:.2f} fps, hand in {found}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
