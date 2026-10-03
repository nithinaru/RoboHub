"""Media for the site and the demo video: skeleton overlays on the generated clips and MuJoCo renders."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]
INK = (20, 20, 20)
LIME = (80, 255, 190)  # BGR
AMBER = (40, 190, 255)


class Encoder:
    """Raw RGB frames -> H.264 mp4 via ffmpeg (browser-playable, yuv420p)."""

    def __init__(self, out: Path, w: int, h: int, fps: float):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-preset",
                "medium",
                "-g",
                "15",
                "-movflags",
                "+faststart",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )

    def add(self, rgb: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


def overlay_clip(
    clip: Path, pose_json: Path, events: dict, out: Path, scale: float = 0.5
) -> None:
    """The generated clip with what the pipeline measured: hand skeleton, pinch point, block box, events."""
    d = json.loads(Path(pose_json).read_text())
    cap = cv2.VideoCapture(str(clip))
    w, h = int(d["width"] * scale), int(d["height"] * scale)
    enc = Encoder(out, w, h, d["fps"])
    for i, f in enumerate(d["frames"]):
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        b = f.get("block")
        if b:
            x, y, bw, bh = (int(v * scale) for v in b["bbox"])
            cv2.rectangle(bgr, (x, y), (x + bw, y + bh), AMBER, 2)
        hnd = f.get("hand")
        if hnd:
            p = (np.array(hnd["pts"]) * scale).astype(int)
            for a_, b_ in HAND_EDGES:
                cv2.line(bgr, tuple(p[a_]), tuple(p[b_]), INK, 4, cv2.LINE_AA)
                cv2.line(bgr, tuple(p[a_]), tuple(p[b_]), LIME, 2, cv2.LINE_AA)
            for q in p:
                cv2.circle(bgr, tuple(q), 3, LIME, -1, cv2.LINE_AA)
            pc = (np.array(hnd["pinch"]) * scale).astype(int)
            cv2.circle(bgr, tuple(pc), 7, (255, 255, 255), 2, cv2.LINE_AA)
        label = ""
        if events.get("ok"):
            if events["t_grasp"] <= i < events["t_release"]:
                label = "HELD"
            if abs(i - events["t_grasp"]) <= 3:
                label = "GRASP"
            if abs(i - events["t_release"]) <= 3:
                label = "RELEASE"
        if label:
            cv2.putText(
                bgr,
                label,
                (14, h - 16),
                cv2.FONT_HERSHEY_PLAIN,
                2.0,
                INK,
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                label,
                (14, h - 16),
                cv2.FONT_HERSHEY_PLAIN,
                2.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        enc.add(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    enc.close()
    cap.release()


class SimFilm:
    """Collects MuJoCo frames from the front camera during a rollout and encodes them."""

    def __init__(
        self, scene, out: Path, w: int = 640, h: int = 480, camera="front"
    ):
        """camera: a named model camera, or an MjvCamera (see hero_cam)."""
        self.scene, self.camera = scene, camera
        g = scene.model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, w), max(g.offheight, h)
        self.r = mujoco.Renderer(scene.model, h, w)
        self.enc = Encoder(out, w, h, 30)

    def frame(self, *_):
        self.r.update_scene(self.scene.data, camera=self.camera)
        self.enc.add(self.r.render())

    def close(self):
        self.enc.close()
        self.r.close()


def hero_cam():
    """A closer free camera than the dataset's "front" camera, on the same side, so the arm fills the frame
    (the film v3/v4 hero camera)."""
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = (0.19, -0.035, 0.09)
    c.distance, c.azimuth, c.elevation = 0.58, -155.0, -24.0
    return c
