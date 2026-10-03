"""The whole run: prompt -> Runway clips -> landmarks -> gates -> retarget -> LeRobot dataset -> policy -> sim.

Writes everything under data/runs/<slug>/ and a summary (run.json) that the site reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import gates as G
from .audit import audit
from .plan import Task
from .retarget import TIME_SCALE, replay
from .track import find_events, load_track, to_robot

REPO = Path(__file__).resolve().parents[2]
POSE_PY = REPO / ".venv-pose" / "bin" / "python"
POSE_CACHE = REPO / "data" / "cache" / "pose"
POSE_VERSION = 2  # src/pose/extract.py VERSION
EXTRACTOR = "mediapipe"  # chosen by measurement, docs/POSE-COMPARE.md
JITTER = (
    0.03,
    0.02,
)  # re-anchor / eval region around each accepted clip's cube start (x, y metres)


def log(msg: str) -> None:
    print(msg, flush=True)


def trajectory_embedding(xyz: np.ndarray, dims: int = 1536) -> list[float]:
    """Resample a pinch path (T, 3) into a fixed vector for pgvector cosine search."""
    flat = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return [0.0] * dims
    src = np.linspace(0.0, 1.0, flat.size)
    dst = np.linspace(0.0, 1.0, dims)
    return np.interp(dst, src, flat).astype(float).tolist()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:20]


def extract(clip: Path) -> Path:
    POSE_CACHE.mkdir(parents=True, exist_ok=True)
    out = POSE_CACHE / f"{sha(clip)}_{EXTRACTOR}_v{POSE_VERSION}.json"
    if not out.exists():
        subprocess.run(
            [
                str(POSE_PY),
                str(REPO / "src" / "pose" / "extract.py"),
                str(clip),
                str(out),
                "--extractor",
                EXTRACTOR,
            ],
            check=True,
        )
    return out


def process_clip(clip: Path, meta: dict, run_dir: Path, scene, emit=None) -> dict:
    emit = emit or (lambda ev: None)
    cid = meta["clip_id"]
    pose = extract(clip)
    shutil.copy(pose, run_dir / "pose" / f"{cid}.json")
    tr = load_track(pose)
    ev = find_events(tr)
    from .media import overlay_clip

    evd0 = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in asdict(ev).items()
    }
    overlay_clip(clip, pose, evd0, run_dir / "media" / f"{cid}_skeleton.mp4")
    emit(
        {
            "type": "tracked",
            "clip": cid,
            "file": f"media/{cid}_skeleton.mp4",
            "frames": tr.n,
            "fps": tr.fps,
            "grasp_frame": ev.t_grasp,
            "release_frame": ev.t_release,
        }
    )
    emit({"type": "clip_stage", "clip": cid, "stage": "audit"})
    au = audit(clip, tr.width, tr.height)
    emit({"type": "clip_stage", "clip": cid, "stage": "gates"})
    if au.get("available"):
        bb = au["bowl_bbox_first"]
        bowl_u, bowl_src = (bb[0] + bb[2]) / 2, "Claude bowl box"
    else:
        bowl_u, bowl_src = (
            (float(ev.b1[0]) if ev.b1 is not None else tr.width / 2),
            "block rest position",
        )
    demo = to_robot(tr, ev, bowl_u) if ev.ok else None
    rows = G.clip_gates(tr, ev, au, demo)
    rep = None
    if demo is not None:
        rep = replay(scene, demo)
        rows += G.robot_gates(rep)
    ok, why = G.verdict(rows)
    vec = trajectory_embedding(demo.xyz) if demo is not None else None
    G.publish(
        rows,
        task_id=os.environ.get("ROBOHUB_TASK_ID"),
        clip_id=cid,
        video_storage_path=str(clip),
        trajectory_vector=vec,
    )
    evd = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in asdict(ev).items()
    }
    res = {
        "clip_id": cid,
        "accepted": ok,
        "reason": why,
        "gates": rows,
        "events": evd,
        "audit": au,
        "bowl_u": bowl_u,
        "bowl_source": bowl_src,
        "prompt": {
            k: meta.get(k)
            for k in (
                "lighting",
                "table",
                "bowl",
                "angle",
                "image_prompt",
                "video_prompt",
                "image_model",
                "video_model",
                "note",
            )
        },
        "fps": tr.fps,
        "frames": tr.n,
    }
    if demo is not None:
        res["demo"] = {
            "cube_start": demo.cube_start.round(4).tolist(),
            "m_per_px": demo.m_per_px,
            "lift_m": round(demo.lift_height, 4),
            "carry_m": round(demo.carry_dist, 4),
            "xyz": demo.xyz.round(4).tolist(),
            "closed": demo.closed.astype(int).tolist(),
        }
    if rep is not None:
        res["replay"] = {
            "success": rep.success,
            "cube_end": rep.extra["cube_end"],
            "frames": len(rep.states),
        }
    (run_dir / "results" / f"{cid}.json").write_text(
        json.dumps(res, indent=1, default=float)
    )
    return {"res": res, "demo": demo}


def run(task: Task, run_dir: Path, a) -> dict:
    """The whole pipeline from the command line: the same three stages the site runs (stages.py)."""
    from . import runway, stages

    t_start = time.time()
    for sub in ("clips", "pose", "results", "media", "policy"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    summary: dict = {"task": task.text, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not a.no_generate:
        stages.footage(task, run_dir, a.clips, parallel=False)
    log(f"[run] {len(list(run_dir.glob('clips/*.mp4')))} clips")
    data = stages.training_data(task, run_dir, reanchor_n=a.reanchor, log=log)
    summary["clips"] = data["clips"]
    if not data.get("dataset"):
        (run_dir / "run.json").write_text(json.dumps(summary, indent=1))
        return summary
    summary["reanchor"] = data["reanchor"]
    summary["dataset"] = data["dataset"]
    accepted = [c["clip_id"] for c in data["clips"] if c["accepted"]]
    te = stages.train_and_eval(
        run_dir, accepted, steps=a.train_steps, eval_seeds=a.eval_seeds, log=log, film_cam="front"
    )
    summary["train"] = te["train"]
    summary["eval"] = te["eval"] | {
        f"film_{tag}_seed": f["seed"] for tag, f in te["films"].items()
    }
    summary["time_scale"] = TIME_SCALE
    summary["seconds"] = round(time.time() - t_start, 1)
    summary["runway"] = {"ledger": runway.ledger()}
    (run_dir / "run.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"[run] done in {summary['seconds']} s -> {run_dir / 'run.json'}")
    return summary
