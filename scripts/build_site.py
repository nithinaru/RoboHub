"""Copy one run's media and numbers into site/ (site/data/run.json + site/media/*).

Usage: .venv/bin/python scripts/build_site.py [data/runs/<slug>]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else REPO / "data" / "runs" / "put-the-red-block-in-the-bowl"
)
SITE = REPO / "site" / "report"


def small(src: Path, dst: Path, width: int = 640) -> None:
    """Re-encode to a small browser-friendly H.264 file (skipped if already newer than the source)."""
    if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"scale={width}:-2",
            "-c:v",
            "libx264",
            "-crf",
            "24",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
    )


def main() -> None:
    run = json.loads((RUN / "run.json").read_text())
    media = SITE / "media"
    media.mkdir(parents=True, exist_ok=True)
    (SITE / "data").mkdir(exist_ok=True)
    clips = []
    for c in run["clips"]:
        cid = c["clip_id"]
        res = json.loads((RUN / "results" / f"{cid}.json").read_text())
        small(RUN / "clips" / f"{cid}.mp4", media / f"{cid}_runway.mp4")
        small(RUN / "media" / f"{cid}_skeleton.mp4", media / f"{cid}_skeleton.mp4")
        robot = RUN / "media" / f"{cid}_robot.mp4"
        if robot.exists():
            small(robot, media / f"{cid}_robot.mp4", 480)
        p = res["prompt"]
        clips.append(
            {
                "id": cid,
                "accepted": res["accepted"],
                "reason": res["reason"],
                "scene": ", ".join(
                    x
                    for x in (
                        p.get("table"),
                        (p.get("bowl") or "") + " bowl" if p.get("bowl") else None,
                        p.get("lighting"),
                    )
                    if x
                ),
                "note": p.get("note"),
                "image_model": p.get("image_model"),
                "video_model": p.get("video_model"),
                "video_prompt": p.get("video_prompt"),
                "gates": [
                    {k: g[k] for k in ("id", "name", "passed", "value", "limit", "why")}
                    for g in res["gates"]
                ],
                "robot": robot.exists(),
                "lift_cm": round(res["demo"]["lift_m"] * 100, 1)
                if "demo" in res
                else None,
                "carry_cm": round(res["demo"]["carry_m"] * 100, 1)
                if "demo" in res
                else None,
            }
        )
    for tag in ("success", "failure"):
        f = RUN / "media" / f"policy_{tag}.mp4"
        if f.exists():
            small(f, media / f"policy_{tag}.mp4", 480)
    ledger = run.get("runway", {}).get("ledger", [])
    credits = sum(
        int(e.get("cost", 0))
        for e in ledger
        if e.get("status") in ("SUCCEEDED", "UNLOGGED")
    )
    ev = dict(run.get("eval", {}))
    ev.pop("episodes", None)
    out = {
        "task": run["task"],
        "clips": clips,
        "reanchor": run.get("reanchor"),
        "dataset": run.get("dataset"),
        "train": run.get("train"),
        "eval": ev,
        "time_scale": run.get("time_scale"),
        "credits": credits,
        "runway_calls": len([e for e in ledger if e.get("task_id")]),
        "seconds": run.get("seconds"),
    }
    (SITE / "data" / "run.json").write_text(json.dumps(out, indent=1))
    (SITE / "data" / "run.js").write_text("window.RUN = " + json.dumps(out) + ";\n")  # works from file:// too
    print(f"site data: {len(clips)} clips, {credits} credits, media in {media}")


if __name__ == "__main__":
    main()
