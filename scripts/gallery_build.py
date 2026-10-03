"""Collect every Runway clip per task into deploy-v2/runs/<slug>/clips (web-compressed) and write
deploy-v2/data/gallery.json: {slug: [{id, url, router, model, credits, verdict, reason}]}.
Run after scripts/gallery_generate.py. Idempotent: skips clips already compressed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "deploy-v2"
WEB = REPO / "data" / "web-runs"
REPORT = Path.home() / "helloworld" / "robohub" / "site" / "report"

BOWL = "put-the-red-block-in-the-bowl"
PUSH = "push-the-red-block-onto-the-blue-square"
STACK = "stack-the-red-block-on-the-blue-block"
TOWER = "stack-three-blocks-into-a-tower"


def width(p: Path) -> int:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width",
            "-of",
            "csv=p=0",
            str(p),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(r.stdout.strip().split(",")[0])


def compress(src: Path, dst: Path) -> None:
    """640 px wide, libx264 crf 28, no audio. Copies as-is when already small and web-sized."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and width(dst) <= 640:
        return
    if src.resolve() != dst.resolve() and width(src) <= 640:
        shutil.copy(src, dst)
        return
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-an",
            "-vf",
            "scale=640:-2",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(dst)


def routing(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def entry(
    slug: str, cid: str, r: dict, verdict: str | None = None, reason: str = ""
) -> dict:
    credits = r.get("realized_credits", r.get("estimated_credits"))
    return {
        "id": cid,
        "url": f"runs/{slug}/clips/{cid}.mp4",
        "router": r.get("router") or "",
        "model": r.get("model") or "",
        "credits": int(credits) if credits is not None else None,
        "verdict": verdict,
        "reason": reason,
    }


def short(reason: str) -> str:
    """'no second red object appears: a second red block appears on 10 frames ...' -> the part after the gate name."""
    s = reason.split(": ", 1)[-1] if ": " in reason else reason
    return s.split(":")[0].strip()


def main() -> None:
    g: dict[str, list[dict]] = {BOWL: [], PUSH: [], STACK: [], TOWER: []}

    # bowl: the original v00-v07 run (direct gen4_turbo, judged) then the router bench v08-v25
    run = json.loads((REPORT / "data" / "run.json").read_text())
    for c in run["clips"]:
        cid = c["id"]
        compress(
            REPORT / "media" / f"{cid}_runway.mp4",
            OUT / "runs" / BOWL / "clips" / f"{cid}.mp4",
        )
        verdict = "Accepted" if c["accepted"] else "Rejected"
        g[BOWL].append(
            entry(
                BOWL,
                cid,
                {
                    "router": "direct",
                    "model": c.get("video_model"),
                    "estimated_credits": 25,
                },
                verdict,
                "" if c["accepted"] else short(c["reason"]),
            )
        )
    bench = WEB / BOWL / "clips"
    for mp4 in sorted(bench.glob("v*.mp4")):
        cid = mp4.stem
        r = routing(bench / f"{cid}.routing.json")
        if not r or int(cid[1:]) < 8:
            continue
        compress(mp4, OUT / "runs" / BOWL / "clips" / f"{cid}.mp4")
        g[BOWL].append(entry(BOWL, cid, r))

    # push, stack, tower: the first web-run clip, then the gallery generations (g01...)
    firsts = [
        (PUSH, WEB / "push-the-red-block-into-the-blue-square" / "clips", "v01"),
        (STACK, OUT / "runs" / STACK / "clips", "b01"),
        (STACK, WEB / STACK / "clips", "v01"),
        (TOWER, WEB / TOWER / "clips", "v01"),
    ]
    for slug, d, cid in firsts:
        compress(d / f"{cid}.mp4", OUT / "runs" / slug / "clips" / f"{cid}.mp4")
        g[slug].append(entry(slug, cid, routing(d / f"{cid}.routing.json")))
    for slug in (PUSH, STACK, TOWER):
        d = OUT / "runs" / slug / "clips"
        for mp4 in sorted(d.glob("g*.mp4")):
            g[slug].append(
                entry(slug, mp4.stem, routing(mp4.with_suffix(".routing.json")))
            )

    (OUT / "data" / "gallery.json").write_text(json.dumps(g, indent=1))
    for slug, xs in g.items():
        print(slug, len(xs), sum(x["credits"] or 0 for x in xs), "credits")


if __name__ == "__main__":
    main()
