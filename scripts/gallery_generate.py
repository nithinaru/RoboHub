"""Generate text-to-video Runway clips for the per-task gallery on the replay site.

Usage: python scripts/gallery_generate.py dry   -> free dry runs, prints router pick and estimated credits
       python scripts/gallery_generate.py go    -> live generations (8 at a time), download, compress, routing.json

Each clip lands in deploy-v2/runs/<slug>/clips/<id>.mp4 (640 px wide, libx264 crf 28, no audio) with
<id>.routing.json {router, model, provider, estimated_credits, seconds, prompt, task_id}.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "deploy-v2" / "runs"
API = "https://api.dev.runwayml.com/v1"

LIGHTING = [
    "soft natural daylight from a window on the left",
    "warm late-afternoon sunlight with long soft shadows",
    "cool overcast light, flat and even",
    "a warm desk lamp at night with a dark background",
    "bright even studio light",
    "mixed daylight with a slight blue tint",
]
TABLE = [
    "a light oak wooden table",
    "a matte white laminate desk",
    "a dark walnut table",
    "a grey concrete worktop",
    "a pale birch plywood workbench",
    "a black matte desk",
]
ANGLE = [
    "a low front view from table height, about 25 degrees down",
    "a front view from slightly to the left, about 35 degrees down",
    "a front view, about 45 degrees down",
]

TAIL = (
    " The static camera stays locked off, the motion is smooth and at natural speed, and every block keeps its"
    " shape and color the whole time. The whole hand and all objects stay fully in frame."
)

SCENES = {
    "push-the-red-block-onto-the-blue-square": {
        "router": "demo-cheap",
        "count": 15,
        "objects": (
            "exactly two things on it: one small red wooden toy block (a 3 cm cube) on the left, and about 15 cm to"
            " its right a square outline of blue masking tape, about 8 cm on each side, stuck flat on the table"
        ),
        "action": (
            " A person's right hand rests its fingertips against the left face of the red block and pushes it"
            " steadily across the table surface, sliding it until the block sits fully inside the blue taped square."
            " The block stays flat on the table while it slides. Then the open hand lifts up and away."
        ),
    },
    "stack-the-red-block-on-the-blue-block": {
        "router": "demo-best",
        "count": 14,
        "objects": (
            "exactly two things on it: one small red wooden cube (3 cm) on the left, and about 20 cm to its right one"
            " flat blue wooden block, a square slab about 6 cm wide and 2 cm tall"
        ),
        "action": (
            " A person's right hand pinches the red cube between thumb and index finger, lifts it about 10 cm,"
            " carries it to the right and sets it down centred on top of the flat blue block, then opens the fingers"
            " and moves away. At the end the red cube rests steadily on top of the blue block."
        ),
    },
    "stack-three-blocks-into-a-tower": {
        "router": "demo-best",
        "count": 15,
        "objects": (
            "exactly three things on it: one flat blue wooden slab (about 6 cm wide and 2 cm tall) in the centre,"
            " one small red wooden cube (3 cm) to its left, and one small green wooden cube (3 cm) to its right"
        ),
        "action": (
            " A person's right hand picks up the red cube and places it centred on top of the blue slab, then picks"
            " up the green cube and places it centred on top of the red cube, then moves away. At the end the three"
            " pieces stand as one tower: blue slab at the bottom, red cube in the middle, green cube on top."
        ),
    },
}


def key() -> str:
    return subprocess.run(
        ["security", "find-generic-password", "-s", "RUNWAY_API_KEY", "-w"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


KEY = key()


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {KEY}",
            "X-Runway-Version": "2024-11-06",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"{e.code} {msg}") from None
    raise RuntimeError("unreachable")


def jobs() -> list[dict]:
    out = []
    for slug, s in SCENES.items():
        for i in range(s["count"]):
            n = i + 2  # offset so the scenes differ from the existing v01 / b01
            prompt = (
                f"Photorealistic video, {ANGLE[n % 3]}, of {TABLE[n % 6]} lit by {LIGHTING[(n + 2 + n // 6) % 6]}, with "
                + s["objects"]
                + "."
                + s["action"]
                + TAIL
            )
            out.append(
                {
                    "slug": slug,
                    "id": f"g{i + 1:02d}",
                    "router": s["router"],
                    "prompt": prompt,
                }
            )
    return out


def body(j: dict) -> dict:
    return {
        "configId": j["router"],
        "input": {"promptText": j["prompt"], "aspectRatio": "16:9", "duration": 5},
    }


def dry(j: dict) -> dict:
    r = call("POST", "/generate/video", {**body(j), "dryRun": True})
    return r.get("routing", r)


def run(j: dict) -> dict:
    dst = RUNS / j["slug"] / "clips" / f"{j['id']}.mp4"
    if dst.exists():
        return {**j, "status": "EXISTS"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    d = dry(j)
    est = (d.get("estimatedCost") or {}).get("credits")
    t0 = time.time()
    task = call("POST", "/generate/video", body(j))
    live = task.get("routing") or d
    tid = task["id"]
    while True:
        time.sleep(8)
        t = call("GET", f"/tasks/{tid}")
        if t["status"] == "SUCCEEDED":
            break
        if t["status"] in ("FAILED", "CANCELLED"):
            return {
                **j,
                "status": t["status"],
                "error": t.get("failure") or t.get("failureCode"),
            }
    secs = round(time.time() - t0, 1)
    raw = dst.with_suffix(".raw.mp4")
    urllib.request.urlretrieve(t["output"][0], raw)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(raw),
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
            str(dst),
        ],
        check=True,
    )
    raw.unlink()
    rec = {
        "router": j["router"],
        "model": live.get("model"),
        "provider": live.get("provider"),
        "estimated_credits": est,
        "seconds": secs,
        "task_id": tid,
        "prompt": j["prompt"],
        "text_to_video": True,
    }
    dst.with_suffix(".routing.json").write_text(json.dumps(rec, indent=1))
    return {**j, "status": "SUCCEEDED", **rec}


def main() -> None:
    mode = sys.argv[1]
    js = jobs()
    if len(sys.argv) > 2:
        js = [j for j in js if j["slug"] in sys.argv[2:]]
    if mode == "dry":
        total = 0.0
        for j in js:
            d = dry(j)
            c = (d.get("estimatedCost") or {}).get("credits") or 0
            total += c
            print(
                j["slug"][:24],
                j["id"],
                j["router"],
                d.get("model"),
                d.get("provider"),
                c,
                flush=True,
            )
        print("TOTAL", total)
        return
    if mode == "more":  # only the ids past the first batch (9 / 8 / 8), which may still be in flight
        first = {"push-the-red-block-onto-the-blue-square": 9, "stack-the-red-block-on-the-blue-block": 8,
                 "stack-three-blocks-into-a-tower": 8}
        js = [j for j in js if int(j["id"][1:]) > first[j["slug"]]]
    with ThreadPoolExecutor(20) as ex:
        for r in ex.map(lambda j: _safe(run, j), js):
            print(
                r["slug"][:24],
                r["id"],
                r.get("status"),
                r.get("model"),
                r.get("estimated_credits"),
                r.get("seconds"),
                r.get("error", ""),
                flush=True,
            )


def _safe(f, j):
    try:
        return f(j)
    except Exception as e:  # report and keep the other generations going
        return {**j, "status": "ERROR", "error": str(e)[:300]}


if __name__ == "__main__":
    main()
