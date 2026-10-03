"""Scene audit by Claude (vision): counts what is on the table and where the bowl is, from the first and last frames.

Runs the local Claude Code CLI headless (`claude -p`, model sonnet, Read tool only) on two stills. The answer is
cached by the clip's hash, so a clip is audited once. If the CLI is missing or fails, the audit returns
available=False and the gates that need it say so instead of passing silently.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "cache" / "audit"
CLAUDE = (
    shutil.which("claude", path="/usr/local/bin:/opt/homebrew/bin")
    or "/usr/local/bin/claude"
)

PROMPT = """You are auditing a generated video clip for a robot-learning dataset. Read the two images in the
current directory: first.jpg (the first frame) and last.jpg (the last frame). Both are {w}x{h} pixels.
Answer ONLY with one line of compact JSON, no prose, with exactly these keys:
{{"red_blocks_first": int, "bowls_first": int, "hands_first": int, "bowl_bbox_first": [x0, y0, x1, y1],
"block_in_bowl_last": bool, "red_blocks_last": int, "note": "at most 12 words"}}
red_blocks counts every red block or cube visible anywhere. bowl_bbox_first is the bowl's pixel box in
first.jpg. block_in_bowl_last is true only if the red block rests inside the bowl in last.jpg."""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:20]


def audit(clip: Path, width: int = 1280, height: int = 720) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = _sha(clip)
    out = CACHE / f"{key}.json"
    if out.exists():
        return json.loads(out.read_text())
    work = CACHE / key
    work.mkdir(exist_ok=True)
    for name, args in (
        ("first.jpg", ["-i", str(clip)]),
        ("last.jpg", ["-sseof", "-0.15", "-i", str(clip)]),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                *args,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(work / name),
            ],
            check=True,
        )
    try:
        r = subprocess.run(
            [
                CLAUDE,
                "-p",
                PROMPT.format(w=width, h=height),
                "--allowedTools",
                "Read",
                "--model",
                "sonnet",
            ],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=240,
        )
        m = re.search(r"\{.*\}", r.stdout, re.S)
        res = json.loads(m.group(0)) if m else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        res = None
    if not res:
        return {"available": False, "why": "Claude CLI gave no parseable answer"}
    res["available"] = True
    res["model"] = "claude sonnet via Claude Code CLI"
    out.write_text(json.dumps(res, indent=1))
    return res
