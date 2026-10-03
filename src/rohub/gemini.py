"""Gemini stills and Veo clips. GEMINI_API_KEY only. Results cache under data/cache/gemini/."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://generativelanguage.googleapis.com/v1beta"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"
VEO_FAST = "veo-3.1-fast-generate-preview"
VEO = "veo-3.1-generate-preview"
REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "cache" / "gemini"


class GeminiError(RuntimeError):
    pass


def model_for_budget(usd: int | float | None) -> str:
    try:
        amount = int(usd)
    except (TypeError, ValueError):
        amount = 2
    return VEO if amount >= 6 else VEO_FAST


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k:
        raise GeminiError("GEMINI_API_KEY is not set")
    return k


def _call(path: str, body: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        path if path.startswith("http") else f"{BASE}/{path.lstrip('/')}",
        data=data,
        method="POST" if body is not None else "GET",
        headers={"x-goog-api-key": _key(), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise GeminiError(f"Gemini HTTP {e.code}: {detail}") from e


def _cache(kind: str, payload: dict, suffix: str) -> Path:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return CACHE / f"{kind}-{digest}{suffix}"


def image_file(prompt: str, seed: int, on_status=None, reference: Path | None = None) -> Path:
    """One still. reference is unused by the image model; the prompt already restyles the scene."""
    del reference
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = _cache("still", {"model": IMAGE_MODEL, "prompt": prompt, "seed": seed}, ".png")
    if dest.exists():
        return dest
    if on_status:
        on_status({"status": "RUNNING", "progress": 0.2, "model": IMAGE_MODEL})
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    data = _call(f"models/{IMAGE_MODEL}:generateContent", body)
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    inline = next((p.get("inlineData") or p.get("inline_data") for p in parts if p.get("inlineData") or p.get("inline_data")), None)
    if not inline or not inline.get("data"):
        raise GeminiError("Gemini returned no first frame")
    dest.write_bytes(base64.b64decode(inline["data"]))
    if on_status:
        on_status({"status": "SUCCEEDED", "progress": 1, "model": IMAGE_MODEL})
    return dest


def video_file(prompt: str, frame: Path, seed: int, budget: int | None = None, on_status=None) -> tuple[Path, dict]:
    """Animate a still with Veo. Returns the mp4 path and a small routing record."""
    model = model_for_budget(budget if budget is not None else os.environ.get("ROBOHUB_BUDGET", "2"))
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = _cache("veo", {"model": model, "prompt": prompt, "seed": seed, "frame": hashlib.sha256(frame.read_bytes()).hexdigest()[:16]}, ".mp4")
    routing = {"model": model, "provider": "google", "router": f"gemini-{model}"}
    if dest.exists():
        routing["cached"] = True
        return dest, routing
    if on_status:
        on_status({"status": "RUNNING", "progress": 0.1, "model": model})
    b64 = base64.b64encode(frame.read_bytes()).decode()
    started = _call(
        f"models/{model}:predictLongRunning",
        {
            "instances": [{
                "prompt": prompt,
                "image": {"inlineData": {"mimeType": "image/png", "data": b64}},
            }],
            "parameters": {"aspectRatio": "16:9", "durationSeconds": 8},
        },
    )
    name = started.get("name")
    if not name:
        raise GeminiError("Veo did not start a video job")
    deadline = time.time() + 360
    while time.time() < deadline:
        time.sleep(8)
        status = _call(name)
        if status.get("error"):
            raise GeminiError(status["error"].get("message") or "Veo failed")
        if status.get("done"):
            uri = (((status.get("response") or {}).get("generateVideoResponse") or {}).get("generatedSamples") or [{}])[0].get("video", {}).get("uri")
            if not uri:
                raise GeminiError("Veo finished without a video")
            req = urllib.request.Request(uri, headers={"x-goog-api-key": _key()})
            with urllib.request.urlopen(req, timeout=120) as res:
                dest.write_bytes(res.read())
            if on_status:
                on_status({"status": "SUCCEEDED", "progress": 1, "model": model})
            return dest, routing
        if on_status:
            on_status({"status": "RUNNING", "progress": 0.5, "model": model})
    raise GeminiError("Veo timed out")
