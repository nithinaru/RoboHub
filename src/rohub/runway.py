"""Minimal Runway API client with a hard credit cap, a cost ledger and a content-addressed cache.

The API key is read from the RUNWAY_API_KEY environment variable only (bin/robohub fills it from the macOS
Keychain at runtime). It is never written to disk or logged.

Every paid call:
  1. checks the cache (a request that was already generated is never regenerated),
  2. reads the live balance (GET /v1/organization) and refuses if the call would take the balance below
     FLOOR_BALANCE (check and submission under one lock, so concurrent calls never both pass on one balance),
  3. submits, polls, downloads the output into data/cache/runway/,
  4. appends a ledger line (model, estimated cost, balance before/after) to data/runway-ledger.jsonl.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = "https://api.dev.runwayml.com/v1"
VERSION = "2024-11-06"
LEDGER = REPO / "data" / "runway-ledger.jsonl"
CACHE = REPO / "data" / "cache" / "runway"

START_BALANCE = 460  # balance on 2026-09-23 before this build made any call
# His budget rule since 2026-09-24 (replaces the 350-credit cap): spend down to a balance of 25, no lower.
FLOOR_BALANCE = int(os.environ.get("ROBOHUB_FLOOR_BALANCE", "25"))
HARD_CAP = START_BALANCE - FLOOR_BALANCE  # kept for `rohub budget`
_LOCK = threading.Lock()

# Published prices (docs.dev.runwayml.com/guides/pricing, read 2026-09-23)
PRICE = {
    "gen4_image": lambda req: (
        5 if req.get("ratio", "1280:720") in ("1280:720", "720:1280") else 8
    ),
    "gen4_image_turbo": lambda req: 2,
    "gen4_turbo": lambda req: 5 * int(req.get("duration", 10)),
}


class BudgetError(RuntimeError):
    pass


def _key() -> str:
    k = os.environ.get("RUNWAY_API_KEY", "")
    if not k:
        raise RuntimeError(
            "RUNWAY_API_KEY is not set; run through bin/robohub, which reads it from the Keychain"
        )
    return k


def _call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {_key()}",
            "X-Runway-Version": VERSION,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Runway {method} {path} -> HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"
        ) from None


def balance() -> int:
    return int(_call("GET", "/organization")["creditBalance"])


def ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [
        json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()
    ]


def spent_so_far(current_balance: int | None = None) -> int:
    """Credits spent by this build: the live balance drop since START_BALANCE (the ground truth), or the ledger's
    estimate if the balance is not given."""
    if current_balance is not None:
        return START_BALANCE - current_balance
    return sum(
        int(e.get("cost", 0)) for e in ledger() if e.get("status") == "SUCCEEDED"
    )


def _log(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def cache_key(endpoint: str, body: dict) -> str:
    return hashlib.sha256(
        json.dumps([endpoint, body], sort_keys=True).encode()
    ).hexdigest()[:20]


def price(body: dict) -> int:
    return PRICE[body["model"]](body)


def is_cached(endpoint: str, body: dict, suffix: str) -> bool:
    return cached_output(endpoint, body, suffix) is not None


def cached_output(endpoint: str, body: dict, suffix: str) -> Path | None:
    key = cache_key(endpoint, body)
    out = CACHE / f"{key}{suffix}"
    return out if out.exists() and (CACHE / f"{key}.json").exists() else None


def generate(
    endpoint: str,
    body: dict,
    suffix: str,
    label: str,
    poll_s: float = 5.0,
    on_status=None,
) -> Path:
    """Run one generation (or return the cached output). endpoint: 'text_to_image' | 'image_to_video'.

    on_status(dict) is called with every task status Runway reports (status, progress), for live progress."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key(endpoint, body)
    out = CACHE / f"{key}{suffix}"
    meta = CACHE / f"{key}.json"
    if out.exists() and meta.exists():
        if on_status:
            on_status({"status": "CACHED", "progress": 1.0, "cost": 0})
        return out

    model = body["model"]
    cost = PRICE[model](body)
    # Runway takes the credits when a task is created (the ledger's balances show it), so the check and the
    # submission happen under one lock: a concurrent call always sees a balance that already excludes every
    # task submitted before it.
    with _LOCK:
        bal = balance()
        if bal - cost < FLOOR_BALANCE:
            raise BudgetError(
                f"{label}: {model} costs {cost}; balance {bal} would go below the floor of {FLOOR_BALANCE}"
            )
        t0 = time.time()
        task = _call("POST", f"/{endpoint}", body)
    tid = task["id"]
    status, detail = "PENDING", {}
    if on_status:
        on_status({"status": status, "progress": 0.0, "task_id": tid, "cost": cost})
    while True:
        time.sleep(poll_s)
        detail = _call("GET", f"/tasks/{tid}")
        status = detail.get("status", "")
        if on_status:
            on_status(
                {
                    "status": status,
                    "progress": float(detail.get("progress") or 0.0),
                    "task_id": tid,
                    "cost": cost,
                    "seconds": round(time.time() - t0, 1),
                }
            )
        if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if time.time() - t0 > 900:
            status = "TIMEOUT"
            break
    bal_after = balance()
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "endpoint": endpoint,
        "model": model,
        "task_id": tid,
        "status": status,
        "cost": cost if status == "SUCCEEDED" else 0,
        "balance_before": bal,
        "balance_after": bal_after,
        "seconds": round(time.time() - t0, 1),
        "cache_key": key,
    }
    if status != "SUCCEEDED":
        entry["failure"] = str(
            detail.get("failure") or detail.get("failureCode") or ""
        )[:300]
        _log(entry)
        raise RuntimeError(
            f"{label}: Runway task {tid} ended {status}: {entry['failure']}"
        )
    url = detail["output"][0]
    with urllib.request.urlopen(url, timeout=120) as r:
        out.write_bytes(r.read())
    _log(entry)
    stored = {
        k: v for k, v in body.items() if k not in ("promptImage", "referenceImages")
    }
    meta.write_text(
        json.dumps(
            {
                "request": stored,
                "task_id": tid,
                "label": label,
                "model": model,
                "cost": cost,
            },
            indent=2,
        )
    )
    return out


# ---------- Model Router (POST /v1/generate/video with a router config id) ----------
# A router is a saved config (slug) that picks the model per request: optimizeFor cost | latency | quality, an
# optional price ceiling (settings.maxCreditsPerGeneration.video), capacity fallback. A dry run is free: it returns
# the model the router would pick and the estimated credits, and creates no task.

ROUTER_CACHE_SUFFIX = ".mp4"


def routers() -> list[dict]:
    return _call("GET", "/routers").get("data", [])


def router_body(config_id: str, prompt: str, first_frame: Path, duration: int = 5, ratio: str = "16:9") -> dict:
    return {
        "configId": config_id,
        "input": {
            "promptText": prompt,
            "aspectRatio": ratio,
            "duration": duration,
            "referenceImages": [{"uri": image_data_uri(first_frame), "role": "first"}],
        },
    }


def router_dry_run(body: dict) -> dict:
    """The router's pick for this request, without generating: {model, provider, configId, resolvedSettings,
    resolvedInput, estimatedCost: {credits}}. Free: no task, nothing billed."""
    return _call("POST", "/generate/video", {**body, "dryRun": True})["routing"]


def _router_key(body: dict) -> str:
    return cache_key("router_video", body)


def cached_routed(body: dict) -> tuple[Path, dict] | None:
    key = _router_key(body)
    out, meta = CACHE / f"{key}.mp4", CACHE / f"{key}.json"
    if out.exists() and meta.exists():
        return out, json.loads(meta.read_text()).get("routing_record", {})
    return None


def generate_routed(
    body: dict,
    label: str,
    poll_s: float = 5.0,
    on_status=None,
    max_credits: float | None = None,
    dry: dict | None = None,
) -> tuple[Path, dict]:
    """One video through a Model Router (or the cached output). Returns (mp4, record) where record is
    {router, model, provider, reason, estimated_credits, realized_credits, seconds, task_id, cached}.

    The dry run happens first (free) unless `dry` is given; if its estimate exceeds max_credits the call is refused
    before anything is spent. realized_credits is the live balance drop across task creation (Runway takes the
    credits when the task is created), measured under the same lock as the direct path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    hit = cached_routed(body)
    if hit:
        out, rec = hit
        rec = {**rec, "cached": True}
        if on_status:
            on_status({"status": "CACHED", "progress": 1.0, "cost": 0, "routing": rec})
        return out, rec
    key = _router_key(body)
    out, meta = CACHE / f"{key}.mp4", CACHE / f"{key}.json"
    router = body["configId"]
    dry = dry or router_dry_run(body)
    est = float(dry.get("estimatedCost", {}).get("credits") or 0)
    if max_credits is not None and est > max_credits:
        raise BudgetError(f"{label}: router {router} picks {dry.get('model')} at {est:g} credits, over the {max_credits:g} left")
    rec = {
        "router": router,
        "model": dry.get("model"),
        "provider": dry.get("provider"),
        "optimize_for": (dry.get("resolvedSettings") or {}).get("optimizeFor"),
        "price_ceiling": (dry.get("resolvedSettings") or {}).get("priceCeiling"),
        "estimated_credits": est,
        "dry_run": dry,
    }
    if on_status:
        on_status({"status": "ROUTED", "progress": 0.0, "cost": est, "routing": rec})
    with _LOCK:
        bal = balance()
        if bal - est < FLOOR_BALANCE:
            raise BudgetError(f"{label}: {est:g} credits would take balance {bal} below the floor of {FLOOR_BALANCE}")
        t0 = time.time()
        task = _call("POST", "/generate/video", body)
        bal_mid = balance()
    tid = task["id"]
    live = task.get("routing") or {}
    rec.update(
        {
            "model": live.get("model", rec["model"]),
            "provider": live.get("provider", rec["provider"]),
            "task_id": tid,
            "realized_credits": bal - bal_mid,
            "live_estimated_credits": (live.get("estimatedCost") or {}).get("credits"),
            "capacity_fallback": live.get("capacityFallback"),
            "balance_before": bal,
        }
    )
    status, detail = "PENDING", {}
    if on_status:
        on_status({"status": status, "progress": 0.0, "task_id": tid, "cost": est, "routing": rec})
    while True:
        time.sleep(poll_s)
        detail = _call("GET", f"/tasks/{tid}")
        status = detail.get("status", "")
        if on_status:
            on_status(
                {
                    "status": status,
                    "progress": float(detail.get("progress") or 0.0),
                    "task_id": tid,
                    "cost": est,
                    "seconds": round(time.time() - t0, 1),
                    "routing": rec,
                }
            )
        if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if time.time() - t0 > 900:
            status = "TIMEOUT"
            break
    rec["seconds"] = round(time.time() - t0, 1)
    rec["status"] = status
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "endpoint": "generate/video",
        "router": router,
        "model": rec["model"],
        "provider": rec["provider"],
        "task_id": tid,
        "status": status,
        "estimated_cost": est,
        "cost": rec["realized_credits"],
        "balance_before": bal,
        "balance_after_create": bal_mid,
        "seconds": rec["seconds"],
        "cache_key": key,
    }
    if status != "SUCCEEDED":
        entry["failure"] = str(detail.get("failure") or detail.get("failureCode") or "")[:300]
        _log(entry)
        raise RuntimeError(f"{label}: routed task {tid} ({rec['model']}) ended {status}: {entry['failure']}")
    with urllib.request.urlopen(detail["output"][0], timeout=180) as r:
        out.write_bytes(r.read())
    _log(entry)
    stored = {"configId": router, "input": {k: v for k, v in body["input"].items() if k != "referenceImages"}}
    rec["cached"] = False
    meta.write_text(json.dumps({"request": stored, "task_id": tid, "label": label, "routing_record": rec}, indent=2))
    return out, rec
