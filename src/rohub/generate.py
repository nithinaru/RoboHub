"""Stage 1: Runway generates the human demonstrations.

Clip 1's first frame comes from gen4_image (text only). Every later first frame is a gen4_image_turbo restyle of
that base frame (new table, bowl colour, lighting, camera angle), which keeps the hand-ready-to-pinch composition
and costs 2 credits instead of 5. Each first frame is animated by gen4_turbo image-to-video, 5 s.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import runway
from .plan import Mismatch, Task, Variant, as_dict, variants

REPO = runway.REPO
RATIO_IMG = "1280:720"
RATIO_VID = "1280:720"
CLIP_SECONDS = 5


def cost_per_clip(first: bool) -> int:
    return (5 if first else 2) + 5 * CLIP_SECONDS


def frame_body(v: Variant, base: Path | None, seed: int) -> dict:
    if base is None:
        body = {
            "model": "gen4_image",
            "promptText": v.image_prompt,
            "ratio": RATIO_IMG,
            "seed": seed,
        }
    else:
        prompt = v.restyle_prompt
        body = {
            "model": "gen4_image_turbo",
            "promptText": prompt,
            "ratio": RATIO_IMG,
            "seed": seed,
            "referenceImages": [{"uri": runway.image_data_uri(base), "tag": "base"}],
        }
    return body


def first_frame(v: Variant, base: Path | None, seed: int, on_status=None) -> Path:
    body = frame_body(v, base, seed)
    return runway.generate(
        "text_to_image", body, ".png", f"{v.clip_id} first frame", on_status=on_status
    )


def video_body(v: Variant, frame: Path, seed: int) -> dict:
    return {
        "model": "gen4_turbo",
        "promptImage": runway.image_data_uri(frame),
        "promptText": v.video_prompt,
        "ratio": RATIO_VID,
        "duration": CLIP_SECONDS,
        "seed": seed,
    }


def animate(v: Variant, frame: Path, seed: int, on_status=None) -> Path:
    body = video_body(v, frame, seed)
    return runway.generate(
        "image_to_video", body, ".mp4", f"{v.clip_id} video", on_status=on_status
    )


def route(v: Variant, frame: Path, router: str, on_status=None, max_credits=None, label: str | None = None,
          dry: dict | None = None) -> tuple[Path, dict]:
    """The clip's 5 s video through a Runway Model Router (router = config slug) instead of a fixed model: a free
    dry run first (model, provider, estimated credits), then the live task. Cached by request hash."""
    body = runway.router_body(router, v.video_prompt, frame, duration=CLIP_SECONDS)
    return runway.generate_routed(body, label or f"{v.clip_id} video via {router}", on_status=on_status,
                                  max_credits=max_credits, dry=dry)


def _record(v: Variant, i: int, img: Path, vid: Path, clips_dir: Path, routing: dict | None = None,
            clip_id: str | None = None) -> dict:
    cid = clip_id or v.clip_id
    shutil.copy(img, clips_dir / f"{cid}.png")
    shutil.copy(vid, clips_dir / f"{cid}.mp4")
    rec = {
        **as_dict(v),
        "clip_id": cid,
        "image_model": "gen4_image" if i == 0 else "gen4_image_turbo",
        "video_model": routing["model"] if routing else "gen4_turbo",
        "first_frame": f"clips/{cid}.png",
        "video": f"clips/{cid}.mp4",
    }
    if cid != v.clip_id:
        rec["variant"] = v.clip_id  # same scene and prompts as this variant (a router comparison reuses them)
    if routing:
        rec["router"] = routing["router"]
        rec["routing"] = f"clips/{cid}.routing.json"
        (clips_dir / f"{cid}.routing.json").write_text(json.dumps(routing, indent=2))
    (clips_dir / f"{cid}.json").write_text(json.dumps(rec, indent=2))
    return rec


def existing_clips(run_dir: Path) -> list[str]:
    """The clip ids a run already has footage for (clips/vNN.json), in order."""
    return sorted(p.stem for p in (run_dir / "clips").glob("v[0-9][0-9].json"))


def next_index(run_dir: Path) -> int:
    """The variant index a new clip of this run starts at: after the highest existing id, so a clip that was cut
    from the run (a rejection) is never regenerated or re-bought."""
    ids = existing_clips(run_dir)
    return max(int(c[1:]) for c in ids) if ids else 0


def base_frame(run_dir: Path) -> Path:
    """Clip 1's gen4_image first frame: the frame every restyle of this run refers to."""
    return run_dir / "clips" / "v01.png"


def estimate_more(task: Task, run_dir: Path, n: int, fixes: dict[str, list[Mismatch]] | None = None) -> dict:
    """Credits generate(..., start=next_index(run_dir)) will really spend for n new clips of an existing run."""
    seed0, start, base = seed_for(task), next_index(run_dir), base_frame(run_dir)
    cost = clips_cached = 0
    for j, v in enumerate(variants(task, start + n, fixes)[start:]):
        i = start + j
        fb = frame_body(v, base, seed0 + i)
        img = runway.cached_output("text_to_image", fb, ".png")
        if img is None:
            cost += cost_per_clip(False)
            continue
        vb = video_body(v, img, seed0 + i)
        if runway.cached_output("image_to_video", vb, ".mp4") is None:
            cost += runway.price(vb)
        else:
            clips_cached += 1
    return {"cost": cost, "cached_frames": 0, "cached_clips": clips_cached, "start": start}


def estimate(
    task: Task, n: int, fixes: dict[str, list[Mismatch]] | None = None
) -> dict:
    """Credits the next generate() for this sentence will really spend: requests already in the cache are free."""
    seed0 = seed_for(task)
    cost = cached = clips_cached = 0
    base = None
    for i, v in enumerate(variants(task, n, fixes)):
        fb = frame_body(v, base, seed0 + i)
        img = runway.cached_output("text_to_image", fb, ".png")
        if img is None:
            cost += runway.price(fb) + 5 * CLIP_SECONDS
            if base is None:
                break  # every restyle refers to the base frame: nothing further can be known to be cached
            continue
        cached += 1
        base = base or img
        vb = video_body(v, img, seed0 + i)
        if runway.cached_output("image_to_video", vb, ".mp4") is None:
            cost += runway.price(vb)
        else:
            clips_cached += 1
    else:
        return {"cost": cost, "cached_frames": cached, "cached_clips": clips_cached}
    return {
        "cost": cost + (n - 1) * cost_per_clip(False),
        "cached_frames": 0,
        "cached_clips": 0,
    }


FIRST_RUN_TASK = "put the red block in the bowl"


def seed_for(task: Task) -> int:
    """Runway seeds come from the sentence, so a new sentence gets new footage even when it parses to the same
    block, colour and container (the prompts are built from those). The first run's sentence keeps seed 11, the
    seed its cached clips were made with, so re-running it still spends nothing."""
    t = " ".join(task.text.lower().split())
    if t == FIRST_RUN_TASK:
        return 11
    return 11 + int(hashlib.sha256(t.encode()).hexdigest(), 16) % 100_000


def generate(
    task: Task,
    run_dir: Path,
    n: int,
    seed0: int | None = None,
    emit=None,
    parallel: bool = False,
    fixes: dict[str, list[Mismatch]] | None = None,
    start: int = 0,
    router: str | None = None,
) -> list[dict]:
    """Generate (or reuse) n clips; copies land in run_dir/clips/<id>.mp4. Stops cleanly at the credit floor.

    parallel: after clip 1's first frame (the base every restyle refers to), the other clips run concurrently,
    each first frame then its video. Same requests, same seeds, same cache keys as the sequential path.
    fixes: clip id -> mismatches from the last take's rejections (plan.refine_fixes); rewrites those prompts.
    start: add n NEW clips to a run that already has footage, as variants start .. start+n-1 (ids v{start+1}..),
    each restyled from the run's own base frame (clips/v01.png); no earlier clip is regenerated.
    router: a Runway Model Router config slug. The videos go through POST /v1/generate/video with that router
    (free dry run first, then live) instead of the fixed gen4_turbo; clips/<id>.routing.json records the pick.
    emit(event) receives live progress: {"type": "runway", clip, stage, model, status, progress, cost}, then
    {"type": "frame_ready"} and {"type": "clip_ready"} with the file names."""
    emit = emit or (lambda ev: None)
    if seed0 is None:
        seed0 = seed_for(task)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    vs = variants(task, start + n, fixes)

    def status(cid: str, stage: str, model: str):
        return lambda st: emit(
            {"type": "runway", "clip": cid, "stage": stage, "model": model, **st}
        )

    def one(i: int, v: Variant, base: Path | None) -> tuple[dict, Path]:
        im = "gen4_image" if base is None else "gen4_image_turbo"
        img = first_frame(v, base, seed0 + i, status(v.clip_id, "frame", im))
        shutil.copy(img, clips_dir / f"{v.clip_id}.png")
        emit(
            {"type": "frame_ready", "clip": v.clip_id, "file": f"clips/{v.clip_id}.png"}
        )
        if router:
            vid, routing = route(v, img, router, status(v.clip_id, "video", f"router:{router}"))
            rec = _record(v, i, img, vid, clips_dir, routing)
            emit({"type": "routed", "clip": v.clip_id, **{k: routing.get(k) for k in (
                "router", "model", "provider", "estimated_credits", "realized_credits", "seconds", "cached")}})
        else:
            vid = animate(v, img, seed0 + i, status(v.clip_id, "video", "gen4_turbo"))
            rec = _record(v, i, img, vid, clips_dir)
        emit({"type": "clip_ready", "clip": v.clip_id, "file": rec["video"]})
        print(f"[generate] {v.clip_id} ready", flush=True)
        return rec, img

    out: list[dict] = []
    if start:
        base = base_frame(run_dir)
        if not base.exists():
            raise FileNotFoundError(f"{base}: a run needs clip 1's first frame before clips can be added to it")
        with ThreadPoolExecutor(max_workers=max(1, n)) as pool:
            futs = [pool.submit(one, i, vs[i], base) for i in range(start, start + n)]
            for f in futs:
                try:
                    out.append(f.result()[0])
                except runway.BudgetError as e:
                    print(f"[generate] skipped: {e}")
                    emit({"type": "budget_stop", "message": str(e)})
        return out
    if not parallel:
        base = None
        for i, v in enumerate(vs):
            try:
                rec, img = one(i, v, base)
            except runway.BudgetError as e:
                print(f"[generate] stopping: {e}")
                emit({"type": "budget_stop", "clip": v.clip_id, "message": str(e)})
                break
            base = base or img
            out.append(rec)
        return out

    try:
        rec, base = one(0, vs[0], None)
    except runway.BudgetError as e:
        emit({"type": "budget_stop", "clip": vs[0].clip_id, "message": str(e)})
        return out
    out.append(rec)
    with ThreadPoolExecutor(max_workers=max(1, n - 1)) as pool:
        futs = [pool.submit(one, i, v, base) for i, v in enumerate(vs) if i > 0]
        for f in futs:
            try:
                out.append(f.result()[0])
            except runway.BudgetError as e:
                print(f"[generate] skipped: {e}")
                emit({"type": "budget_stop", "message": str(e)})
    return out
