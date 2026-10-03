"""Router experiment: which Runway Model Router gives the most physics-accepted robot demonstrations per credit?

  PYTHONPATH=src python scripts/router_bench.py put-the-red-block-in-the-bowl \
      --routers demo-cheap,demo-fast,demo-best --clips 6 --budget 2500

For each router it makes N new clips of the run (ids after the run's highest, so nothing is regenerated). Every
router gets the same N scenes: variant k's restyled first frame (gen4_image_turbo from the run's own clips/v01.png,
bought once, shared) and variant k's motion prompt. Only the router differs. The video goes through
POST /v1/generate/video with the router's config id: a free dry run first (model, estimated credits), then the
live task. Then the same footage -> pose -> Claude audit -> gates -> MuJoCo replay path the data stage runs.

Hard budget stop: before each live call the dry-run estimate is reserved against --budget (frames included); a
call that would cross it is skipped, never started. Writes data/web-runs/<slug>/router_bench.json as it goes.

  --stage generate   only the Runway calls
  --stage gates      only the pipeline on clips the bench already generated
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rohub import generate as GEN  # noqa: E402
from rohub import runway  # noqa: E402
from rohub.plan import parse_task, variants  # noqa: E402

LOCK_DIR = Path.home() / "helloworld" / ".heavy-lock"
USD_PER_CREDIT = 0.01


def log(msg: str) -> None:
    print(msg, flush=True)


def free_gib(p: Path) -> float:
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 2**30


class Budget:
    def __init__(self, cap: float):
        self.cap, self.reserved, self.lock = cap, 0.0, threading.Lock()

    def take(self, credits: float) -> bool:
        with self.lock:
            if self.reserved + credits > self.cap:
                return False
            self.reserved += credits
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self.lock:
            self.reserved += actual - reserved


def load_bench(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(path: Path, bench: dict) -> None:
    bench["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    bench["table"] = table(bench)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(bench, indent=1, default=float))
    tmp.replace(path)


def table(bench: dict) -> list[dict]:
    rows = []
    for r in bench.get("routers", []):
        cl = [
            c
            for c in bench.get("clips", [])
            if c["router"] == r and c.get("status") == "SUCCEEDED"
        ]
        judged = [c for c in cl if c.get("accepted") is not None]
        acc = sum(1 for c in judged if c["accepted"])
        cred = sum(float(c.get("realized_credits") or 0) for c in cl)
        secs = [
            c["seconds"]
            for c in cl
            if c.get("seconds") is not None and not c.get("cached")
        ]
        models = sorted({c["model"] for c in cl if c.get("model")})
        rows.append(
            {
                "router": r,
                "optimize_for": next(
                    (c.get("optimize_for") for c in cl if c.get("optimize_for")), None
                ),
                "models": models,
                "clips": len(cl),
                "judged": len(judged),
                "accepted": acc,
                "acceptance_rate": round(acc / len(judged), 3) if judged else None,
                "credits": round(cred, 1),
                "usd": round(cred * USD_PER_CREDIT, 2),
                "credits_per_accepted": round(cred / acc, 1) if acc else None,
                "accepted_per_1000_credits": round(1000 * acc / cred, 2)
                if cred and judged
                else None,
                "mean_generation_s": round(sum(secs) / len(secs), 1) if secs else None,
            }
        )
    scored = [x for x in rows if x["accepted_per_1000_credits"] is not None]
    if scored:
        best = max(scored, key=lambda x: x["accepted_per_1000_credits"])
        if best["accepted_per_1000_credits"] > 0:
            for x in rows:
                x["winner"] = x is best
    return rows


def generate_stage(a, run_dir: Path, bench_path: Path, bench: dict) -> None:
    task = parse_task((run_dir / "task.txt").read_text().strip())
    seed0 = GEN.seed_for(task)
    base = GEN.base_frame(run_dir)
    start = GEN.next_index(run_dir)
    routers = a.routers.split(",")
    vs = variants(task, start + a.clips)[start:]
    budget = Budget(a.budget)
    clips_dir = run_dir / "clips"
    bench.update(
        {
            "slug": run_dir.name,
            "task": task.text,
            "routers": routers,
            "clips_per_router": a.clips,
            "budget": a.budget,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "balance_start": runway.balance(),
            "variants": [v.clip_id for v in vs],
            "clips": [],
            "frames": [],
            "method": "same N scenes per router (shared restyled first frames + motion prompts); only the router "
            "differs; free dry run then live generate/video; then pose, Claude audit, 21 gates, MuJoCo replay",
        }
    )

    # 1. the shared first frames (direct gen4_image_turbo, 2 credits each, cached by request hash)
    frames: dict[str, Path] = {}
    for j, v in enumerate(vs):
        i = start + j
        fb = GEN.frame_body(v, base, seed0 + i)
        cost = (
            0 if runway.cached_output("text_to_image", fb, ".png") else runway.price(fb)
        )
        if not budget.take(cost):
            raise SystemExit(f"budget {a.budget} too small even for the first frames")
        frames[v.clip_id] = GEN.first_frame(v, base, seed0 + i)
        bench["frames"].append(
            {"variant": v.clip_id, "model": "gen4_image_turbo", "credits": cost}
        )
        log(f"[frames] {v.clip_id} first frame ({cost} cr)")
    save(bench_path, bench)

    # 2. clip ids, router by router: vNN after the run's highest
    jobs = []
    n = start
    for r in routers:
        for j, v in enumerate(vs):
            n += 1
            jobs.append((r, j, v, f"v{n:02d}"))
    for r, j, v, cid in jobs:
        bench["clips"].append(
            {"clip_id": cid, "router": r, "variant": v.clip_id, "status": "QUEUED"}
        )
    save(bench_path, bench)
    blk = threading.Lock()

    def upd(cid: str, **kw) -> None:
        with blk:
            c = next(c for c in bench["clips"] if c["clip_id"] == cid)
            c.update(kw)
            save(bench_path, bench)

    def one(job) -> None:
        r, j, v, cid = job
        i = start + j
        img = frames[v.clip_id]
        body = runway.router_body(r, v.video_prompt, img, duration=GEN.CLIP_SECONDS)
        hit = runway.cached_routed(body)
        dry = None if hit else runway.router_dry_run(body)
        est = 0.0 if hit else float(dry["estimatedCost"]["credits"])
        if not budget.take(est):
            upd(
                cid,
                status="SKIPPED_BUDGET",
                estimated_credits=est,
                model=dry.get("model") if dry else None,
            )
            log(
                f"[bench] {cid} {r}: skipped, {est:g} credits would cross the {a.budget} budget"
            )
            return
        upd(
            cid, status="RUNNING", estimated_credits=est, model=(dry or {}).get("model")
        )
        log(
            f"[bench] {cid} {r}: dry run picks {(dry or {}).get('model', 'cached')} at {est:g} cr"
        )
        try:
            vid, rec = GEN.route(
                v,
                img,
                r,
                label=f"{cid} bench via {r}",
                dry=dry,
                max_credits=a.budget - budget.reserved + est,
            )
        except (
            Exception
        ) as e:  # a failed task: reported in the bench, the rest continue
            budget.settle(est, 0)
            upd(cid, status="FAILED", error=str(e)[:300])
            log(f"[bench] {cid} {r}: FAILED {e}")
            return
        budget.settle(
            est, 0 if rec.get("cached") else float(rec.get("realized_credits") or est)
        )
        GEN._record(v, i, img, vid, clips_dir, rec, clip_id=cid)
        upd(
            cid,
            status="SUCCEEDED",
            **{
                k: rec.get(k)
                for k in (
                    "model",
                    "provider",
                    "optimize_for",
                    "price_ceiling",
                    "estimated_credits",
                    "realized_credits",
                    "seconds",
                    "cached",
                    "task_id",
                    "capacity_fallback",
                )
            },
        )
        log(
            f"[bench] {cid} {r}: {rec['model']} {rec.get('realized_credits')} cr in {rec.get('seconds')} s"
        )

    # interleave the routers so each one sees the same account load
    order = sorted(jobs, key=lambda x: (x[1], routers.index(x[0])))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(one, order))
    bench["balance_end"] = runway.balance()
    bench["credits_spent_balance"] = bench["balance_start"] - bench["balance_end"]
    bench["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(bench_path, bench)


class HeavyLock:
    def __enter__(self):
        while True:
            try:
                LOCK_DIR.mkdir()
                (LOCK_DIR / "owner").write_text(str(os.getpid()))
                return self
            except FileExistsError:
                try:
                    pid = int((LOCK_DIR / "owner").read_text().strip())
                    os.kill(pid, 0)
                except (OSError, ValueError):
                    shutil.rmtree(LOCK_DIR, ignore_errors=True)
                    continue
                log(f"[gates] waiting for the heavy-job lock (pid {pid})")
                time.sleep(10)

    def __exit__(self, *exc):
        shutil.rmtree(LOCK_DIR, ignore_errors=True)


def gates_stage(a, run_dir: Path, bench_path: Path, bench: dict) -> None:
    from rohub.audit import audit
    from rohub.pipeline import extract, process_clip
    from rohub.scene import load

    gib = free_gib(run_dir)
    if gib < 12 and not a.force_disk:
        raise SystemExit(
            f"[gates] {gib:.1f} GiB free, under the 12 GiB floor for evaluation steps; not starting"
        )
    for sub in ("pose", "results", "media"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    todo = [
        c
        for c in bench["clips"]
        if c.get("status") == "SUCCEEDED" and c.get("accepted") is None
    ]
    log(f"[gates] {len(todo)} clips")
    with HeavyLock():
        # pose extraction and the Claude audit are subprocesses and cached by clip hash: run them 3 at a time
        def pre(c):
            clip = run_dir / "clips" / f"{c['clip_id']}.mp4"
            extract(clip)
            from rohub.track import load_track

            tr = load_track(extract(clip))
            audit(clip, tr.width, tr.height)
            log(f"[gates] {c['clip_id']} tracked + audited")

        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in [pool.submit(pre, c) for c in todo]:
                try:
                    f.result()
                except Exception as e:
                    log(f"[gates] pre-pass error: {e}")
        scene = load(task="pick")
        for c in todo:
            clip = run_dir / "clips" / f"{c['clip_id']}.mp4"
            meta = json.loads(clip.with_suffix(".json").read_text())
            try:
                r = process_clip(clip, meta, run_dir, scene)["res"]
            except Exception as e:
                c.update(
                    accepted=False,
                    reason=f"pipeline error: {e}"[:200],
                    failed=["pipeline"],
                )
                save(bench_path, bench)
                continue
            c.update(
                accepted=bool(r["accepted"]),
                reason=r["reason"],
                failed=[g["id"] for g in r["gates"] if g["passed"] is False],
                gates_total=len(r["gates"]),
                gates_passed=sum(1 for g in r["gates"] if g["passed"] is True),
            )
            save(bench_path, bench)
            log(
                f"[gates] {c['clip_id']} {c['router']} {c['model']}: {'ACCEPTED' if r['accepted'] else 'REJECTED'} ({r['reason']})"
            )
    bench["judged"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(bench_path, bench)


def datasets_stage(a, run_dir: Path, bench_path: Path, bench: dict) -> None:
    """One LeRobot dataset per router from its physics-accepted clips, with the same re-anchoring as aug25
    (scripts/augment_dataset.py: 25 re-anchored copies per clip, each gated again, cameras front + wrist)."""
    import subprocess

    gib = free_gib(run_dir)
    if gib < 12 and not a.force_disk:
        raise SystemExit(
            f"[datasets] {gib:.1f} GiB free, under the 12 GiB floor; not starting"
        )
    bench.setdefault("datasets", {})
    with HeavyLock():
        for r in bench["routers"]:
            ids = [
                c["clip_id"]
                for c in bench["clips"]
                if c["router"] == r and c.get("accepted")
            ]
            out = REPO / "data" / "probes" / f"router-{r}"
            if not ids:
                bench["datasets"][r] = {
                    "path": None,
                    "clips": [],
                    "episodes": 0,
                    "frames": 0,
                    "note": "no physics-accepted clip",
                }
                save(bench_path, bench)
                log(f"[datasets] {r}: no accepted clip, no dataset")
                continue
            if out.exists():
                shutil.move(str(out), str(out) + f".old-{int(time.time())}")
            t0 = time.time()
            cmd = [
                "nice",
                "-n",
                "19",
                "taskpolicy",
                "-b",
                sys.executable,
                "-u",
                str(REPO / "scripts" / "augment_dataset.py"),
                str(run_dir),
                "--per-clip",
                str(a.per_clip),
                "--cameras",
                a.cameras,
                "--out",
                str(out),
                "--clip-ids",
                ",".join(ids),
                "--repo-id",
                f"local/rohub_router_{r.replace('-', '_')}",
            ]
            log(f"[datasets] {r}: {len(ids)} clips {ids} -> {out}")
            subprocess.run(
                cmd, check=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}
            )
            aug = json.loads((out / "augment.json").read_text())
            info = aug["dataset"]
            bench["datasets"][r] = {
                "path": str(out.relative_to(REPO)),
                "abs_path": str(out),
                "repo_id": f"local/rohub_router_{r.replace('-', '_')}",
                "clips": ids,
                "episodes": info.get("episodes"),
                "frames": info.get("frames"),
                "per_clip": a.per_clip,
                "cameras": a.cameras,
                "seconds": round(time.time() - t0, 1),
            }
            save(bench_path, bench)
            log(
                f"[datasets] {r}: {info.get('episodes')} episodes, {info.get('frames')} frames"
            )


def print_table(bench: dict) -> None:
    hdr = [
        "router",
        "model(s)",
        "clips",
        "accepted",
        "rate",
        "credits",
        "cr/accepted",
        "acc/1k cr",
        "mean gen s",
    ]
    log(" | ".join(hdr))
    for x in bench.get("table", []):
        log(
            " | ".join(
                str(v)
                for v in [
                    x["router"] + (" (winner)" if x.get("winner") else ""),
                    ",".join(x["models"]),
                    x["clips"],
                    x["accepted"],
                    x["acceptance_rate"],
                    x["credits"],
                    x["credits_per_accepted"],
                    x["accepted_per_1000_credits"],
                    x["mean_generation_s"],
                ]
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--routers", default="demo-cheap,demo-fast,demo-best")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument(
        "--budget",
        type=float,
        default=1e9,
        help="optional hard credit cap (frames included); default none",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--stage", choices=["all", "generate", "gates", "datasets"], default="all"
    )
    ap.add_argument(
        "--per-clip",
        type=int,
        default=25,
        help="re-anchored copies per accepted clip (aug25: 25)",
    )
    ap.add_argument("--cameras", default="front,wrist")
    ap.add_argument("--force-disk", action="store_true")
    a = ap.parse_args()
    run_dir = REPO / "data" / "web-runs" / a.slug
    bench_path = run_dir / "router_bench.json"
    bench = load_bench(bench_path)
    if a.stage in ("all", "generate"):
        if bench.get("clips"):
            raise SystemExit(
                f"{bench_path} already has a bench; use --stage gates, or move it away first"
            )
        generate_stage(a, run_dir, bench_path, bench)
    if a.stage in ("all", "gates"):
        gates_stage(a, run_dir, bench_path, bench)
    if a.stage in ("all", "datasets"):
        datasets_stage(a, run_dir, bench_path, bench)
    save(bench_path, bench)
    print_table(bench)


if __name__ == "__main__":
    main()
