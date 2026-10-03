"""The pipeline as three stages the site (and the CLI) call, each reporting live progress through emit(event).

  footage(task, run_dir, n)            Runway: first frames, then 5 s videos          (generate.py)
  training_data(task, run_dir)         track, audit, gate, retarget, re-anchor, write LeRobot v3.0
  train_and_eval(run_dir, clip_ids)    train the policy on the episodes of the chosen clips, evaluate in MuJoCo

Every event is a plain dict with a "type". Nothing here is simulated for show: the events are emitted by the
code that does the work, at the moment it does it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from . import gates as G
from .plan import Task

JITTER = (
    0.03,
    0.02,
)  # re-anchor / eval region around each accepted clip's cube start (x, y metres)


def _noop(ev: dict) -> None:
    pass


def cpu_name() -> str:
    try:
        return (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or platform.processor()
        )
    except OSError:
        return platform.processor()


def footage(
    task: Task, run_dir: Path, n: int, emit=_noop, parallel: bool = True, fixes=None, start: int = 0,
    router: str | None = None,
) -> list[dict]:
    """start > 0: add n new clips to a run that already has footage (generate.next_index).
    router: a Runway Model Router slug; the videos go through it instead of the fixed gen4_turbo."""
    from .generate import generate

    return generate(task, run_dir, n, emit=emit, parallel=parallel, fixes=fixes, start=start, router=router)


def training_data(
    task: Task, run_dir: Path, reanchor_n: int = 15, emit=_noop, log=print
) -> dict:
    """Clips on disk -> per-clip verdicts -> episodes -> LeRobot dataset. Writes run_dir/data.json."""
    from .dataset import load_arrays, write
    from .media import SimFilm
    from .pipeline import process_clip
    from .retarget import reanchor, replay
    from .scene import load

    t0 = time.time()
    for sub in ("pose", "results", "media"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    clips = sorted(run_dir.glob("clips/*.mp4"))
    emit({"type": "data_start", "clips": [c.stem for c in clips]})
    scene = load(task="pick")
    per_clip = []
    for c in clips:
        meta_p = c.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        meta.setdefault("clip_id", c.stem)
        emit({"type": "clip_stage", "clip": c.stem, "stage": "track"})
        out = process_clip(c, meta, run_dir, scene, emit=emit)
        r = out["res"]
        failed = [g for g in r["gates"] if g["passed"] is False]
        emit(
            {
                "type": "verdict",
                "clip": r["clip_id"],
                "accepted": r["accepted"],
                "reason": r["reason"],
                "gates_total": len(r["gates"]),
                "gates_passed": sum(1 for g in r["gates"] if g["passed"] is True),
                "failed": [g["id"] for g in failed],
                "gates": [
                    {k: g[k] for k in ("id", "name", "passed", "value", "limit", "why")}
                    for g in r["gates"]
                ],
                "demo": {k: r["demo"][k] for k in ("lift_m", "carry_m", "m_per_px")}
                if "demo" in r
                else None,
            }
        )
        log(
            f"[gates] {r['clip_id']}: {'ACCEPTED' if r['accepted'] else 'REJECTED'} ({r['reason']})"
        )
        per_clip.append(out)
    accepted = [p for p in per_clip if p["res"]["accepted"]]
    summary: dict = {
        "task": task.text,
        "clips": [
            {k: p["res"][k] for k in ("clip_id", "accepted", "reason")}
            | {"failed": [g["id"] for g in p["res"]["gates"] if g["passed"] is False]}
            for p in per_clip
        ],
    }
    if not accepted:
        log("[run] no clip passed the gates; nothing to train on")
        summary["dataset"] = None
        (run_dir / "data.json").write_text(json.dumps(summary, indent=1))
        emit({"type": "data_done", "accepted": [], "episodes": 0})
        return summary

    # the SO-101 performing each accepted clip, physics replay, for the site
    for p in accepted:
        cid = p["res"]["clip_id"]
        emit({"type": "clip_stage", "clip": cid, "stage": "retarget"})
        film = SimFilm(scene, run_dir / "media" / f"{cid}_robot.mp4")
        replay(scene, p["demo"], on_frame=film.frame)
        film.close()
        emit({"type": "robot_film", "clip": cid, "file": f"media/{cid}_robot.mp4"})

    # each accepted clip as retargeted, plus re-anchored copies, every one gated again in the sim
    rng = np.random.default_rng(1000)
    episodes, re_stats = [], {"tried": 0, "accepted": 0, "reasons": {}}
    for p in accepted:
        cid, demo = p["res"]["clip_id"], p["demo"]
        episodes.append(
            {
                "demo": demo,
                "origin": {
                    "clip": cid,
                    "kind": "retargeted clip",
                    "cube_xy": demo.cube_start[:2].round(4).tolist(),
                },
            }
        )
        ok_n = 0
        for _ in range(reanchor_n):
            xy = demo.cube_start[:2] + rng.uniform(-np.array(JITTER), JITTER)
            d2 = reanchor(demo, xy)
            ok, why = G.verdict(G.robot_gates(replay(scene, d2)))
            re_stats["tried"] += 1
            if ok:
                ok_n += 1
                re_stats["accepted"] += 1
                episodes.append(
                    {
                        "demo": d2,
                        "origin": {
                            "clip": cid,
                            "kind": "re-anchored",
                            "cube_xy": xy.round(4).tolist(),
                        },
                    }
                )
            else:
                key = why.split(":")[0]
                re_stats["reasons"][key] = re_stats["reasons"].get(key, 0) + 1
        emit({"type": "reanchor", "clip": cid, "tried": reanchor_n, "accepted": ok_n})
    log(
        f"[episodes] {len(episodes)} episodes ({re_stats['accepted']}/{re_stats['tried']} re-anchored copies passed)"
    )
    summary["reanchor"] = re_stats
    emit({"type": "episodes", "total": len(episodes), "reanchor": re_stats})

    repo_id = "local/rohub_" + run_dir.name.replace("-", "_")[:40]
    ds_root = run_dir / "lerobot"
    info = write(
        scene,
        episodes,
        ds_root,
        repo_id,
        task.text,
        on_episode=lambda i, n, origin, frames: emit(
            {
                "type": "dataset_progress",
                "episode": i + 1,
                "total": n,
                "clip": origin["clip"],
                "kind": origin["kind"],
                "frames": frames,
            }
        ),
    )
    arr = load_arrays(ds_root, repo_id)
    info["loaded_back"] = {
        "episodes": arr["num_episodes"],
        "frames": int(len(arr["state"])),
        "codebase_version": arr["codebase_version"],
    }
    log(f"[dataset] {info}")
    summary["dataset"] = info
    summary["seconds"] = round(time.time() - t0, 1)
    (run_dir / "data.json").write_text(json.dumps(summary, indent=1, default=float))
    emit(
        {
            "type": "data_done",
            "accepted": [p["res"]["clip_id"] for p in accepted],
            "episodes": info["episodes"],
            "frames": info["frames"],
            "codebase_version": info["codebase_version"],
            "loaded_back": info["loaded_back"],
            "seconds": summary["seconds"],
        }
    )
    return summary


def _anchor(run_dir: Path, cid: str) -> np.ndarray:
    """The exact (unrounded) cube start of an accepted clip, recomputed from its landmarks the way the data stage
    did (results/<clip>.json stores it rounded)."""
    from .track import find_events, load_track, to_robot

    res = json.loads((run_dir / "results" / f"{cid}.json").read_text())
    tr = load_track(run_dir / "pose" / f"{cid}.json")
    return to_robot(tr, find_events(tr), res["bowl_u"]).cube_start[:2]


def model_card(info: dict) -> dict:
    from .policy import CHUNK, ENSEMBLE_M, HIDDEN, MLPChunk

    m = MLPChunk()
    layers = [
        [m2.in_features, m2.out_features] for m2 in m.net if hasattr(m2, "in_features")
    ]
    return {
        "type": "MLP with action chunking and ACT-style temporal ensembling",
        "layers": layers,
        "hidden": HIDDEN,
        "chunk": CHUNK,
        "ensemble_m": ENSEMBLE_M,
        "params": info["params"],
        "inputs": "6 joint angles + the cube's xyz from the simulator (state-based, no camera)",
        "outputs": f"the next {CHUNK} joint targets for all 6 joints",
        "optimizer": "AdamW, lr 1e-3, cosine, batch 256, L1 loss",
        "steps": info["steps"],
        "samples": info["samples"],
        "final_l1": info["final_l1"],
        "train_seconds": info["train_seconds"],
        "device": cpu_name() + " CPU",
    }


def train_and_eval(
    run_dir: Path,
    clip_ids: list[str],
    steps: int = 4000,
    eval_seeds: int = 50,
    emit=_noop,
    log=print,
    film_cam: str = "hero",
) -> dict:
    """Train on the episodes whose origin is one of clip_ids (the clips dropped on the trainer), then roll the
    policy out on unseen cube positions around those clips. Writes run_dir/policy/ and run_dir/train.json."""
    from .dataset import load_arrays
    from .media import SimFilm, hero_cam
    from .policy import Runner, rollout, train, wilson
    from .scene import load

    # clip refs: "v01" is a clip of this run, "<slug>/v01" a clip of another run under the same runs folder
    # (footage that passed the gates earlier). Each run's LeRobot dataset is read back and filtered to the
    # episodes of the chosen clips; episode ids are offset per run so they stay distinct.
    refs = [(run_dir.parent / c.split("/")[0], c.split("/")[1]) if "/" in c else (run_dir, c) for c in clip_ids]
    parts, used, anchors = [], [], []
    for k, src in enumerate(dict.fromkeys(d for d, _ in refs)):
        ids = [c for d, c in refs if d == src]
        ds_root = src / "lerobot"
        meta = json.loads((ds_root / "rohub_episodes.json").read_text())
        keep = sorted({e["episode_index"] for e in meta["episodes"] if e["clip"] in ids})
        if not keep:
            continue
        arr = load_arrays(ds_root, meta["info"]["repo_id"])
        mask = np.isin(arr["episode"], keep)
        parts.append({key: arr[key][mask] for key in ("state", "action", "env", "episode", "frame")})
        parts[-1]["episode"] = parts[-1]["episode"] + 100_000 * k
        for c in sorted({e["clip"] for e in meta["episodes"] if e["episode_index"] in keep}):
            used.append(c if src == run_dir else f"{src.name}/{c}")
            anchors.append(_anchor(src, c))
    if not parts:
        raise ValueError(f"no training episodes come from {clip_ids}; only clips that passed the gates can train")
    sub = {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}
    n_eps = len(np.unique(sub["episode"]))
    n_frames = int(len(sub["state"]))
    emit({"type": "train_start", "clips": used, "episodes": n_eps, "frames": n_frames, "steps": steps})

    tinfo = train(
        sub,
        run_dir / "policy",
        steps=steps,
        log=log,
        on_step=lambda s, l1, sec: emit(
            {
                "type": "train_step",
                "step": s,
                "steps": steps,
                "l1": round(l1, 4),
                "seconds": round(sec, 1),
            }
        ),
    )
    card = model_card(tinfo) | {"episodes": n_eps, "frames": n_frames, "clips": used}
    emit({"type": "trained", "card": card})

    scene = load(task="pick")
    runner = Runner(run_dir / "policy")
    results = []
    for s in range(eval_seeds):
        r = np.random.default_rng(10_000 + s)
        xy = anchors[s % len(anchors)] + r.uniform(-np.array(JITTER), JITTER)
        out = rollout(scene, runner, xy)
        results.append(
            {"seed": 10_000 + s, "cube_xy": xy.round(4).tolist(), "xy": xy, **out}
        )
        emit(
            {
                "type": "eval",
                "i": s + 1,
                "n": eval_seeds,
                "seed": 10_000 + s,
                "success": out["success"],
                "cube_xy": xy.round(4).tolist(),
            }
        )
    k = sum(r["success"] for r in results)
    lo, hi = wilson(k, len(results))
    log(f"[eval] policy success {k}/{len(results)} (Wilson 95% {lo:.2f}-{hi:.2f})")
    ev = {
        "seeds": len(results),
        "successes": k,
        "rate": round(k / len(results), 3),
        "wilson95": [round(lo, 3), round(hi, 3)],
    }
    emit({"type": "eval_done", **ev})

    films = {}
    for tag, pick in (
        ("success", [r for r in results if r["success"]]),
        ("failure", [r for r in results if not r["success"]]),
    ):
        if film_cam == "hero" and tag == "failure":
            continue  # the site shows the success film; a 1080p failure film costs ~35 s for nothing on screen
        if not pick:
            continue
        if film_cam == "hero":
            name, cam, wh = f"media/policy_{tag}_hero.mp4", hero_cam(), (1920, 1080)
        else:
            name, cam, wh = f"media/policy_{tag}.mp4", "front", (640, 480)
        film = SimFilm(scene, run_dir / name, w=wh[0], h=wh[1], camera=cam)
        again = rollout(scene, runner, pick[0]["xy"], on_frame=film.frame)
        film.close()
        # the film shows exactly what the evaluation scored: same start, same outcome, same length
        assert (
            again["success"] == pick[0]["success"]
            and again["frames"] == pick[0]["frames"]
        )
        films[tag] = {
            "file": name,
            "seed": pick[0]["seed"],
            "cube_xy": pick[0]["cube_xy"],
            "frames": again["frames"],
        }
        emit({"type": "rollout_film", "tag": tag, **films[tag]})

    summary = {
        "clips": used,
        "train": tinfo,
        "card": card,
        "eval": ev
        | {"episodes": [{k2: v for k2, v in r.items() if k2 != "xy"} for r in results]},
        "films": films,
    }
    (run_dir / "train.json").write_text(json.dumps(summary, indent=1, default=float))
    emit({"type": "train_done", "eval": ev, "films": films})
    return summary
