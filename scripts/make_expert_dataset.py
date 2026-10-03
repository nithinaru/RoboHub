"""Benchmark arm "expert": the aug25 dataset's 74 episodes, re-recorded by the scripted expert.

Same scene (task "pick"), same cube start positions and yaw as data/probes/aug25, same dataset writer
(rohub.dataset.write: same features, fps, 240x320 images, cameras front+wrist, same per-frame task string),
so the two roots are interchangeable for lerobot-train. The only thing that changes is who produced the motion:
the Runway-video retargeter (aug25) or pick_expert.run_episode (this script). See docs/BENCHMARK.md.

Start conditions. aug25 stores each episode's cube_xy rounded to 0.1 mm in rohub_episodes.json. The exact
positions are recovered by re-drawing scripts/augment_dataset.py's RNG (seed and per-clip count from
augment.json, stages.JITTER, the accepted clips' results/<clip>.json in sorted order) and matching each rounded
value; if that fails for an episode the rounded value is used and logged. Yaw is 0 for every aug25 episode
(dataset.write calls retarget.replay with its default cube_yaw=0.0) and for the vla.py eval.

Acceptance = the eval's rule: the cube was lifted and ends resting in the bowl. A failed start is retried at the
same position nudged by up to +-2 mm (rng seeded per episode and attempt, --retries times); if every retry fails,
the episode is replaced by a fresh draw from the same clip's jitter region. Everything is logged in
expert_episodes.json so the episode count always equals aug25's.

    MUJOCO_GL=egl PYTHONPATH=src python scripts/make_expert_dataset.py --out data/probes/expert74
    PYTHONPATH=src python scripts/make_expert_dataset.py --limit 2 --out /tmp/expert_smoke   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rohub import dataset as DS  # noqa: E402
from rohub import pick_expert as PE  # noqa: E402
from rohub.scene import load  # noqa: E402
from rohub.stages import JITTER  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "local/rohub_expert"
NUDGE = 0.002  # metres, retry perturbation of a failed start


@dataclass
class Start:
    episode_index: int
    clip: str
    kind: str
    cube_xy: np.ndarray
    yaw: float
    exact: bool  # recovered from the augment RNG (else the 0.1 mm rounded value)
    clip_origin: np.ndarray  # the clip's own cube start (centre of its jitter region)


@dataclass
class _Rec:
    """What dataset.write reads from a replay result."""

    success: bool
    states: list


def recover_starts(aug_root: Path, run_dir: Path) -> list[Start]:
    meta = json.loads((aug_root / "rohub_episodes.json").read_text())["episodes"]
    aug = json.loads((aug_root / "augment.json").read_text())
    rng = np.random.default_rng(aug["seed"])
    draws: dict[str, list[np.ndarray]] = {}
    origin: dict[str, np.ndarray] = {}
    for p in sorted(
        p
        for p in (run_dir / "results").glob("*.json")
        if json.loads(p.read_text())["accepted"]
    ):
        res = json.loads(p.read_text())
        cs = np.array(res["demo"]["cube_start"], float)[:2]
        origin[res["clip_id"]] = cs
        draws[res["clip_id"]] = [
            cs + rng.uniform(-np.array(JITTER), JITTER) for _ in range(aug["per_clip"])
        ]
    starts = []
    for e in meta:
        rounded = np.array(e["cube_xy"], float)
        cands = (
            [origin[e["clip"]]]
            if e["kind"] == "retargeted clip"
            else draws.get(e["clip"], [])
        )
        hit = [c for c in cands if np.allclose(c.round(4), rounded, atol=1e-9)]
        exact = len(hit) == 1
        starts.append(
            Start(
                e["episode_index"],
                e["clip"],
                e["kind"],
                hit[0] if exact else rounded,
                0.0,
                exact,
                origin.get(e["clip"], rounded),
            )
        )
    return starts


def run_expert(scene, xy: np.ndarray, yaw: float, on_frame=None) -> PE.EpisodeResult:
    """pick_expert.run_episode from a fixed start (its start sampler is swapped for this one call)."""
    orig = PE.sample_start
    PE.sample_start = lambda rng, spot: (np.asarray(xy, float).copy(), float(yaw))
    try:
        return PE.run_episode(scene, 0, np.random.default_rng(0), on_frame=on_frame)
    finally:
        PE.sample_start = orig


def accepted(r: PE.EpisodeResult) -> bool:
    return bool(r.success and r.lifted)


def aug25_stats(aug_root: Path) -> dict:
    meta = json.loads((aug_root / "rohub_episodes.json").read_text())
    fr = [e["frames"] for e in meta["episodes"]]
    return {
        "episodes": len(fr),
        "frames": int(sum(fr)),
        "mean_len": round(float(np.mean(fr)), 1),
        "min_len": int(min(fr)),
        "max_len": int(max(fr)),
    }


def aug25_task(aug_root: Path) -> str:
    import pandas as pd

    t = pd.read_parquet(aug_root / "meta" / "tasks.parquet")
    names = list(t.index) if "task" not in t.columns else list(t["task"])
    assert len(names) == 1, names
    return str(names[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data/probes/expert74")
    ap.add_argument(
        "--aug-root",
        type=Path,
        default=REPO / "data/probes/aug25",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=REPO / "data/web-runs/put-the-red-block-in-the-bowl",
    )
    ap.add_argument("--cameras", default="front,wrist")
    ap.add_argument(
        "--limit", type=int, default=0, help="only the first N episodes (smoke test)"
    )
    ap.add_argument("--retries", type=int, default=5)
    a = ap.parse_args()

    if a.out.exists():
        sys.exit(f"{a.out} exists; move it away first (this script never deletes)")
    task = (a.run_dir / "task.txt").read_text().strip()
    stored = aug25_task(a.aug_root)
    assert task == stored, f"task.txt {task!r} != aug25 tasks.parquet {stored!r}"

    starts = recover_starts(a.aug_root, a.run_dir)
    if a.limit:
        starts = starts[: a.limit]
    scene = load(task="pick")
    t0 = time.time()

    # pass 1: find an accepted start for every aug25 episode (no rendering)
    plan, log = [], []
    fails = retries = replaced = 0
    for s in starts:
        xy, how, tries = s.cube_xy, "same start", []
        r = run_expert(scene, xy, s.yaw)
        tries.append(
            {"xy": xy.round(5).tolist(), "success": r.success, "lifted": r.lifted}
        )
        if not accepted(r):
            fails += 1
            ok = False
            for k in range(a.retries):
                retries += 1
                g = np.random.default_rng(1_000_000 + 100 * s.episode_index + k)
                xy = s.cube_xy + g.uniform(-NUDGE, NUDGE, 2)
                r = run_expert(scene, xy, s.yaw)
                tries.append(
                    {
                        "xy": xy.round(5).tolist(),
                        "success": r.success,
                        "lifted": r.lifted,
                    }
                )
                if accepted(r):
                    ok, how = True, f"nudged (retry {k + 1})"
                    break
            k = 0
            while not ok:
                replaced += 1
                g = np.random.default_rng(2_000_000 + 100 * s.episode_index + k)
                base = s.clip_origin
                xy = base + g.uniform(-np.array(JITTER), JITTER)
                r = run_expert(scene, xy, s.yaw)
                tries.append(
                    {
                        "xy": xy.round(5).tolist(),
                        "success": r.success,
                        "lifted": r.lifted,
                    }
                )
                ok, how, k = accepted(r), "replaced by a fresh jitter draw", k + 1
        plan.append((s, xy))
        log.append(
            {
                "episode_index": s.episode_index,
                "clip": s.clip,
                "aug25_kind": s.kind,
                "aug25_cube_xy": s.cube_xy.round(6).tolist(),
                "exact_start": s.exact,
                "yaw": s.yaw,
                "expert_cube_xy": np.asarray(xy).round(6).tolist(),
                "outcome": how,
                "frames": r.frames,
                "attempts": tries,
            }
        )
        print(
            f"[expert] ep {s.episode_index} {s.clip} {how} frames {r.frames}",
            flush=True,
        )

    # pass 2: record with the unchanged dataset writer; its replay() call is routed to the expert
    def fake_replay(scene_, spec, on_frame=None, **_):
        states: list = []

        def cb(st, ac):
            states.append(st)
            on_frame(st, ac)

        r = run_expert(scene_, spec["xy"], spec["yaw"], on_frame=cb)
        return _Rec(accepted(r), states)

    DS.replay = fake_replay
    episodes = [
        {
            "demo": {"xy": xy, "yaw": s.yaw},
            "origin": {
                "clip": s.clip,
                "kind": "scripted expert",
                "aug25_kind": s.kind,
                "cube_xy": np.asarray(xy).round(4).tolist(),
            },
        }
        for s, xy in plan
    ]
    winfo = DS.write(
        scene,
        episodes,
        a.out,
        REPO_ID,
        task,
        cameras=tuple(a.cameras.split(",")),
        on_episode=lambda i, n, o, f: print(
            f"[dataset] {i + 1}/{n} {f} frames", flush=True
        ),
    )

    meta = json.loads((a.out / "rohub_episodes.json").read_text())["episodes"]
    fr = [e["frames"] for e in meta]
    ref = aug25_stats(a.aug_root)
    n = len(fr)
    ref_sub = [
        e["frames"]
        for e in json.loads((a.aug_root / "rohub_episodes.json").read_text())[
            "episodes"
        ][:n]
    ]
    summary = {
        "out": str(a.out),
        "repo_id": REPO_ID,
        "task": task,
        "cameras": a.cameras.split(","),
        "episodes": n,
        "frames": int(sum(fr)),
        "mean_len": round(float(np.mean(fr)), 1),
        "min_len": int(min(fr)),
        "max_len": int(max(fr)),
        "exact_starts": sum(s.exact for s, _ in plan),
        "first_try_failures": fails,
        "retries_run": retries,
        "replaced": replaced,
        "aug25_all": ref,
        "aug25_same_episodes": {
            "episodes": n,
            "frames": int(sum(ref_sub)),
            "mean_len": round(float(np.mean(ref_sub)), 1),
        },
        "writer_seconds": winfo["seconds"],
        "seconds": round(time.time() - t0, 1),
    }
    (a.out / "expert_episodes.json").write_text(
        json.dumps({"summary": summary, "episodes": log}, indent=1)
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
