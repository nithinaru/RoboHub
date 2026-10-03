"""v8 research probe (0 Runway credits): MimicGen-style augmentation of the ACCEPTED clips, in simulation only.

The same 4 accepted human demonstrations (the run's results/<clip>.json), each re-anchored to N new cube positions
drawn from the same region the site uses (stages.JITTER around the clip's own cube start), every copy gated again by
the unchanged robot gates in a MuJoCo replay, then written as a LeRobot v3.0 dataset. Optionally also records the
SO-101's wrist camera. Positions come from a different RNG seed than the site's dataset (1000) and the eval
(10000-10049), so no copy sits on an eval position.

    PYTHONPATH=src .venv/bin/python scripts/augment_dataset.py data/web-runs/put-the-red-block-in-the-bowl \
        --per-clip 50 --cameras front,wrist --out data/probes/aug50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rohub import gates as G  # noqa: E402
from rohub.dataset import write  # noqa: E402
from rohub.retarget import reanchor, replay  # noqa: E402
from rohub.scene import load  # noqa: E402
from rohub.stages import JITTER  # noqa: E402
from rohub.track import Demo  # noqa: E402


def demo_from_result(res: dict) -> Demo:
    """Rebuild the retargeted demonstration the site stored for an accepted clip (xyz rounded to 0.1 mm)."""
    d = res["demo"]
    closed = np.array(d["closed"], bool)
    gi = int(np.argmax(closed))
    ri = gi + int(np.argmax(~closed[gi:]))
    xyz = np.array(d["xyz"], float)
    return Demo(
        fps=float(res["fps"]),
        start=0,
        end=len(xyz) - 1,
        xyz=xyz,
        closed=closed,
        cube_start=np.array(d["cube_start"], float),
        m_per_px=float(d["m_per_px"]),
        grasp_i=gi,
        release_i=ri,
        lift_height=float(d["lift_m"]),
        carry_dist=float(d["carry_m"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--per-clip",
        type=int,
        default=50,
        help="re-anchored copies tried per accepted clip",
    )
    ap.add_argument("--cameras", default="front")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip-ids", default="", help="only these accepted clips (comma separated), e.g. one router's")
    ap.add_argument("--repo-id", default="local/rohub_aug")
    a = ap.parse_args()
    only = set(filter(None, a.clip_ids.split(",")))

    scene = load(task="pick")
    rng = np.random.default_rng(a.seed)
    task = (a.run_dir / "task.txt").read_text().strip()
    accepted = sorted(
        p
        for p in (a.run_dir / "results").glob("*.json")
        if json.loads(p.read_text())["accepted"] and (not only or p.stem in only)
    )
    episodes, stats, t0 = [], {}, time.time()
    for p in accepted:
        res = json.loads(p.read_text())
        cid, demo = res["clip_id"], demo_from_result(res)
        rep = replay(scene, demo)
        ok, why = G.verdict(G.robot_gates(rep))
        assert ok and len(rep.states) == res["replay"]["frames"], (
            cid,
            why,
            len(rep.states),
            res["replay"],
        )
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
        n_ok, reasons = 0, {}
        for _ in range(a.per_clip):
            xy = demo.cube_start[:2] + rng.uniform(-np.array(JITTER), JITTER)
            d2 = reanchor(demo, xy)
            ok, why = G.verdict(G.robot_gates(replay(scene, d2)))
            if ok:
                n_ok += 1
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
                reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1
        stats[cid] = {"tried": a.per_clip, "accepted": n_ok, "rejected": reasons}
        print(
            f"[augment] {cid}: {n_ok}/{a.per_clip} re-anchored copies passed the robot gates {reasons}",
            flush=True,
        )
    print(
        f"[augment] {len(episodes)} episodes gated in {time.time() - t0:.0f} s; writing the dataset",
        flush=True,
    )
    cams = tuple(a.cameras.split(","))
    info = write(
        scene,
        episodes,
        a.out,
        a.repo_id,
        task,
        cameras=cams,
        on_episode=lambda i, n, o, f: (
            print(
                f"[dataset] {i + 1}/{n} {o['clip']} {o['kind']} {f} frames", flush=True
            )
            if (i + 1) % 10 == 0 or i + 1 == n
            else None
        ),
    )
    (a.out / "augment.json").write_text(
        json.dumps(
            {
                "per_clip": a.per_clip,
                "seed": a.seed,
                "cameras": cams,
                "clips": stats,
                "dataset": info,
            },
            indent=1,
        )
    )
    print(f"[augment] done: {info}", flush=True)


if __name__ == "__main__":
    main()
