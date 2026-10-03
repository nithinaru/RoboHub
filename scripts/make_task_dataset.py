"""Scripted-expert LeRobot dataset for the push or stack task (same writer, features, fps, 240x320 images and
cameras front+wrist as aug25). Source is the scripted expert, not Runway footage; every episode says so.

    PYTHONPATH=src .venv/bin/python scripts/make_task_dataset.py push --n 30 --out data/probes/push30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rohub import dataset as DS  # noqa: E402
from rohub.scene import load  # noqa: E402


class _Rec:
    def __init__(self, success, states):
        self.success, self.states = success, states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["push", "stack", "tower"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.task == "push":
        from rohub import expert as E
        text = "Push the red block into the blue taped square."
    elif a.task == "stack":
        from rohub import stack_expert as E
        text = "Stack the red block on the blue block."
    else:
        from rohub import tower_expert as E
        text = "Stack three blocks into a tower."
    scene = load(task=a.task)
    rng = np.random.default_rng(a.seed)
    plan, tries = [], 0
    while len(plan) < a.n:
        spot = tries % len(E.SPOTS)
        st = rng.bit_generator.state
        r = E.run_episode(scene, spot, rng)
        tries += 1
        if r.success:
            plan.append((spot, st, r.cube_start))
        print(f"[expert] try {tries} spot {spot} success {r.success} kept {len(plan)}/{a.n}", flush=True)

    def fake_replay(scene_, spec, on_frame=None, **_):
        states = []

        def cb(s, ac):
            states.append(s)
            on_frame(s, ac)

        g = np.random.default_rng()
        g.bit_generator.state = spec["state"]
        r = E.run_episode(scene_, spec["spot"], g, on_frame=cb)
        return _Rec(r.success, states)

    DS.replay = fake_replay
    eps = [{"demo": {"spot": s, "state": st},
            "origin": {"clip": "expert", "kind": "scripted expert", "spot": s,
                       "cube_xy": np.asarray(cs[:2]).round(4).tolist(), "yaw": round(float(cs[2]), 4)}}
           for s, st, cs in plan]
    info = DS.write(scene, eps, a.out, f"local/rohub_{a.task}", text, cameras=("front", "wrist"),
                    on_episode=lambda i, n, o, f: print(f"[dataset] {i + 1}/{n} {f} frames", flush=True))
    info["expert_tries"] = tries
    info["source"] = "expert"
    (a.out / "task_dataset.json").write_text(json.dumps(info, indent=1))
    print("[done]", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
