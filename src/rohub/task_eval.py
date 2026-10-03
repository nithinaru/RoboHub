"""SmolVLA on the push or stack task in MuJoCo: unseen starts (seeds 20000+, never used for the dataset), the task's
own success rule, and one success film re-drawn from the evaluated episode's own states (hero camera, 1280x720).

    PYTHONPATH=src .venv/bin/python -m rohub.task_eval push <ckpt>/pretrained_model --seeds 10 --out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from .dataset import IMG_H, IMG_W
from .scene import CUBE_HALF, GRIPPER_CLOSED, HOME, STEPS_PER_FRAME, load
from .units import from_lerobot_units, to_lerobot_units
from .vla import VLARunner

TEXT = {"push": "Push the red block into the blue taped square.", "stack": "Stack the red block on the blue block.",
        "tower": "Stack three blocks into a tower."}


def start(task: str, seed: int):
    from . import expert as E

    rng = np.random.default_rng(seed)
    xy, yaw = E.sample_start(rng, seed % len(E.SPOTS))
    return xy, (yaw if task == "push" else 0.0)


def ok(task, scene) -> bool:
    c = scene.cube_pos()
    if task == "tower":
        from .tower_expert import tower_ok

        return tower_ok(scene)
    if task == "push":
        from .expert import success

        return success(c[:2]) and c[2] < CUBE_HALF + 0.005
    from .stack_expert import base_pos, on_base

    return on_base(c, base_pos(scene))


def rollout(task, scene, runner, renderer, seed, max_s=25.0):
    m, d = scene.model, scene.data
    if task == "tower":
        from . import tower_expert as TE

        max_s = 30.0
        xy, green = TE.sample_start(np.random.default_rng(seed))
        yaw = 0.0
        TE.reset(scene, xy, green)
    else:
        xy, yaw = start(task, seed)
        scene.reset(HOME, GRIPPER_CLOSED, xy, yaw)
    runner.reset()
    torch.manual_seed(seed)
    settled, traj = 0, []
    for t in range(int(max_s * 30)):
        imgs = {}
        for cam in ("front", "wrist"):
            renderer.update_scene(d, camera=cam)
            imgs[cam] = renderer.render()
        a = runner.act(to_lerobot_units(scene.joint_pos()), imgs, TEXT[task])
        d.ctrl[:] = from_lerobot_units(a)
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(m, d)
        traj.append(d.qpos.copy())
        still = np.linalg.norm(d.qvel[-6:-3]) < 1e-3 and (task != "tower" or np.linalg.norm(d.qvel[-12:-9]) < 1e-3)
        settled = settled + 1 if (ok(task, scene) and still) else 0
        if settled > 30:
            break
    return {"seed": seed, "cube_xy": np.round(xy, 4).tolist(), "yaw": round(float(yaw), 4), "success": ok(task, scene),
            "cube_end": scene.cube_pos().round(4).tolist(), "frames": t + 1}, np.array(traj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TEXT))
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stop-after-success", type=int, default=0, help="stop once this many successes are in (0: run all)")
    a = ap.parse_args()
    a.seeds = min(a.seeds, 5)  # hackathon timebox: 5 unseen starts
    a.out.mkdir(parents=True, exist_ok=True)
    scene = load(task=a.task)
    renderer = mujoco.Renderer(scene.model, IMG_H, IMG_W)
    runner = VLARunner(a.ckpt, a.device)
    eps, trajs, t0 = [], {}, time.time()
    for s in range(a.seeds):
        r, tr = rollout(a.task, scene, runner, renderer, 20_000 + s)
        eps.append(r)
        trajs[str(r["seed"])] = tr
        k = sum(e["success"] for e in eps)
        print(f"[eval] {a.task} seed {r['seed']} {'SUCCESS' if r['success'] else 'miss'} {k}/{len(eps)} {time.time() - t0:.0f} s", flush=True)
        ev = {"task": TEXT[a.task], "ckpt": str(a.ckpt), "successes": k, "seeds": len(eps), "cameras": "front,wrist",
              "seed_base": 20_000, "episodes": eps}
        (a.out / "eval.json").write_text(json.dumps(ev, indent=1, default=lambda o: o.item()))
        np.savez_compressed(a.out / "eval_traj.npz", **trajs)
        if a.stop_after_success and k >= a.stop_after_success:
            break


def film(task: str, eval_dir: Path, out_mp4: Path) -> dict:
    from .media import SimFilm, hero_cam

    ev = json.loads((eval_dir / "eval.json").read_text())
    pick = next(e for e in ev["episodes"] if e["success"])
    traj = np.load(eval_dir / "eval_traj.npz")[str(pick["seed"])]
    scene = load(task=task)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    f = SimFilm(scene, out_mp4, w=1280, h=720, camera=hero_cam())
    for q in traj:
        scene.data.qpos[:] = q
        mujoco.mj_forward(scene.model, scene.data)
        f.frame()
    for _ in range(20):  # a short hold on the final state
        f.frame()
    f.close()
    return pick


if __name__ == "__main__":
    main()
