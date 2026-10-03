"""SmolVLA (lerobot/smolvla_base, fine-tuned with lerobot-train on an RoboHub dataset) in the same MuJoCo harness
as the small MLP: the same scene, the same 50 unseen cube positions (seeds 10000-10049, the positions train.json
logged for the MLP), the same success rule (lifted, then resting in the bowl) and the same early stop.

Differences, by design of the two models: SmolVLA sees the dataset's camera (the sim's "front" camera, 320x240,
rendered live) plus the 6 joint angles and the task sentence; it does not get the cube position. The MLP gets
the cube position from the simulator and no camera.

    PYTHONPATH=src .venv/bin/python -m rohub.vla <checkpoint>/pretrained_model <run_dir> [--seeds 50]
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
from .pick_expert import in_bowl
from .scene import CUBE_HALF, GRIPPER_CLOSED, HOME, STEPS_PER_FRAME, Scene
from .units import from_lerobot_units, to_lerobot_units


class VLARunner:
    def __init__(
        self, ckpt: Path, device: str, weights: str | None = None, proc: str | None = None, n_action_steps: int | None = None
    ):
        """weights: load the network from here instead of ckpt (e.g. "lerobot/smolvla_base" for the zero-shot
        BEFORE number) while the processors (rename map + our dataset's normalisation stats) still come from ckpt,
        so the only difference from the fine-tuned policy is the gradient steps. proc: take the processors from here
        instead (e.g. the base model's own SO-100 pretraining stats = the "out of the box" variant), with our camera
        renamed to camera1 as in training."""
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(weights or str(ckpt))
        self.policy.config.device = device
        if n_action_steps:  # replan every n frames instead of executing the whole 50-action chunk open loop
            self.policy.config.n_action_steps = n_action_steps
        self.policy.to(device).eval()
        over = {"device_processor": {"device": device}}
        if proc:
            over["rename_observations_processor"] = {
                "rename_map": {"observation.images.front": "observation.images.camera1"}
            }
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=proc or str(ckpt),
            preprocessor_overrides=over,
        )

    def reset(self) -> None:
        self.policy.reset()

    def act(self, state_lr: np.ndarray, rgb: np.ndarray | dict, task: str) -> np.ndarray:
        """rgb: the front frame, or {camera name: frame} when the policy was trained on several cameras."""
        imgs = rgb if isinstance(rgb, dict) else {"front": rgb}
        obs = {
            "observation.state": torch.from_numpy(state_lr.astype(np.float32))[None],
            **{
                f"observation.images.{c}": torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255.0
                for c, im in imgs.items()
            },
            "task": [task],
        }
        with torch.inference_mode():
            a = self.policy.select_action(self.pre(obs))
        return self.post(a)[0].detach().cpu().numpy()


def rollout(
    scene: Scene,
    runner: VLARunner,
    renderer,
    cube_xy,
    task: str,
    seed: int,
    max_s: float = 25.0,
    on_frame=None,
    cameras: tuple[str, ...] = ("front",),
    noise_offset: int = 0,
) -> dict:
    """policy.rollout's loop and success rule, with SmolVLA choosing the action from the camera frame. SmolVLA
    samples its actions from noise (flow matching) and the GPU kernels are not bit-exact from run to run, so the
    episode's full state (qpos) is kept per frame: the hero film re-draws the evaluated episode itself."""
    m, d = scene.model, scene.data
    scene.reset(HOME, GRIPPER_CLOSED, cube_xy, 0.0)
    runner.reset()
    torch.manual_seed(seed + noise_offset)  # offset 0 = the recorded eval; other offsets redraw the flow-matching noise
    lifted, settled = False, 0
    traj = []
    for t in range(int(max_s * 30)):
        imgs = {}
        for cam in cameras:
            renderer.update_scene(d, camera=cam)
            imgs[cam] = renderer.render()
        a = runner.act(to_lerobot_units(scene.joint_pos()), imgs, task)
        d.ctrl[:] = from_lerobot_units(a)
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(m, d)
        c = scene.cube_pos()
        traj.append(d.qpos.copy())
        lifted |= bool(c[2] > CUBE_HALF + 0.02)
        if on_frame is not None:
            on_frame(t)
        settled = (
            settled + 1
            if (lifted and in_bowl(c) and np.linalg.norm(d.qvel[-6:-3]) < 1e-3)
            else 0
        )
        if settled > 30:
            break
    c = scene.cube_pos()
    return {
        "success": bool(lifted and in_bowl(c)),
        "lifted": lifted,
        "cube_end": c.round(4).tolist(),
        "frames": t + 1,
        "traj": np.array(traj, np.float64),
    }


def eval_positions(run_dir: Path) -> list[tuple[int, np.ndarray]]:
    """The MLP's exact eval starts: anchors of the trained clips, jittered by seed (stages.train_and_eval)."""
    from .stages import JITTER, _anchor

    tj = json.loads((run_dir / "train.json").read_text())
    anchors = [_anchor(run_dir, c) for c in tj["clips"]]
    out = []
    for s, logged in enumerate(tj["eval"]["episodes"]):
        r = np.random.default_rng(10_000 + s)
        xy = anchors[s % len(anchors)] + r.uniform(-np.array(JITTER), JITTER)
        assert logged["seed"] == 10_000 + s and np.allclose(
            xy.round(4), logged["cube_xy"]
        ), (s, xy, logged)
        out.append((10_000 + s, xy))
    return out


def main() -> None:
    from .scene import load

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--weights", default=None, help="network weights to evaluate (default: ckpt); processors stay from ckpt")
    ap.add_argument("--proc", default=None, help="take the pre/post processors (normalisation stats) from here instead of ckpt")
    ap.add_argument("--n-action-steps", type=int, default=None, help="execute this many actions of each chunk, then replan (default: the checkpoint's, 50)")
    ap.add_argument("--cameras", default="front", help="comma list of sim cameras the policy was trained on (front,wrist)")
    ap.add_argument("--noise-offset", type=int, default=0, help="add to each episode's torch seed: same positions, new action noise (eval-noise check)")
    ap.add_argument("--hero", action="store_true", help="film one evaluated episode (first success) at 1920x1080")
    a = ap.parse_args()

    task = (a.run_dir / "task.txt").read_text().strip()
    scene = load(task="pick")
    renderer = mujoco.Renderer(scene.model, IMG_H, IMG_W)
    mlp = json.loads((a.run_dir / "train.json").read_text())["eval"]["episodes"]
    if a.hero:
        from .media import SimFilm, hero_cam

        ev = json.loads((a.ckpt.parent / "eval.json").read_text())
        pick = next((e for e in ev["episodes"] if e["success"]), ev["episodes"][0])
        traj = np.load(a.ckpt.parent / "eval_traj.npz")[str(pick["seed"])]
        assert len(traj) == pick["frames"]
        film = SimFilm(scene, a.ckpt.parent / "hero.mp4", w=1920, h=1080, camera=hero_cam())
        for q in traj:  # the evaluated episode's own states, frame by frame
            scene.data.qpos[:] = q
            mujoco.mj_forward(scene.model, scene.data)
            film.frame()
        film.close()
        c = scene.cube_pos()
        print(f"[hero] seed {pick['seed']}: the evaluated episode, {len(traj)} frames, ends at {c.round(4).tolist()} (eval {pick['cube_end']})", flush=True)
        assert np.allclose(c.round(4), pick["cube_end"])
        (a.ckpt.parent / "hero.json").write_text(json.dumps({"seed": pick["seed"], "success": pick["success"], "frames": len(traj),
                                                             "cube_xy": pick["cube_xy"]}, indent=1))
        return
    assert not ((a.weights or a.n_action_steps or a.noise_offset) and a.out is None), "--weights / --n-action-steps need --out (keep the fine-tuned eval.json)"
    runner = VLARunner(a.ckpt, a.device, a.weights, a.proc, a.n_action_steps)
    print(
        f"[eval] SmolVLA {a.weights or a.ckpt} on {a.seeds} unseen cube positions (the MLP's seeds), task: {task!r}",
        flush=True,
    )
    results, k, t0, trajs = [], 0, time.time(), {}
    for i, (seed, xy) in enumerate(eval_positions(a.run_dir)[: a.seeds]):
        r = rollout(scene, runner, renderer, xy, task, seed, cameras=tuple(a.cameras.split(",")), noise_offset=a.noise_offset)
        trajs[str(seed)] = r.pop("traj")
        k += r["success"]
        results.append({"seed": seed, "cube_xy": xy.round(4).tolist(), **r})
        print(
            f"[eval] seed {seed} cube ({xy[0]:.3f}, {xy[1]:.3f})  SmolVLA {'IN BOWL' if r['success'] else 'miss   '}"
            f"  MLP {'IN BOWL' if mlp[i]['success'] else 'miss'}  {k}/{i + 1}  {time.time() - t0:.0f} s",
            flush=True,
        )
    from .policy import wilson

    lo, hi = wilson(k, len(results))
    print(
        f"[eval] SmolVLA success {k}/{len(results)} (Wilson 95% {lo:.2f}-{hi:.2f}), MLP {sum(e['success'] for e in mlp[: len(results)])}/{len(results)}",
        flush=True,
    )
    out = a.out or a.ckpt.parent / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.parent / "eval_traj.npz", **trajs)
    out.write_text(
        json.dumps(
            {
                "ckpt": str(a.ckpt),
                "weights": a.weights or str(a.ckpt),
                "processors": a.proc or str(a.ckpt),
                "n_action_steps": runner.policy.config.n_action_steps,
                "cameras": a.cameras,
                "noise_offset": a.noise_offset,
                "task": task,
                "successes": k,
                "seeds": len(results),
                "wilson95": [round(lo, 3), round(hi, 3)],
                "seconds": round(time.time() - t0, 1),
                "device": a.device,
                "episodes": results,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
