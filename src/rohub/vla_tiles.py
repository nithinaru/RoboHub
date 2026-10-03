"""VLA tile films: the fine-tuned SmolVLA itself, running each accepted clip's scenario in MuJoCo, filmed for the
site's tiles. One film per clip: <run>/media/vla/<clip>.mp4 plus <clip>.json
{clip, success, frames, cube_xy, model, ...}. The page shows a tile's film once its JSON exists.

Two modes:

  rollout (default)  per clip, the cube starts at that clip's own anchor (the exact cube start the data stage
                     measured, stages._anchor), SmolVLA runs ONE episode (vla.rollout, seed 10000 + the clip's
                     index among the run's accepted clips), and the episode's recorded states are filmed with the
                     hero camera, exactly like vla.py --hero.
  --from-eval        no model: per clip, the first SUCCESSFUL evaluated episode whose start came from that clip's
                     anchor (eval seed 10000+s uses anchor s % n_clips of train.json, see vla.eval_positions),
                     else that clip's first episode; its stored states are filmed.

    PYTHONPATH=src python scripts/vla_tiles.py <run_dir> --ckpt <checkpoint dir> [--device cuda|mps|cpu]
    PYTHONPATH=src python scripts/vla_tiles.py <run_dir> --from-eval eval.json eval_traj.npz

Headless: on Linux MUJOCO_GL defaults to egl.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

MODEL = "SmolVLA (lerobot/smolvla_base, fine-tuned on this run's dataset)"


def _noop(*_a, **_k) -> None:
    pass


def accepted_clips(run_dir: Path) -> list[str]:
    """The run's accepted clips, in order: the data stage's verdicts (data.json), else train.json's clips."""
    dj = run_dir / "data.json"
    if dj.exists():
        return sorted(
            c["clip_id"] for c in json.loads(dj.read_text())["clips"] if c["accepted"]
        )
    return list(json.loads((run_dir / "train.json").read_text())["clips"])


def resolve_ckpt(p: Path) -> Path:
    """A checkpoint dir (.../checkpoints/last) or its pretrained_model dir -> the pretrained_model dir."""
    p = Path(p).expanduser()
    return p / "pretrained_model" if (p / "pretrained_model").is_dir() else p


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def films(out_dir: Path) -> dict[str, dict]:
    """The films already made: clip -> its JSON."""
    return {
        p.stem: json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("v*.json"))
    }


def film_traj(
    scene, traj: np.ndarray, out: Path, w: int, h: int, on_frame=_noop
) -> np.ndarray:
    """Re-draw an episode from its recorded states (qpos per frame) with the hero camera. Writes to a temporary
    name and renames, so the page never sees half a film. Returns the cube's final position."""
    import mujoco

    from .media import SimFilm, hero_cam

    tmp = out.with_name(out.stem + ".part.mp4")
    film = SimFilm(scene, tmp, w=w, h=h, camera=hero_cam())
    for t, q in enumerate(traj):
        scene.data.qpos[:] = q
        mujoco.mj_forward(scene.model, scene.data)
        film.frame()
        on_frame(t)
    film.close()
    tmp.replace(out)
    return scene.cube_pos()


def _write_json(out_dir: Path, rec: dict) -> None:
    p = out_dir / f"{rec['clip']}.json"
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(p)


def from_eval(
    run_dir: Path,
    eval_json: Path,
    traj_npz: Path,
    out_dir: Path,
    clips=None,
    w=1280,
    h=720,
    emit=_noop,
    log=print,
) -> list[dict]:
    """Film one evaluated episode per clip, no model (see the module docstring)."""
    from .scene import load

    ev = json.loads(Path(eval_json).read_text())
    trajs = np.load(traj_npz)
    tclips = list(json.loads((run_dir / "train.json").read_text())["clips"])
    want = clips or tclips
    scene = load(task="pick")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for cid in want:
        if cid not in tclips:
            log(
                f"[vla] {cid}: not one of the evaluated clips {tclips}; needs rollout mode (--ckpt)"
            )
            emit({"type": "vla_skip", "clip": cid, "reason": "not in the evaluation"})
            continue
        k = tclips.index(cid)
        eps = [e for e in ev["episodes"] if (e["seed"] - 10_000) % len(tclips) == k]
        pick = next((e for e in eps if e["success"]), eps[0])
        traj = trajs[str(pick["seed"])]
        assert len(traj) == pick["frames"], (
            cid,
            pick["seed"],
            len(traj),
            pick["frames"],
        )
        emit({"type": "vla_clip", "clip": cid, "stage": "film", "frames": len(traj)})
        c = film_traj(
            scene,
            traj,
            out_dir / f"{cid}.mp4",
            w,
            h,
            on_frame=lambda t: (
                t % 30 == 0
                and emit(
                    {
                        "type": "vla_progress",
                        "clip": cid,
                        "stage": "film",
                        "frame": t,
                        "frames": len(traj),
                    }
                )
            ),
        )
        assert np.allclose(c.round(4), pick["cube_end"]), (cid, c, pick["cube_end"])
        rec = {
            "clip": cid,
            "success": bool(pick["success"]),
            "frames": len(traj),
            "cube_xy": pick["cube_xy"],
            "cube_end": pick["cube_end"],
            "seed": pick["seed"],
            "model": MODEL,
            "weights": ev.get("weights"),
            "cameras": ev.get("cameras", "front"),
            "source": "evaluated episode (vla.py eval, same start)",
            "film": f"{cid}.mp4",
        }
        _write_json(out_dir, rec)
        emit({"type": "vla_film", **rec, "file": f"media/vla/{cid}.mp4"})
        log(
            f"[vla] {cid}: eval seed {pick['seed']} {'IN BOWL' if pick['success'] else 'miss'}, {len(traj)} frames -> {out_dir / (cid + '.mp4')}"
        )
        done.append(rec)
    return done


def rollouts(
    run_dir: Path,
    ckpt: Path,
    out_dir: Path,
    clips=None,
    device="auto",
    cameras=("front",),
    w=1280,
    h=720,
    emit=_noop,
    log=print,
) -> list[dict]:
    """One SmolVLA episode per clip, from that clip's anchor, filmed (see the module docstring)."""
    import mujoco

    from .dataset import IMG_H, IMG_W
    from .scene import load
    from .stages import _anchor
    from .vla import VLARunner, rollout

    order = accepted_clips(run_dir)
    want = clips or order
    task = (run_dir / "task.txt").read_text().strip()
    ckpt = resolve_ckpt(ckpt)
    device = pick_device(device)
    emit(
        {
            "type": "vla_clip",
            "clip": want[0] if want else None,
            "stage": "load",
            "device": device,
        }
    )
    log(f"[vla] loading SmolVLA from {ckpt} on {device}")
    runner = VLARunner(ckpt, device)
    scene = load(task="pick")
    renderer = mujoco.Renderer(scene.model, IMG_H, IMG_W)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for cid in want:
        idx = order.index(cid) if cid in order else len(order) + want.index(cid)
        seed = 10_000 + idx
        xy = _anchor(run_dir, cid)
        emit({"type": "vla_clip", "clip": cid, "stage": "rollout", "seed": seed})
        t0 = time.time()
        r = rollout(
            scene,
            runner,
            renderer,
            xy,
            task,
            seed,
            cameras=tuple(cameras),
            on_frame=lambda t: (
                t % 15 == 0
                and emit(
                    {
                        "type": "vla_progress",
                        "clip": cid,
                        "stage": "rollout",
                        "frame": t,
                    }
                )
            ),
        )
        traj = r.pop("traj")
        log(
            f"[vla] {cid}: seed {seed} cube ({xy[0]:.3f}, {xy[1]:.3f}) {'IN BOWL' if r['success'] else 'miss'} {r['frames']} frames {time.time() - t0:.0f} s"
        )
        emit(
            {
                "type": "vla_clip",
                "clip": cid,
                "stage": "film",
                "frames": r["frames"],
                "success": r["success"],
            }
        )
        c = film_traj(
            scene,
            traj,
            out_dir / f"{cid}.mp4",
            w,
            h,
            on_frame=lambda t: (
                t % 30 == 0
                and emit(
                    {
                        "type": "vla_progress",
                        "clip": cid,
                        "stage": "film",
                        "frame": t,
                        "frames": r["frames"],
                    }
                )
            ),
        )
        assert np.allclose(c.round(4), r["cube_end"]), (cid, c, r["cube_end"])
        rec = {
            "clip": cid,
            "success": r["success"],
            "lifted": r["lifted"],
            "frames": r["frames"],
            "cube_xy": xy.round(4).tolist(),
            "cube_end": r["cube_end"],
            "seed": seed,
            "model": MODEL,
            "weights": str(ckpt),
            "cameras": ",".join(cameras),
            "device": device,
            "source": "one rollout from this clip's own cube start",
            "seconds": round(time.time() - t0, 1),
            "film": f"{cid}.mp4",
        }
        _write_json(out_dir, rec)
        emit({"type": "vla_film", **rec, "file": f"media/vla/{cid}.mp4"})
        done.append(rec)
    renderer.close()
    return done


@contextlib.contextmanager
def heavy_lock(path: str | None, emit=_noop, log=print, poll_s: float = 10.0):
    """The machine's one-heavy-job-at-a-time lock (mkdir is atomic; owner PID inside). No-op when path is empty."""
    if not path:
        yield
        return
    lock = Path(path).expanduser()
    said = False
    while True:
        try:
            lock.mkdir()
            (lock / "owner").write_text(str(os.getpid()))
            break
        except FileExistsError:
            try:
                pid = int((lock / "owner").read_text().strip())
                os.kill(pid, 0)
            except (OSError, ValueError):  # stale: the owner is gone
                with contextlib.suppress(OSError):
                    (lock / "owner").unlink()
                with contextlib.suppress(OSError):
                    lock.rmdir()
                continue
            if not said:
                log(f"[vla] waiting for the heavy-job lock {lock} (pid {pid})")
                emit({"type": "vla_wait", "lock": str(lock), "pid": pid})
                said = True
            time.sleep(poll_s)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            (lock / "owner").unlink()
        with contextlib.suppress(OSError):
            lock.rmdir()


def web_stage(run_dir: Path, args: dict, emit=_noop, log=print) -> None:
    """The server's "vla" stage: rollouts for the clips asked for (default: accepted clips without a film)."""
    out_dir = run_dir / "media" / "vla"
    have = films(out_dir)
    clips = args.get("clips") or [c for c in accepted_clips(run_dir) if c not in have]
    emit({"type": "vla_start", "clips": clips})
    if clips:
        with heavy_lock(args.get("heavy_lock"), emit, log):
            rollouts(
                run_dir,
                Path(args["ckpt"]),
                out_dir,
                clips,
                args.get("device", "auto"),
                tuple(args.get("cameras", "front").split(",")),
                emit=emit,
                log=log,
            )
    emit({"type": "vla_done", "films": films(out_dir)})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--ckpt",
        type=Path,
        help="SmolVLA checkpoint dir (.../checkpoints/last or its pretrained_model)",
    )
    ap.add_argument(
        "--from-eval",
        nargs=2,
        type=Path,
        metavar=("EVAL_JSON", "EVAL_TRAJ_NPZ"),
        help="film evaluated episodes instead of running the model",
    )
    ap.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="default: every accepted clip of the run",
    )
    ap.add_argument(
        "--missing", action="store_true", help="only clips that have no film yet"
    )
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument(
        "--cameras",
        default="front",
        help="comma list of sim cameras the policy was trained on (front,wrist)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <run_dir>/media/vla (what the site serves)",
    )
    ap.add_argument("--size", default="1280x720")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print which episode each clip would get; film nothing",
    )
    a = ap.parse_args(argv)
    out = a.out or a.run_dir / "media" / "vla"
    w, h = (int(x) for x in a.size.split("x"))
    clips = a.clips or None
    if a.missing:
        have = films(out)
        clips = [c for c in (clips or accepted_clips(a.run_dir)) if c not in have]
        if not clips:
            print("[vla] every clip already has a film")
            return
    if a.from_eval:
        if a.dry_run:
            ev = json.loads(a.from_eval[0].read_text())
            tclips = list(json.loads((a.run_dir / "train.json").read_text())["clips"])
            for cid in clips or tclips:
                if cid not in tclips:
                    print(f"[dry-run] {cid}: not evaluated")
                    continue
                k = tclips.index(cid)
                eps = [
                    e for e in ev["episodes"] if (e["seed"] - 10_000) % len(tclips) == k
                ]
                pick = next((e for e in eps if e["success"]), eps[0])
                print(
                    f"[dry-run] {cid}: eval seed {pick['seed']} success={pick['success']} frames={pick['frames']} "
                    f"cube_xy={pick['cube_xy']} ({sum(e['success'] for e in eps)}/{len(eps)} of its episodes succeeded)"
                )
            return
        from_eval(a.run_dir, *a.from_eval, out, clips, w, h)
        return
    assert a.ckpt, "rollout mode needs --ckpt (or use --from-eval)"
    if a.dry_run:
        order = accepted_clips(a.run_dir)
        for cid in clips or order:
            print(
                f"[dry-run] {cid}: rollout seed {10_000 + order.index(cid) if cid in order else '?'} from its anchor"
            )
        return
    rollouts(a.run_dir, a.ckpt, out, clips, a.device, tuple(a.cameras.split(",")), w, h)


if __name__ == "__main__":
    main()
