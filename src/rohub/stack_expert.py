"""Scripted expert for "Stack the red block on the blue block." (scene task "stack").

The pick expert's motion (same start spots, jitter, yaw, grasp and carry), with the release raised so the cube is
let go just above the flat blue block at the target spot instead of inside a bowl.

Success: the red cube rests on top of the base block (centre within the base's half width in x and y, bottom on
the base's top face) and the base block is still upright near its spot.
"""

from __future__ import annotations

import numpy as np

from . import pick_expert as PE
from .expert import SPOTS, sample_start
from .scene import BASE_HALF, BASE_HALF_Z, CUBE_HALF, TARGET_XY, Scene

TASK = "Stack the red block on the blue block."
__all__ = ["TASK", "SPOTS", "run_episode", "on_base", "sample_start"]

RELEASE_Z = 2 * BASE_HALF_Z + CUBE_HALF + 0.008  # cube centre when the jaws open: bottom 8 mm above the base top
TOP_Z = 2 * BASE_HALF_Z + CUBE_HALF  # cube centre when resting on the base


def on_base(cube_xyz, base_xyz) -> bool:
    c, b = np.asarray(cube_xyz), np.asarray(base_xyz)
    return bool(
        np.all(np.abs(c[:2] - b[:2]) < BASE_HALF)
        and abs(c[2] - (b[2] + BASE_HALF_Z + CUBE_HALF)) < 0.006
        and abs(b[2] - BASE_HALF_Z) < 0.004
        and np.linalg.norm(b[:2] - TARGET_XY) < 0.03
    )


def base_pos(scene: Scene) -> np.ndarray:
    return scene.data.xpos[scene.model.body("base_block").id].copy()


def _start_yaw0(rng, spot):
    xy, _ = sample_start(rng, spot)  # same draw as the pick/push experts, yaw fixed at 0 like the pick eval
    return xy, 0.0


def run_episode(scene: Scene, spot: int, rng, on_frame=None, max_seconds: float = 20.0) -> PE.EpisodeResult:
    orig, PE.sample_start = PE.sample_start, _start_yaw0
    try:
        return _run(scene, spot, rng, on_frame, max_seconds)
    finally:
        PE.sample_start = orig


def _run(scene, spot, rng, on_frame, max_seconds):
    r = PE.run_episode(
        scene, spot, rng, on_frame=on_frame, max_seconds=max_seconds, release_z=RELEASE_Z, correct_slip=True,
        success_fn=lambda end: on_base(end, base_pos(scene)),
    )
    return r


if __name__ == "__main__":
    import sys
    import time

    from .scene import load

    sc = load(task="stack")
    rng = np.random.default_rng(0)
    ok = 0
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    for i in range(n):
        r = run_episode(sc, i % len(SPOTS), rng)
        ok += r.success
        print(f"ep {i} spot {r.spot} success {r.success} lifted {r.lifted} frames {r.frames} "
              f"start {r.cube_start.round(3)} end {r.cube_end.round(3)} base {base_pos(sc).round(3)}")
    print(f"raw success {ok}/{n} in {time.time() - t0:.1f}s")
