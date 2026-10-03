"""Scripted expert for "Stack three blocks into a tower." (scene task "tower"): a thin blue slab at TOWER_XY, a red
cube to its left and a green cube to its right. Two sequential placements with the pick expert's grasp: red cube
onto the slab, then green cube onto the red cube. Each carry is closed loop on the held cube (nudged over the
tower before lowering).

Success: green on red on slab, all resting, stack centred within the slab's half width.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .pick_expert import (
    APPROACH_SPEED, CENTRE_GRASP, CENTRE_OPEN, DT, GRIP_OPEN, GRIP_RATE, HOVER_Z, LIFT_SPEED, TRANSIT_SPEED,
    EpisodeResult, GraspIK, site_for_cube,
)
from .scene import BASE_HALF, CUBE_HALF, GRIPPER_CLOSED, HOME, STEPS_PER_CONTROL, STEPS_PER_FRAME, TOWER_BASE_HALF_Z, TOWER_XY, Scene

TASK = "Stack three blocks into a tower."
RED0, GREEN0 = np.array([0.15, 0.10]), np.array([0.15, -0.10])
JITTER = 0.015
SPOTS = [0]  # one layout, jittered
CARRY_Z = 0.08
SLAB_TOP = 2 * TOWER_BASE_HALF_Z


def body_pos(scene, name):
    return scene.data.xpos[scene.model.body(name).id].copy()


def tower_ok(scene) -> bool:
    b, r, g = body_pos(scene, "base_block"), body_pos(scene, "cube"), body_pos(scene, "cube2")
    return bool(
        np.all(np.abs(r[:2] - b[:2]) < BASE_HALF) and abs(r[2] - (SLAB_TOP + CUBE_HALF)) < 0.006
        and np.all(np.abs(g[:2] - r[:2]) < CUBE_HALF * 1.2) and abs(g[2] - (SLAB_TOP + 3 * CUBE_HALF)) < 0.008
        and abs(b[2] - TOWER_BASE_HALF_Z) < 0.003
    )


def sample_start(rng):
    return RED0 + rng.uniform(-JITTER, JITTER, 2), GREEN0 + rng.uniform(-JITTER, JITTER, 2)


def reset(scene: Scene, red, green):
    scene.reset(HOME, GRIPPER_CLOSED, red, 0.0)
    adr = scene.model.joint("cube2_free").qposadr[0]
    scene.data.qpos[adr: adr + 7] = [green[0], green[1], CUBE_HALF, 1, 0, 0, 0]
    mujoco.mj_forward(scene.model, scene.data)


def heading_for(xy):
    radial = np.asarray(xy) / np.linalg.norm(xy)
    normals = [np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2), 0.0]) for k in range(4)]
    return max(normals, key=lambda n: float(n[:2] @ radial))


def run_episode(scene: Scene, spot: int, rng, on_frame=None, max_seconds: float = 20.0) -> EpisodeResult:
    red, green = sample_start(rng)
    reset(scene, red, green)
    ik = GraspIK(scene)
    m, d = scene.model, scene.data
    st = {"q": HOME.copy(), "ee": ik.fk(HOME)[0], "g": GRIPPER_CLOSED, "gg": GRIPPER_CLOSED, "err": 0.0, "step": 0, "frames": 0}

    def tick(goal, speed):
        delta = goal - st["ee"]
        dist = float(np.linalg.norm(delta))
        sm = speed * DT
        st["ee"] = goal.copy() if dist <= sm else st["ee"] + delta * (sm / dist)
        st["g"] += float(np.clip(st["gg"] - st["g"], -GRIP_RATE * DT, GRIP_RATE * DT))
        st["q"], err = ik.solve(st["ee"], st["q"])
        st["err"] = max(st["err"], err)
        d.ctrl[:5] = st["q"]
        d.ctrl[5] = st["g"]
        for _ in range(STEPS_PER_CONTROL):
            mujoco.mj_step(m, d)
            st["step"] += 1
            if st["step"] % STEPS_PER_FRAME == 0:
                st["frames"] += 1
                if on_frame is not None:
                    on_frame(scene.joint_pos(), d.ctrl.copy())
        return float(np.linalg.norm(goal - st["ee"]))

    def hold(s):
        for _ in range(int(round(s / DT))):
            tick(st["ee"].copy(), TRANSIT_SPEED)

    def goto(goal, speed, settle=0.1):
        for _ in range(int(max_seconds / DT)):
            if tick(np.asarray(goal, float), speed) < 1e-6 and abs(st["gg"] - st["g"]) < 1e-6:
                break
        hold(settle)

    def grip(goal, settle):
        st["gg"] = goal
        goto(st["ee"].copy(), TRANSIT_SPEED, settle)

    lifted = True

    def place(body, target_xy, release_z):
        nonlocal lifted
        c0 = body_pos(scene, body)
        ik.heading = heading_for(c0[:2])
        above = site_for_cube(c0, ik.heading, CENTRE_OPEN)
        st["gg"] = GRIP_OPEN
        goto([*above[:2], HOVER_Z + 0.01], TRANSIT_SPEED)
        goto(above, APPROACH_SPEED)
        goto(site_for_cube(c0, ik.heading, CENTRE_GRASP), APPROACH_SPEED * 0.5, settle=0.05)
        grip(GRIPPER_CLOSED, 0.4)
        c = body_pos(scene, body)
        goto(site_for_cube([*c[:2], CARRY_Z], ik.heading, CENTRE_GRASP), LIFT_SPEED, settle=0.1)
        lifted &= bool(body_pos(scene, body)[2] > CUBE_HALF + 0.02)
        goto(site_for_cube([*target_xy, CARRY_Z], ik.heading, CENTRE_GRASP), TRANSIT_SPEED, settle=0.15)
        for _ in range(4):
            err = target_xy - body_pos(scene, body)[:2]
            if np.linalg.norm(err) < 0.002:
                break
            goto(st["ee"] + np.array([*err, 0.0]), APPROACH_SPEED, settle=0.15)
        # lower on the held cube's own height (it can sit lower in the jaws than planned)
        dz = body_pos(scene, body)[2] - release_z
        if dz > 0:
            goto(st["ee"] - np.array([0.0, 0.0, dz]), LIFT_SPEED * 0.6, settle=0.15)
        grip(GRIP_OPEN, 0.4)
        goto([*st["ee"][:2], HOVER_Z + 0.02], LIFT_SPEED, settle=0.0)

    hold(0.3)
    place("cube", TOWER_XY, SLAB_TOP + CUBE_HALF + 0.008)
    place("cube2", body_pos(scene, "cube")[:2], SLAB_TOP + 3 * CUBE_HALF + 0.008)
    st["gg"] = GRIPPER_CLOSED
    goto(ik.fk(HOME)[0], TRANSIT_SPEED, settle=0.5)
    return EpisodeResult(success=tower_ok(scene) and lifted, frames=st["frames"], cube_start=np.array([*red, 0.0]),
                         cube_end=body_pos(scene, "cube2"), spot=0, max_ik_err=st["err"], lifted=lifted)


if __name__ == "__main__":
    import sys
    import time

    from .scene import load

    sc = load(task="tower")
    rng = np.random.default_rng(0)
    ok, n, t0 = 0, int(sys.argv[1]) if len(sys.argv) > 1 else 5, time.time()
    for i in range(n):
        r = run_episode(sc, 0, rng)
        ok += r.success
        print(f"ep {i} success {r.success} frames {r.frames} ik {r.max_ik_err:.3f} red {body_pos(sc,'cube').round(3)} green {body_pos(sc,'cube2').round(3)} base {body_pos(sc,'base_block').round(3)}")
    print(f"raw success {ok}/{n} in {time.time() - t0:.1f}s")
