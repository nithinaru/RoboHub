"""Scripted expert for "Push the cube into the taped square." on the simulated SO-101.

The expert plans in end-effector space (the MJCF's `gripperframe` site, at the jaw tips) and turns every
waypoint into joint targets with damped-least-squares IK (mj_jacSite): position plus "jaws point straight
down", with a weak pull toward the home pose in the null space. The gripper stays closed so the two jaws act
as one compact pusher.

Phases: settle, move above a point behind the cube (on the far side from the target), lower to push height,
push closed-loop (the pusher target is re-aimed every control tick at "behind the cube, on the cube->target
line", so the cube's drift and spin get corrected), stop when the cube center is well inside the square,
lift and return home.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import (
    CUBE_HALF,
    GRIPPER_CLOSED,
    HOME,
    STEPS_PER_CONTROL,
    STEPS_PER_FRAME,
    TARGET_HALF,
    TARGET_XY,
    TIMESTEP,
    Scene,
)

TASK = "Push the cube into the taped square."

# Five marked start spots around the fixed target square (0.25, 0.0); x forward from the base, y to its left.
SPOTS = [
    (0.15, 0.10),
    (0.15, -0.10),
    (0.24, 0.14),
    (0.24, -0.14),
    (0.16, 0.0),
]
JITTER = 0.01  # +-1 cm uniform on x and y
YAW_RANGE = np.pi / 4  # a cube is 90-degree symmetric, so +-45 deg covers every yaw

HOVER_Z = 0.07
PUSH_Z = 0.012  # jaw tips 12 mm above the table, below the cube's center (15 mm)
CONTACT = CUBE_HALF + 0.014  # pusher-site distance behind the cube center while pushing
STANDOFF = CUBE_HALF + 0.035  # where to lower down, clear of the cube
TRANSIT_SPEED = 0.12  # m/s
PUSH_SPEED = 0.05  # m/s
STOP_ALONG = 0.006  # or once it is this far short of the center along the push heading
MAX_SEGMENTS = 4
DONE_BOX = 0.025  # no further push once |dx|, |dy| < 2.5 cm (1.5 cm inside the tape)
LATERAL_ABORT = 0.012  # abort a segment when the cube slides this far off the push line
DT = STEPS_PER_CONTROL * TIMESTEP


def success(cube_xy) -> bool:
    d = np.abs(np.asarray(cube_xy[:2]) - TARGET_XY)
    return bool(d[0] < TARGET_HALF and d[1] < TARGET_HALF)


class IK:
    """Damped least squares on the gripperframe site for the 5 arm joints."""

    def __init__(
        self,
        scene: Scene,
        w_ori: float = 0.08,
        w_post: float = 0.02,
        damping: float = 1e-3,
    ):
        self.m = scene.model
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("gripperframe").id
        self.qadr = scene.qadr[:5]
        self.dof = scene.dofadr[:5]
        self.lo = self.m.jnt_range[[self.m.joint(i).id for i in range(5)], 0]
        self.hi = self.m.jnt_range[[self.m.joint(i).id for i in range(5)], 1]
        self.w_ori, self.w_post, self.damping = w_ori, w_post, damping

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(
            3, 3
        ).copy()

    def solve(self, target, q0, iters: int = 60, tol: float = 5e-4):
        q = np.array(q0, dtype=float)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(iters):
            pos, mat = self.fk(q)
            ep = target - pos
            eo = np.cross(
                mat[:, 0], down
            )  # rotate the jaw axis (site +x) onto straight down
            if np.linalg.norm(ep) < tol and np.linalg.norm(eo) < 0.02:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dof], self.w_ori * jr[:, self.dof]])
            e = np.concatenate([ep, self.w_ori * eo])
            JJt = J @ J.T + self.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # posture preference in the null space of the task
            N = np.eye(5) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (self.w_post * (HOME - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        pos, _ = self.fk(q)
        return q, float(np.linalg.norm(target - pos))


@dataclass
class EpisodeResult:
    success: bool
    frames: int
    cube_start: np.ndarray
    cube_end: np.ndarray
    spot: int
    max_ik_err: float
    log: list = field(default_factory=list)


def sample_start(rng: np.random.Generator, spot: int):
    x, y = SPOTS[spot]
    return (
        np.array([x + rng.uniform(-JITTER, JITTER), y + rng.uniform(-JITTER, JITTER)]),
        float(rng.uniform(-YAW_RANGE, YAW_RANGE)),
    )


def run_episode(
    scene: Scene,
    spot: int,
    rng: np.random.Generator,
    on_frame: Callable[[np.ndarray, np.ndarray], None] | None = None,
    max_seconds: float = 20.0,
) -> EpisodeResult:
    """Runs one episode. on_frame(state_rad[6], action_rad[6]) is called at 30 fps, after the step that ends
    the frame interval, with the measured joint positions and the joint targets commanded at that moment."""
    cube_xy, yaw = sample_start(rng, spot)
    scene.reset(HOME, GRIPPER_CLOSED, cube_xy, yaw)
    ik = IK(scene)
    m, d = scene.model, scene.data

    q_cmd = HOME.copy()
    ee_cmd = ik.fk(HOME)[0]
    max_err = 0.0
    step = 0
    frames = 0

    def tick(ee_goal, speed):
        """Advance one control period toward ee_goal at a capped speed; returns remaining distance."""
        nonlocal q_cmd, ee_cmd, max_err, step, frames
        delta = ee_goal - ee_cmd
        dist = float(np.linalg.norm(delta))
        stepmax = speed * DT
        ee_cmd = (
            ee_goal.copy() if dist <= stepmax else ee_cmd + delta * (stepmax / dist)
        )
        q_cmd, err = ik.solve(ee_cmd, q_cmd)
        max_err = max(max_err, err)
        d.ctrl[:5] = q_cmd
        d.ctrl[5] = GRIPPER_CLOSED
        for _ in range(STEPS_PER_CONTROL):
            mujoco.mj_step(m, d)
            step += 1
            if step % STEPS_PER_FRAME == 0:
                frames += 1
                if on_frame is not None:
                    on_frame(scene.joint_pos(), d.ctrl.copy())
        return float(np.linalg.norm(ee_goal - ee_cmd))

    def hold(seconds):
        for _ in range(int(round(seconds / DT))):
            tick(ee_cmd.copy(), TRANSIT_SPEED)

    def goto(goal, speed, settle=0.15):
        for _ in range(int(max_seconds / DT)):
            if tick(np.asarray(goal, float), speed) < 1e-6:
                break
        hold(settle)

    def line_dir():
        c = scene.cube_pos()[:2]
        v = TARGET_XY - c
        return v / max(np.linalg.norm(v), 1e-9), c

    hold(0.3)
    # Straight-line pushes. Each segment re-plans from where the cube actually is: hover behind it on the
    # cube->target line, lower, push along that fixed line, stop short of the center (the cube coasts a few mm),
    # and abort early if the cube slides off the pusher sideways. Usually one segment; a slide costs one more.
    for _ in range(MAX_SEGMENTS):
        u, c0 = line_dir()
        if np.all(np.abs(TARGET_XY - c0) < DONE_BOX):
            break
        behind = c0 - u * STANDOFF
        if ee_cmd[2] < HOVER_Z - 1e-3:
            goto(np.array([*ee_cmd[:2], HOVER_Z]), TRANSIT_SPEED, settle=0.0)
        goto([*behind, HOVER_Z], TRANSIT_SPEED)
        goto([*behind, PUSH_Z], TRANSIT_SPEED * 0.6)
        end_pusher = np.array([*(TARGET_XY - u * CONTACT), PUSH_Z])
        for _ in range(int(max_seconds / DT)):
            left = tick(end_pusher, PUSH_SPEED)
            c = scene.cube_pos()[:2]
            lateral = abs(float(u[0] * (c - c0)[1] - u[1] * (c - c0)[0]))
            if float(np.dot(TARGET_XY - c, u)) < STOP_ALONG or lateral > LATERAL_ABORT:
                break
            if left < 1e-6:
                hold(0.2)
                break
        hold(0.2)
        goto(ee_cmd + np.array([*(-u * 0.025), 0.0]), TRANSIT_SPEED * 0.5, settle=0.0)

    goto(np.array([*ee_cmd[:2], HOVER_Z]), TRANSIT_SPEED, settle=0.0)
    goto(ik.fk(HOME)[0], TRANSIT_SPEED, settle=0.5)

    end = scene.cube_pos()
    return EpisodeResult(
        success=success(end[:2]) and end[2] < CUBE_HALF + 0.005,
        frames=frames,
        cube_start=np.array([*cube_xy, yaw]),
        cube_end=end,
        spot=spot,
        max_ik_err=max_err,
    )


if __name__ == "__main__":
    import sys
    import time

    from .scene import load

    sc = load()
    rng = np.random.default_rng(0)
    ok = 0
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    for i in range(n):
        r = run_episode(sc, i % len(SPOTS), rng)
        ok += r.success
        print(
            f"ep {i} spot {r.spot} success {r.success} frames {r.frames} start {r.cube_start.round(3)} "
            f"end {r.cube_end.round(3)} ik_err {r.max_ik_err:.4f}"
        )
    print(f"raw success {ok}/{n} in {time.time() - t0:.1f}s")
