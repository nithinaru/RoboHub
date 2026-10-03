"""Scripted expert for "Pick up the cube and place it in the bowl." on the simulated SO-101 (scene task "pick").

Same start spots, jitter and yaw as the push task (expert.SPOTS); the bowl sits where the taped square was.

The IK extends expert.IK with a second orientation term: besides "jaws point straight down" (site +x down), the
jaws' closing axis (site +z, fixed finger -> moving finger) is turned onto a horizontal direction. For a grasp that
direction is the cube face normal closest to the radial direction from the base, so the wrist never rolls more
than 45 degrees away from its natural pose and the jaws close across two parallel faces.

Phases: settle, open, hover above the cube, descend with the cube centred between the open jaws, slide the fixed
finger up to the cube face, close (ramped), lift, carry over the bowl, lower, open, lift, return home.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from .expert import IK, SPOTS, sample_start
from .scene import (
    BOWL_R,
    BOWL_WALL_H,
    CUBE_HALF,
    GRIPPER_CLOSED,
    HOME,
    STEPS_PER_CONTROL,
    STEPS_PER_FRAME,
    TARGET_XY,
    TIMESTEP,
    Scene,
)

TASK = "Pick up the cube and place it in the bowl."
__all__ = ["TASK", "SPOTS", "run_episode", "success", "sample_start"]

GRIP_OPEN = (
    0.7  # rad; ~5 cm between the pads, 1 cm clearance on each side of the 3 cm cube
)
GRIP_DEPTH = 0.012  # cube centre sits this far up the fingers from the tips (tips 3 mm above the table)
CENTRE_OPEN = (
    0.024  # cube centre along the closing axis while descending (between the open pads)
)
CENTRE_GRASP = 0.0135  # ... after sliding the fixed pad (inner face at -0.003) against the cube face
HOVER_Z = 0.07  # site height for transit (0.08 puts wrist_flex on its limit at the near spots)
JAW_REACH = (
    0.05  # how far the open moving jaw reaches past the cube centre along the heading
)
BOWL_CLEAR = 0.02  # keep that jaw point this far outside the bowl's inner radius
CARRY_Z = 0.075  # cube centre height while carrying
RELEASE_Z = 0.048  # cube centre height when the jaws open over the bowl (bottom 3.3 cm up, wall 2.5 cm)
TRANSIT_SPEED = 0.12
APPROACH_SPEED = 0.05
LIFT_SPEED = 0.06
GRIP_RATE = 2.5  # rad/s gripper command ramp
DT = STEPS_PER_CONTROL * TIMESTEP
DOWN = np.array([0.0, 0.0, -1.0])


def in_bowl(cube_xyz) -> bool:
    """Cube centre within the bowl's inner radius and resting low (not held above it)."""
    c = np.asarray(cube_xyz)
    return bool(
        np.linalg.norm(c[:2] - TARGET_XY) < BOWL_R and c[2] < BOWL_WALL_H + CUBE_HALF
    )


def success(cube_xyz, gripper_q: float | None = None) -> bool:
    return in_bowl(cube_xyz)


class GraspIK(IK):
    """DLS IK: position + site x straight down + site z onto a horizontal heading."""

    def __init__(self, scene: Scene, w_ori: float = 0.1, **kw):
        super().__init__(scene, w_ori=w_ori, **kw)
        self.heading = np.array([1.0, 0.0, 0.0])

    def solve(self, target, q0, iters: int = 80, tol: float = 5e-4):
        q = np.array(q0, dtype=float)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        h = self.heading
        for _ in range(iters):
            pos, mat = self.fk(q)
            ep = target - pos
            eo = np.cross(mat[:, 0], DOWN) + np.cross(mat[:, 2], h)
            if np.linalg.norm(ep) < tol and np.linalg.norm(eo) < 0.02:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dof], self.w_ori * jr[:, self.dof]])
            e = np.concatenate([ep, self.w_ori * eo])
            JJt = J @ J.T + self.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(5) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (self.w_post * (HOME - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        pos, _ = self.fk(q)
        return q, float(np.linalg.norm(target - pos))


def grasp_heading(cube_xy, yaw: float) -> np.ndarray:
    """Face normal of the cube closest to the radial direction from the arm base, skipping any that would swing
    the open moving jaw (on the +heading side) into the bowl wall."""
    cube_xy = np.asarray(cube_xy)
    radial = cube_xy / max(np.linalg.norm(cube_xy), 1e-9)
    normals = [
        np.array([np.cos(yaw + k * np.pi / 2), np.sin(yaw + k * np.pi / 2), 0.0])
        for k in range(4)
    ]
    normals.sort(key=lambda n: -float(n[:2] @ radial))
    for n in normals[
        :3
    ]:  # never the one pointing back at the base (needs a ~180 degree wrist roll)
        if (
            np.linalg.norm(cube_xy + n[:2] * JAW_REACH - TARGET_XY)
            > BOWL_R + BOWL_CLEAR
        ):
            return n
    return normals[0]


def cube_yaw(scene: Scene) -> float:
    mat = scene.data.xmat[scene.model.body("cube").id].reshape(3, 3)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def site_for_cube(cube_xyz, heading, centre: float) -> np.ndarray:
    """Site position that puts the cube centre GRIP_DEPTH up the fingers and `centre` along the closing axis."""
    # cube = site + (-GRIP_DEPTH) * x_axis + centre * z_axis, with x_axis = DOWN, z_axis = heading
    return np.asarray(cube_xyz) + GRIP_DEPTH * DOWN - centre * heading


@dataclass
class EpisodeResult:
    success: bool
    frames: int
    cube_start: np.ndarray
    cube_end: np.ndarray
    spot: int
    max_ik_err: float
    lifted: bool  # the cube left the table during the carry


def run_episode(
    scene: Scene,
    spot: int,
    rng: np.random.Generator,
    on_frame: Callable[[np.ndarray, np.ndarray], None] | None = None,
    max_seconds: float = 20.0,
    release_z: float = RELEASE_Z,
    success_fn: Callable | None = None,
    correct_slip: bool = False,
    carry_speed: float = TRANSIT_SPEED,
) -> EpisodeResult:
    """on_frame(state_rad[6], action_rad[6]) at 30 fps, as in expert.run_episode. release_z: cube centre height
    when the jaws open over the target (stack_expert raises it onto the base block); success_fn replaces the
    in-bowl rule."""
    cube_xy, yaw = sample_start(rng, spot)
    scene.reset(HOME, GRIPPER_CLOSED, cube_xy, yaw)
    ik = GraspIK(scene)
    m, d = scene.model, scene.data

    q_cmd = HOME.copy()
    ee_cmd = ik.fk(HOME)[0]
    g_cmd, g_goal = GRIPPER_CLOSED, GRIPPER_CLOSED
    max_err = 0.0
    step = frames = 0
    lifted = False

    def tick(ee_goal, speed):
        nonlocal q_cmd, ee_cmd, g_cmd, max_err, step, frames
        delta = ee_goal - ee_cmd
        dist = float(np.linalg.norm(delta))
        stepmax = speed * DT
        ee_cmd = (
            ee_goal.copy() if dist <= stepmax else ee_cmd + delta * (stepmax / dist)
        )
        g_cmd += float(np.clip(g_goal - g_cmd, -GRIP_RATE * DT, GRIP_RATE * DT))
        q_cmd, err = ik.solve(ee_cmd, q_cmd)
        max_err = max(max_err, err)
        d.ctrl[:5] = q_cmd
        d.ctrl[5] = g_cmd
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

    def goto(goal, speed, settle=0.1):
        for _ in range(int(max_seconds / DT)):
            if (
                tick(np.asarray(goal, float), speed) < 1e-6
                and abs(g_goal - g_cmd) < 1e-6
            ):
                break
        hold(settle)

    def grip(goal, settle):
        nonlocal g_goal
        g_goal = goal
        goto(ee_cmd.copy(), TRANSIT_SPEED, settle)

    hold(0.3)
    c0 = scene.cube_pos()
    ik.heading = grasp_heading(c0[:2], cube_yaw(scene))
    above = site_for_cube(c0, ik.heading, CENTRE_OPEN)
    # open while moving over the cube (the roll turns during the transit)
    g_goal = GRIP_OPEN
    goto([*above[:2], HOVER_Z], TRANSIT_SPEED)
    goto(above, APPROACH_SPEED)
    goto(site_for_cube(c0, ik.heading, CENTRE_GRASP), APPROACH_SPEED * 0.5, settle=0.05)
    grip(GRIPPER_CLOSED, settle=0.4)

    c = scene.cube_pos()
    carry = site_for_cube([*c[:2], CARRY_Z], ik.heading, CENTRE_GRASP)
    goto(carry, LIFT_SPEED, settle=0.1)
    lifted = bool(scene.cube_pos()[2] > CUBE_HALF + 0.02)
    over = site_for_cube([*TARGET_XY, CARRY_Z], ik.heading, CENTRE_GRASP)
    goto(over, carry_speed, settle=0.15)
    if correct_slip:  # closed loop on the held cube itself: nudge the jaws until the cube is over the target
        for _ in range(4):
            err = TARGET_XY - scene.cube_pos()[:2]
            if np.linalg.norm(err) < 0.002:
                break
            goto(ee_cmd + np.array([*err, 0.0]), APPROACH_SPEED, settle=0.15)
    aim_site = ee_cmd.copy()
    aim_site[2] += release_z - CARRY_Z
    goto(
        aim_site,
        LIFT_SPEED,
        settle=0.1,
    )
    grip(GRIP_OPEN, settle=0.4)
    goto([*ee_cmd[:2], HOVER_Z + 0.02], LIFT_SPEED, settle=0.0)
    g_goal = GRIPPER_CLOSED
    goto(ik.fk(HOME)[0], TRANSIT_SPEED, settle=0.5)

    end = scene.cube_pos()
    return EpisodeResult(
        success=(success_fn or success)(end),
        frames=frames,
        cube_start=np.array([*cube_xy, yaw]),
        cube_end=end,
        spot=spot,
        max_ik_err=max_err,
        lifted=lifted,
    )


if __name__ == "__main__":
    import sys
    import time

    from .scene import load

    sc = load(task="pick")
    rng = np.random.default_rng(0)
    ok = 0
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    for i in range(n):
        r = run_episode(sc, i % len(SPOTS), rng)
        ok += r.success
        print(
            f"ep {i} spot {r.spot} success {r.success} lifted {r.lifted} frames {r.frames} "
            f"start {r.cube_start.round(3)} end {r.cube_end.round(3)} ik_err {r.max_ik_err:.4f}"
        )
    print(f"raw success {ok}/{n} in {time.time() - t0:.1f}s")
