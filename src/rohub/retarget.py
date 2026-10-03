"""Stage 4: retarget a human demonstration onto the SO-101 and replay it in MuJoCo with physics.

The human's grasp point becomes the SO-101's grasp point (the spot between the jaw pads where the cube centre
sits, 12 mm up the fingers). The jaws point straight down and close across two cube faces, as in the scripted
pick expert (pick_expert.GraspIK / site_for_cube). Everything the robot does is the human's path, with three
declared additions: a reach from the home pose to where the hand is first seen, a dwell at the grasp (0.3 s to
settle, 0.4 s to close) and at the release (0.4 s) so the servo gripper has time to close / open (a hand pinches faster than an STS3215), and a return
home after the hand leaves. The demo is played TIME_SCALE times slower than the human did it.

The replay is a real physics rollout (cube on a free joint, contacts, gravity). Nothing is teleported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .pick_expert import (
    CENTRE_GRASP,
    CENTRE_OPEN,
    GRIP_DEPTH,
    GRIP_OPEN,
    GRIP_RATE,
    HOVER_Z,
    GraspIK,
    grasp_heading,
    in_bowl,
    site_for_cube,
)
from .scene import (
    CUBE_HALF,
    GRIPPER_CLOSED,
    HOME,
    STEPS_PER_CONTROL,
    STEPS_PER_FRAME,
    TARGET_XY,
    TIMESTEP,
    Scene,
)
from .track import Demo

DT = STEPS_PER_CONTROL * TIMESTEP  # 50 Hz control
TIME_SCALE = 2.0
DWELL_S = 0.4
SETTLE_S = 0.3
REACH_SPEED = 0.10  # m/s for the added reach and return segments
CENTRE_BLEND_S = (
    0.3  # slide the fixed pad onto the cube face over the last 0.3 s before the grasp
)
ARM_BODIES = (
    "base",
    "shoulder",
    "upper_arm",
    "lower_arm",
    "wrist",
    "gripper",
    "moving_jaw_so101_v1",
)


def reanchor(demo: Demo, cube_xy) -> Demo:
    """The same demonstration with the cube somewhere else: the path is shifted fully onto the new cube up to
    the grasp, and the shift fades out linearly between grasp and release so the release lands where the human
    released. Used to multiply one accepted clip into many sim episodes; every copy is gated again."""
    delta = np.array([*(np.asarray(cube_xy) - demo.cube_start[:2]), 0.0])
    T = len(demo.xyz)
    w = np.zeros(T)
    w[: demo.grasp_i + 1] = 1.0
    span = max(demo.release_i - demo.grasp_i, 1)
    for i in range(demo.grasp_i + 1, min(demo.release_i, T)):
        w[i] = 1.0 - (i - demo.grasp_i) / span
    out = Demo(**{**demo.__dict__})
    out.xyz = demo.xyz + w[:, None] * delta
    out.cube_start = demo.cube_start + delta
    return out


@dataclass
class Plan:
    grasp_pt: np.ndarray  # [K, 3] grasp-point targets at 50 Hz
    centre: np.ndarray  # [K] closing-axis offset (CENTRE_OPEN .. CENTRE_GRASP)
    grip: np.ndarray  # [K] gripper goal (rad)
    phase: list  # [K] label: reach | demo | dwell | return
    demo_idx: np.ndarray  # [K] fractional demo frame index (-1 outside the demo)


def make_plan(demo: Demo, time_scale: float = TIME_SCALE) -> Plan:
    pts, cen, grip, phase, didx = [], [], [], [], []
    xyz = demo.xyz.copy()
    xyz[:, 2] = np.maximum(
        xyz[:, 2], CUBE_HALF
    )  # never aim the grasp point below the cube centre
    T = len(xyz)
    dt_demo = 1.0 / demo.fps * time_scale

    # 1) reach: from above the first demo point (at hover height) down onto the path, gripper opening
    p_home_hover = np.array([*xyz[0, :2], max(HOVER_Z + GRIP_DEPTH, xyz[0, 2])])

    def seg(a, b, speed, label, g, c):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / (speed * DT))))
        for k in range(1, n + 1):
            u = k / n
            pts.append(
                a + (b - a) * (10 * u**3 - 15 * u**4 + 6 * u**5)
            )  # minimum-jerk profile
            cen.append(c)
            grip.append(g)
            phase.append(label)
            didx.append(-1.0)

    pts.append(p_home_hover.copy())
    cen.append(CENTRE_OPEN)
    grip.append(GRIP_OPEN)
    phase.append("reach")
    didx.append(-1.0)
    for _ in range(int(0.6 / DT)):  # open the gripper at the hover point
        pts.append(p_home_hover.copy())
        cen.append(CENTRE_OPEN)
        grip.append(GRIP_OPEN)
        phase.append("reach")
        didx.append(-1.0)
    seg(p_home_hover, xyz[0], REACH_SPEED, "reach", GRIP_OPEN, CENTRE_OPEN)

    # 2) the demonstration, resampled to 50 Hz, with dwells at grasp and release
    t_end = (T - 1) * dt_demo
    t = 0.0
    dwelled = set()
    blend_start = (demo.grasp_i * dt_demo) - CENTRE_BLEND_S * time_scale
    while t <= t_end + 1e-9:
        f = t / dt_demo
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, T - 1)
        a = f - i0
        p = xyz[i0] * (1 - a) + xyz[i1] * a
        closed = bool(demo.closed[min(int(round(f)), T - 1)])
        if t < blend_start:
            c = CENTRE_OPEN
        elif f < demo.grasp_i:
            c = CENTRE_OPEN + (CENTRE_GRASP - CENTRE_OPEN) * min(
                1.0, (t - blend_start) / (CENTRE_BLEND_S * time_scale)
            )
        else:
            c = CENTRE_GRASP
        g = GRIPPER_CLOSED if closed else GRIP_OPEN
        for ev_i, label in ((demo.grasp_i, "grasp"), (demo.release_i, "release")):
            if label not in dwelled and f >= ev_i:
                dwelled.add(label)
                g_d = GRIPPER_CLOSED if label == "grasp" else GRIP_OPEN
                if (
                    label == "grasp"
                ):  # let the arm settle onto the grasp point before the jaws close
                    for _ in range(int(SETTLE_S / DT)):
                        pts.append(xyz[ev_i].copy())
                        cen.append(CENTRE_GRASP)
                        grip.append(GRIP_OPEN)
                        phase.append("dwell")
                        didx.append(float(ev_i))
                for _ in range(int(DWELL_S / DT)):
                    pts.append(xyz[ev_i].copy())
                    cen.append(CENTRE_GRASP)
                    grip.append(g_d)
                    phase.append("dwell")
                    didx.append(float(ev_i))
        pts.append(p)
        cen.append(c)
        grip.append(g)
        phase.append("demo")
        didx.append(f)
        t += DT

    # 3) return: up to hover, then home
    last = pts[-1].copy()
    up = np.array([*last[:2], max(last[2], HOVER_Z + GRIP_DEPTH)])
    seg(last, up, REACH_SPEED, "return", GRIP_OPEN, CENTRE_GRASP)
    # the added segments meet the human path with different velocities; a 60 ms Gaussian over the whole
    # 50 Hz path removes those joins without changing the human's motion (already smoothed at 62 ms)
    P = np.array(pts)
    r = 9
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / 3.0) ** 2)
    k /= k.sum()
    Pp = np.pad(P, ((r, r), (0, 0)), mode="edge")
    P = np.stack([np.convolve(Pp[:, c], k, mode="valid") for c in range(3)], 1)
    return Plan(P, np.array(cen), np.array(grip), phase, np.array(didx))


@dataclass
class Replay:
    states: np.ndarray  # [F, 6] measured joints (rad) at 30 fps
    actions: np.ndarray  # [F, 6] commanded joints (rad) at 30 fps
    cube: np.ndarray  # [F, 3]
    ee: np.ndarray  # [F, 3] gripperframe site
    phase: list  # [F]
    q_cmd: (
        np.ndarray
    )  # [K, 6] IK joint targets at 50 Hz (before the gripper ramp: grip goal)
    ik_err: np.ndarray  # [K] metres
    at_limit: np.ndarray  # [K] bool, IK wanted to leave the joint range
    self_contacts: list  # [(frame, body_a, body_b)]
    close_frame: int  # first 30 fps frame the gripper command is fully closed after the grasp dwell starts
    lift_frame: int  # first frame the cube is 1 cm above the table
    open_frame: int  # first frame the gripper opens after the release dwell starts
    cube_at_open: np.ndarray | None
    success: bool
    lifted: bool
    extra: dict = field(default_factory=dict)


def _arm_body_ids(m) -> set:
    return {
        m.body(b).id
        for b in ARM_BODIES
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0
    }


def replay(
    scene: Scene,
    demo: Demo,
    time_scale: float = TIME_SCALE,
    on_frame=None,
    cube_yaw: float = 0.0,
) -> Replay:
    plan = make_plan(demo, time_scale)
    m, d = scene.model, scene.data
    scene.reset(HOME, GRIPPER_CLOSED, demo.cube_start[:2], cube_yaw)
    ik = GraspIK(scene)
    ik.heading = grasp_heading(demo.cube_start[:2], cube_yaw)
    arm_ids = _arm_body_ids(m)
    parent = m.body_parentid

    lo, hi = ik.lo, ik.hi
    q = HOME.copy()
    g_cmd = GRIPPER_CLOSED
    K = len(plan.grasp_pt)
    q_cmds, errs, lim = np.zeros((K, 6)), np.zeros(K), np.zeros(K, bool)
    S, A, C, E, P = [], [], [], [], []
    contacts = []
    step = 0
    close_frame = lift_frame = open_frame = -1
    cube_at_open = None
    seen_release_dwell = False
    seen_grasp_dwell = False

    # warm start: move the IK from HOME to the first target so the first command is continuous
    first_site = site_for_cube(plan.grasp_pt[0], ik.heading, plan.centre[0])
    for _ in range(5):
        q, _ = ik.solve(first_site, q)
    # glide from HOME to that pose over 1.5 s (joint-space, part of the reach)
    q_start = q.copy()
    glide = int(1.5 / DT)
    pre = [
        HOME + (q_start - HOME) * (0.5 - 0.5 * np.cos(np.pi * (k + 1) / glide))
        for k in range(glide)
    ]

    def advance(q_arm, g_goal, label):
        nonlocal g_cmd, step, close_frame, lift_frame, open_frame, cube_at_open
        g_cmd += float(np.clip(g_goal - g_cmd, -GRIP_RATE * DT, GRIP_RATE * DT))
        d.ctrl[:5] = q_arm
        d.ctrl[5] = g_cmd
        for _ in range(STEPS_PER_CONTROL):
            mujoco.mj_step(m, d)
            step += 1
            if step % STEPS_PER_FRAME:
                continue
            fr = len(S)
            S.append(scene.joint_pos())
            A.append(d.ctrl.copy())
            C.append(scene.cube_pos())
            E.append(scene.ee_pos())
            P.append(label)
            for k in range(d.ncon):
                c = d.contact[k]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if (
                    b1 in arm_ids
                    and b2 in arm_ids
                    and parent[b1] != b2
                    and parent[b2] != b1
                    and b1 != b2
                ):
                    contacts.append((fr, m.body(b1).name, m.body(b2).name))
            if seen_grasp_dwell and close_frame < 0 and g_cmd <= GRIPPER_CLOSED + 1e-6:
                close_frame = fr
            if lift_frame < 0 and C[-1][2] > CUBE_HALF + 0.01:
                lift_frame = fr
            if seen_release_dwell and open_frame < 0 and g_cmd > GRIPPER_CLOSED + 0.2:
                open_frame = fr
                cube_at_open = C[-1].copy()
            if on_frame is not None:
                on_frame(S[-1], A[-1])

    for _ in range(int(0.3 / DT)):
        advance(HOME, GRIPPER_CLOSED, "reach")
    for qa in pre:
        advance(qa, GRIP_OPEN, "reach")
    for k in range(K):
        if plan.phase[k] == "dwell":
            if plan.grip[k] <= GRIPPER_CLOSED + 1e-9:
                seen_grasp_dwell = True
            elif seen_grasp_dwell:
                seen_release_dwell = True
        target = site_for_cube(plan.grasp_pt[k], ik.heading, plan.centre[k])
        q, err = ik.solve(target, q)
        q_cmds[k, :5], q_cmds[k, 5], errs[k] = q, plan.grip[k], err
        lim[k] = bool(np.any(q <= lo + 1e-4) or np.any(q >= hi - 1e-4))
        advance(q, plan.grip[k], plan.phase[k])
    # home, closing the gripper on the way
    q_end = q.copy()
    for k in range(int(1.5 / DT)):
        u = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / int(1.5 / DT))
        advance(q_end + (HOME - q_end) * u, GRIPPER_CLOSED, "return")
    for _ in range(int(0.5 / DT)):
        advance(HOME, GRIPPER_CLOSED, "return")

    cube = np.array(C)
    return Replay(
        np.array(S),
        np.array(A),
        cube,
        np.array(E),
        P,
        q_cmds,
        errs,
        lim,
        contacts,
        close_frame,
        lift_frame,
        open_frame,
        cube_at_open,
        success=bool(in_bowl(cube[-1])),
        lifted=bool(cube[:, 2].max() > CUBE_HALF + 0.02),
        extra={"cube_end": cube[-1].round(4).tolist(), "bowl_xy": TARGET_XY.tolist()},
    )
