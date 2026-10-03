"""MuJoCo scene for the simulated SO-101 push-cube task.

The vendor MJCF (vendor/so101/so101_new_calib.xml, TheRobotStudio SO-ARM100 repo) is loaded unmodified with
MjSpec and everything else is added programmatically:

* table: a box whose top surface is z = 0; the arm base sits on it at the origin, facing +X
* cube: 3 cm, 30 g, free joint, friction 0.6 (a painted wooden block)
* target: an 8 cm square outline of "tape" on the table (visual only, contype=0)
* cameras: "front" (fixed, in front of and above the workspace, looking back at the arm, like a clamp-mounted
  webcam) and "wrist" (on the gripper body, looking along the jaws)

Control follows the web sim (src/sim/arm.ts): gravity compensation on every arm body, and the STS3215 gains
from vendor/so101/joints_properties.xml (kp 17.8, kv 0) instead of the MJCF's inline kp 998, which saturates
torque on every step. Physics runs at 300 Hz (not 250) so 50 Hz control (6 steps) and 30 fps recording
(10 steps) both land on whole physics steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MJCF = REPO / "vendor" / "so101" / "so101_new_calib.xml"

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HOME = np.array(
    [0.0, -0.6, 0.9, 0.9, 0.0]
)  # the web sim's home pose (5 arm joints, radians)
GRIPPER_CLOSED = -0.1745  # rad, joint lower limit
GRIPPER_OPEN = 1.7453  # rad, joint upper limit

TIMESTEP = 1.0 / 300.0
CONTROL_HZ = 50
RECORD_FPS = 30
STEPS_PER_CONTROL = 6  # 300 / 50
STEPS_PER_FRAME = 10  # 300 / 30

KP, KV = 17.8, 0.0

CUBE_HALF = 0.015
CUBE_MASS = 0.03
TARGET_XY = np.array([0.25, 0.0])
TARGET_HALF = 0.04  # 8 cm square; success = |dx|, |dy| < TARGET_HALF

IMG_W, IMG_H = 640, 480


def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera xyaxes for a camera at pos looking at target (camera looks along its -z)."""
    pos, target, up = map(np.asarray, (pos, target, up))
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    return np.concatenate([right, cam_up])


BASE_HALF = 0.022  # stack task: a flat blue block, 4.4 x 4.4 cm, 2 cm tall, centred on TARGET_XY
BASE_HALF_Z = 0.01  # (the arm carries the cube's bottom ~3.5 cm up, so a taller block would be swept)
BASE_MASS = 0.15
TOWER_XY = np.array([0.20, 0.0])  # tower task: a thinner blue slab here (the arm reaches higher closer in)
TOWER_BASE_HALF_Z = 0.005

BOWL_R = 0.05  # pick task: bowl inner radius, centred on TARGET_XY
BOWL_WALL_H = 0.025
BOWL_WALL_T = 0.006
BOWL_SEGMENTS = 20

# Pick task grasp contact. The vendor collision meshes are convex hulls; the fixed jaw's hull spans the whole
# gripper housing, so between the jaws it bulges up to ~1 cm into the grasp gap and a cube cannot seat between
# the finger faces. For the pick task the two jaw meshes stop colliding with the cube (contype bits below) and
# two thin box pads, fitted to the measured inner finger faces, do the grasping. Coordinates are in the
# gripperframe site frame with the jaw at GRIP_REF: +x runs from the housing to the jaw tips (straight down when
# the jaws point down), +z points from the fixed finger toward the moving one.
GRIP_REF = 0.0  # gripper angle at which the moving pad is fitted (its inner face is near-parallel here)
PAD_X = (-0.042, -0.002)  # along the fingers, tips at 0
PAD_HALF_Y = 0.009
# measured fixed-finger inner face: z ~ -0.004 over x in [-0.05, -0.005]
FIXED_PAD_Z = (-0.010, -0.003)
# measured moving-finger inner face at GRIP_REF: z ~ 0.016-0.020
MOVING_PAD_Z = (0.017, 0.024)
PAD_FRICTION = 1.0
# Stiffer contact on the pads and the cube: with MuJoCo's default (0.02, 1) the saturated gripper squeeze sinks
# the pads 5.7 mm into the cube; with these, 0.9 mm (measured over 20 pick episodes, both 20/20).
GRASP_SOLREF = [0.007, 1.0]
GRASP_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
# contype/conaffinity bits; jaw meshes use MESH_BIT only, the cube CUBE_BIT only
MESH_BIT, CUBE_BIT = 1, 2


def _pad_frames(kp: float, kv: float):
    """Pad centre/orientation in the gripper body frame and moving jaw body frame, from a compiled push scene."""
    model = build_spec(kp, kv, task="push").compile()
    data = mujoco.MjData(model)
    data.qpos[model.joint("gripper").qposadr[0]] = GRIP_REF
    mujoco.mj_kinematics(model, data)
    s = model.site("gripperframe").id
    ps, rs = data.site_xpos[s], data.site_xmat[s].reshape(3, 3)
    out = {}
    for body, zr in (("gripper", FIXED_PAD_Z), ("moving_jaw_so101_v1", MOVING_PAD_Z)):
        b = model.body(body).id
        pb, rb = data.xpos[b], data.xmat[b].reshape(3, 3)
        centre_site = np.array([np.mean(PAD_X), 0.0, np.mean(zr)])
        pos = rb.T @ (ps + rs @ centre_site - pb)
        rot = rb.T @ rs  # site axes expressed in the body frame
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        size = [(PAD_X[1] - PAD_X[0]) / 2, PAD_HALF_Y, (zr[1] - zr[0]) / 2]
        out[body] = (pos, quat, size)
    return out


def build_spec(kp: float = KP, kv: float = KV, task: str = "push") -> mujoco.MjSpec:
    """task "push": the taped square. task "pick": a bowl at the square's place, plus jaw pads for grasping.
    task "stack": no bowl; a free flat blue block (4.4 x 4.4 x 2 cm) at the square's place, plus the same jaw pads."""
    assert task in ("push", "pick", "stack", "tower"), task
    pads = _pad_frames(kp, kv) if task in ("pick", "stack", "tower") else None
    spec = mujoco.MjSpec.from_file(str(MJCF))
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = IMG_W
    spec.visual.global_.offheight = IMG_H
    spec.visual.quality.shadowsize = 2048

    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    for act in spec.actuators:
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kv

    world = spec.worldbody
    # textures / materials
    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.22, 0.23, 0.25],
        rgb2=[0.18, 0.19, 0.21],
        width=256,
        height=256,
    )
    spec.add_material(name="floor_mat", textures=["", "floor_tex"], texrepeat=[6, 6])
    spec.add_material(name="table_mat", rgba=[0.62, 0.52, 0.40, 1.0])
    spec.add_material(name="tape_mat", rgba=[0.10, 0.55, 0.95, 1.0])
    spec.add_material(name="cube_mat", rgba=[0.90, 0.15, 0.12, 1.0])

    world.add_light(
        name="key",
        pos=[0.4, -0.4, 1.4],
        dir=[-0.3, 0.3, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        specular=[0.2, 0.2, 0.2],
        castshadow=True,
    )
    world.add_light(
        name="fill",
        pos=[-0.3, 0.5, 1.2],
        dir=[0.3, -0.4, -1.0],
        diffuse=[0.35, 0.35, 0.35],
        castshadow=False,
    )

    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.1],
        pos=[0, 0, -0.75],
        material="floor_mat",
    )
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.45, 0.6, 0.02],
        pos=[0.2, 0.0, -0.02],
        material="table_mat",
        friction=[0.6, 0.005, 0.0001],
    )
    tx, ty = TARGET_XY

    if task == "push":
        # taped square: four thin strips, visual only
        t, w, h = TARGET_HALF, 0.006, 0.0005
        for i, (px, py, sx, sy) in enumerate(
            [
                (tx + t, ty, w, t + w),
                (tx - t, ty, w, t + w),
                (tx, ty + t, t + w, w),
                (tx, ty - t, t + w, w),
            ]
        ):
            world.add_geom(
                name=f"tape{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, h],
                pos=[px, py, h],
                material="tape_mat",
                contype=0,
                conaffinity=0,
                group=1,
            )
    elif task == "pick":
        # a shallow round bowl: a thin base disc and a ring of wall segments, fixed to the table
        spec.add_material(name="bowl_mat", rgba=[0.95, 0.95, 0.92, 1.0])
        bowl = world.add_body(name="bowl", pos=[tx, ty, 0.0])
        bowl.add_geom(
            name="bowl_base",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[BOWL_R + BOWL_WALL_T, 0.002, 0],
            pos=[0, 0, 0.002],
            material="bowl_mat",
        )
        seg_half = (BOWL_R + BOWL_WALL_T / 2) * np.tan(np.pi / BOWL_SEGMENTS) * 1.05
        for i in range(BOWL_SEGMENTS):
            a = 2 * np.pi * i / BOWL_SEGMENTS
            r = BOWL_R + BOWL_WALL_T / 2
            bowl.add_geom(
                name=f"bowl_wall{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[BOWL_WALL_T / 2, seg_half, BOWL_WALL_H / 2],
                pos=[r * np.cos(a), r * np.sin(a), BOWL_WALL_H / 2],
                quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                material="bowl_mat",
            )
    if task in ("stack", "tower"):
        # the base block: free, heavier than the cube, added before the cube so the cube stays the last free joint
        spec.add_material(name="base_mat", rgba=[0.15, 0.35, 0.85, 1.0])
        bx, by, bz = (TOWER_XY[0], TOWER_XY[1], TOWER_BASE_HALF_Z) if task == "tower" else (tx, ty, BASE_HALF_Z)
        base = world.add_body(name="base_block", pos=[bx, by, bz])
        base.add_freejoint(name="base_block_free")
        base.add_geom(
            name="base_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[BASE_HALF, BASE_HALF, bz],
            mass=BASE_MASS,
            material="base_mat",
            friction=[0.8, 0.005, 0.0001],
            condim=4,
            contype=CUBE_BIT,  # touches the table, the arm links, the pads and the cube, not the jaw hull meshes
            conaffinity=CUBE_BIT,
        )
    if task == "tower":
        # the second cube (green), before the red cube so the red cube stays the last free joint
        spec.add_material(name="cube2_mat", rgba=[0.15, 0.70, 0.25, 1.0])
        c2 = world.add_body(name="cube2", pos=[0.15, -0.10, CUBE_HALF])
        c2.add_freejoint(name="cube2_free")
        c2.add_geom(name="cube2_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, mass=CUBE_MASS,
                    material="cube2_mat", friction=[0.6, 0.005, 0.0001], condim=4, contype=CUBE_BIT,
                    conaffinity=CUBE_BIT, solref=GRASP_SOLREF, solimp=GRASP_SOLIMP)
    if task in ("pick", "stack", "tower"):
        # the jaw meshes keep colliding with the table and bowl but not with the cube; the pads do the grasping;
        # every other colliding geom (table, bowl, floor, the other arm links) also touches the cube
        for g in spec.geoms:
            if not g.contype or g.name in ("base_geom", "cube2_geom"):
                continue
            if g.parent.name in ("gripper", "moving_jaw_so101_v1"):
                g.contype, g.conaffinity = MESH_BIT, MESH_BIT
            else:
                g.conaffinity = MESH_BIT | CUBE_BIT
        for body, (pos, quat, size) in pads.items():
            b = next(bb for bb in spec.bodies if bb.name == body)
            b.add_geom(
                name=f"pad_{'fixed' if body == 'gripper' else 'moving'}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos,
                quat=quat,
                friction=[PAD_FRICTION, 0.005, 0.0001],
                condim=4,
                contype=CUBE_BIT,
                conaffinity=CUBE_BIT,
                solref=GRASP_SOLREF,
                solimp=GRASP_SOLIMP,
                group=3,  # not drawn: the visual jaw meshes sit on top of the pads
            )

    cube = world.add_body(name="cube", pos=[0.18, 0.08, CUBE_HALF])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF] * 3,
        mass=CUBE_MASS,
        material="cube_mat",
        friction=[0.6, 0.005, 0.0001],
        condim=4,
        **(
            {
                "contype": CUBE_BIT,
                "conaffinity": CUBE_BIT,
                "solref": GRASP_SOLREF,
                "solimp": GRASP_SOLIMP,
            }
            if task in ("pick", "stack", "tower")
            else {}
        ),
    )

    front_pos = [0.56, 0.15, 0.34]
    world.add_camera(
        name="front",
        pos=front_pos,
        xyaxes=_look_at(front_pos, [0.17, 0.0, 0.09]),
        fovy=50,
    )

    # wrist camera: on the gripper body, behind the jaws, looking along the gripperframe +x (the jaw direction)
    grip = next(b for b in spec.bodies if b.name == "gripper")
    grip.add_camera(
        name="wrist",
        pos=[0.0, 0.045, -0.02],
        xyaxes=[1, 0, 0, 0, 1, 0],
        fovy=75,
    )
    return spec


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData

    @property
    def qadr(self) -> np.ndarray:
        return np.array([self.model.joint(j).qposadr[0] for j in JOINTS])

    @property
    def dofadr(self) -> np.ndarray:
        return np.array([self.model.joint(j).dofadr[0] for j in JOINTS])

    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.model.body("cube").id].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.model.site("gripperframe").id].copy()

    def set_cube(self, x: float, y: float, yaw: float) -> None:
        adr = self.model.joint("cube_free").qposadr[0]
        self.data.qpos[adr : adr + 3] = [x, y, CUBE_HALF]
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        vadr = self.model.joint("cube_free").dofadr[0]
        self.data.qvel[vadr : vadr + 6] = 0

    def reset(
        self, arm_q: np.ndarray, gripper: float, cube_xy, cube_yaw: float
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = np.append(arm_q, gripper)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[:] = q
        self.set_cube(cube_xy[0], cube_xy[1], cube_yaw)
        mujoco.mj_forward(self.model, self.data)


def load(kp: float = KP, kv: float = KV, task: str = "push") -> Scene:
    model = build_spec(kp, kv, task).compile()
    return Scene(model, mujoco.MjData(model))


if __name__ == "__main__":
    import sys

    from PIL import Image

    sc = load()
    sc.reset(HOME, GRIPPER_CLOSED, (0.16, 0.09), 0.3)
    r = mujoco.Renderer(sc.model, IMG_H, IMG_W)
    imgs = []
    for cam in ("front", "wrist"):
        r.update_scene(sc.data, camera=cam)
        imgs.append(r.render())
    out = sys.argv[1] if len(sys.argv) > 1 else "scene_preview.png"
    Image.fromarray(np.concatenate(imgs, axis=1)).save(out)
    print("saved", out, "ee", sc.ee_pos().round(3))
