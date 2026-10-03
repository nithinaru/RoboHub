"""Second-engine cross-check: replay every episode of a LeRobot dataset in PyBullet (Bullet Physics, Erwin Coumans et
al.), an engine independent of MuJoCo, and ask it the same question the MuJoCo gate asked: does the block end in the
bowl?

The scene is converted from the compiled MuJoCo model (one description of the world, two physics engines): the SO-101
links with their masses, inertias and joint axes, the vendor meshes as collision hulls, the jaw pads, the table, the
bowl and the 3 cm, 30 g cube. What is re-simulated is the recorded action stream (the 30 fps joint targets written to
the dataset), open loop, from the recorded first state, with the same actuator law as the MuJoCo scene (position
servo kp 17.8, force limit 3.35 N m, joint damping 0.6, friction loss 0.052, gravity compensation) and the same
success rule (lifted above 2 cm, then resting inside the bowl's inner radius). Contacts, friction and integration are
Bullet's own. For a fair comparison the same open-loop replay of the same action stream is also run in MuJoCo.

Known modelling differences (stated, not hidden): Bullet has no joint armature, so the motors' reflected inertia
(0.028 kg m^2) is added to each link's inertia about its joint axis; Bullet multiplies the two frictions of a contact
while MuJoCo takes the larger, so the cube's friction is set to 1.0 in Bullet, which reproduces MuJoCo's value for
every surface the cube touches (all have friction >= the cube's 0.6); the jaw pads are separate fixed links so the jaw
meshes can skip the cube while the pads grip it, as in MuJoCo.

    PYTHONPATH=src .venv/bin/python -m rohub.crosscheck data/web-runs/put-the-red-block-in-the-bowl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .pick_expert import in_bowl
from .scene import CUBE_HALF, JOINTS, STEPS_PER_FRAME, Scene
from .units import from_lerobot_units

PAD_MASS = 1e-3  # kg; the pads are massless in MuJoCo (the vendor inertials set the body masses); Bullet needs some
SETTLE_FRAMES = 45  # hold the last action 1.5 s after the recording ends, then judge


def _xyzw(q_wxyz) -> list[float]:
    return [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]


def _rgba(m: mujoco.MjModel, g: int) -> list[float]:
    return (
        m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    ).tolist()


class Bullet:
    """The MuJoCo pick scene rebuilt in a PyBullet DIRECT client."""

    def __init__(self, scene: Scene, mesh_dir: Path, gui: bool = False, substeps: int = 2):
        import pybullet as p

        self.p, self.scene = p, scene
        m = scene.model
        self.c = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(*m.opt.gravity, physicsClientId=self.c)
        self.substeps = substeps  # Bullet steps per MuJoCo step (same control law, finer contact integration)
        p.setTimeStep(m.opt.timestep / substeps, physicsClientId=self.c)
        # solver accuracy, chosen to keep grasp penetration near MuJoCo's (~1 mm, see scene.GRASP_SOLREF): Bullet's
        # defaults let the pads sink ~4 mm into the cube. Fixed before the full run; not tuned per episode.
        p.setPhysicsEngineParameter(numSolverIterations=300, contactERP=0.95, contactSlop=0.0, physicsClientId=self.c)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_file = {}
        for i in range(m.nmesh):
            f = mesh_dir / f"mesh{i}.obj"
            v0, nv = m.mesh_vertadr[i], m.mesh_vertnum[i]
            f0, nf = m.mesh_faceadr[i], m.mesh_facenum[i]
            with open(f, "w") as fh:
                fh.writelines(
                    f"v {x:.6f} {y:.6f} {z:.6f}\n"
                    for x, y, z in m.mesh_vert[v0 : v0 + nv]
                )
                fh.writelines(
                    f"f {a + 1} {b + 1} {c + 1}\n"
                    for a, b, c in m.mesh_face[f0 : f0 + nf]
                )
            self.mesh_file[i] = str(f)

        d = scene.data
        mujoco.mj_forward(m, d)
        body = lambda n: m.body(n).id  # noqa: E731
        self.pads = {g for g in range(m.ngeom) if m.geom(g).name.startswith("pad_")}

        # static world: floor, table, bowl
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b == 0 or b == body("bowl"):
                col = self._shapes([g], collision=True)
                vis = self._shapes([g], collision=False)
                pos = d.geom_xpos[g]
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, d.geom_xmat[g])
                uid = p.createMultiBody(
                    0,
                    col,
                    vis,
                    basePosition=pos.tolist(),
                    baseOrientation=_xyzw(q),
                    physicsClientId=self.c,
                )
                p.changeDynamics(
                    uid,
                    -1,
                    lateralFriction=float(m.geom_friction[g][0]),
                    spinningFriction=1.0,
                    physicsClientId=self.c,
                )

        # cube
        cg = m.geom("cube_geom").id
        self.cube = p.createMultiBody(
            float(m.body_mass[body("cube")]),
            self._shapes([cg], True),
            self._shapes([cg], False),
            basePosition=[0.2, 0.0, CUBE_HALF],
            physicsClientId=self.c,
        )
        p.changeDynamics(
            self.cube,
            -1,
            lateralFriction=1.0,
            spinningFriction=float(m.geom_friction[cg][1]),
            rollingFriction=0.0,  # condim 4 in MuJoCo: sliding + torsional, no rolling
            linearDamping=0,
            angularDamping=0,
            physicsClientId=self.c,
        )

        # the arm: MuJoCo bodies base (fixed) -> shoulder ... moving jaw, plus the two pads as fixed child links
        chain = [
            "shoulder",
            "upper_arm",
            "lower_arm",
            "wrist",
            "gripper",
            "moving_jaw_so101_v1",
        ]
        ids = [body(n) for n in chain]
        base = body("base")
        L = {
            k: []
            for k in (
                "mass",
                "col",
                "vis",
                "pos",
                "orn",
                "ipos",
                "iorn",
                "parent",
                "type",
                "axis",
            )
        }
        self.inertia, link_of_body = [], {base: 0}
        for bid in ids:
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
            jid = [j for j in range(m.njnt) if m.jnt_bodyid[j] == bid][0]
            L["mass"].append(float(m.body_mass[bid]))
            L["col"].append(
                self._shapes(
                    [g for g in geoms if m.geom_contype[g] and g not in self.pads], True
                )
            )
            L["vis"].append(
                self._shapes([g for g in geoms if m.geom_group[g] < 3], False)
            )
            L["pos"].append(m.body_pos[bid].tolist())
            L["orn"].append(_xyzw(m.body_quat[bid]))
            L["parent"].append(link_of_body[m.body_parentid[bid]])
            L["type"].append(p.JOINT_REVOLUTE)
            L["axis"].append(m.jnt_axis[jid].tolist())
            # armature: the motor's reflected inertia about the joint axis, added to the link's inertia tensor
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.body_iquat[bid])
            R = R.reshape(3, 3)
            a = m.jnt_axis[jid] / np.linalg.norm(m.jnt_axis[jid])
            inert = R @ np.diag(m.body_inertia[bid]) @ R.T + m.dof_armature[
                m.jnt_dofadr[jid]
            ] * np.outer(a, a)
            w, V = np.linalg.eigh(inert)
            if np.linalg.det(V) < 0:
                V[:, 0] *= -1
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, V.flatten())
            L["ipos"].append(m.body_ipos[bid].tolist())
            L["iorn"].append(_xyzw(q))
            self.inertia.append(w.tolist())
            link_of_body[bid] = len(L["mass"])  # multibody link index + 1 (0 = base)
        self.pad_links = []
        for g in sorted(self.pads):
            L["mass"].append(PAD_MASS)
            L["col"].append(self._shapes([g], True))
            L["vis"].append(-1)
            L["pos"].append(m.geom_pos[g].tolist())
            L["orn"].append(_xyzw(m.geom_quat[g]))
            L["ipos"].append([0, 0, 0])
            L["iorn"].append([0, 0, 0, 1])
            L["parent"].append(link_of_body[m.geom_bodyid[g]])
            L["type"].append(p.JOINT_FIXED)
            L["axis"].append([0, 0, 1])
            hx, hy, hz = m.geom_size[g]
            self.inertia.append([PAD_MASS / 3 * (hy * hy + hz * hz), PAD_MASS / 3 * (hx * hx + hz * hz), PAD_MASS / 3 * (hx * hx + hy * hy)])
            self.pad_links.append(len(L["mass"]) - 1)
        base_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == base]
        self.robot = p.createMultiBody(
            0,
            self._shapes([g for g in base_geoms if m.geom_contype[g]], True),
            self._shapes([g for g in base_geoms if m.geom_group[g] < 3], False),
            basePosition=d.xpos[base].tolist(),
            baseOrientation=_xyzw(d.xquat[base]),
            linkMasses=L["mass"],
            linkCollisionShapeIndices=L["col"],
            linkVisualShapeIndices=L["vis"],
            linkPositions=L["pos"],
            linkOrientations=L["orn"],
            linkInertialFramePositions=L["ipos"],
            linkInertialFrameOrientations=L["iorn"],
            linkParentIndices=L["parent"],
            linkJointTypes=L["type"],
            linkJointAxis=L["axis"],
            physicsClientId=self.c,
        )
        for li in self.pad_links:
            g = sorted(self.pads)[self.pad_links.index(li)]
            p.changeDynamics(
                self.robot,
                li,
                lateralFriction=float(m.geom_friction[g][0]),
                spinningFriction=1.0,
                physicsClientId=self.c,
            )
        self.joints = list(range(len(chain)))  # revolute links 0..5 = JOINTS order
        for j, jid in zip(self.joints, [m.joint(n).id for n in JOINTS]):
            lo, hi = m.jnt_range[jid]
            p.changeDynamics(
                self.robot,
                j,
                jointLowerLimit=float(lo),
                jointUpperLimit=float(hi),
                physicsClientId=self.c,
            )
        # collisions as in MuJoCo: no arm self-contact; the jaw meshes skip the cube, the pads grip it
        n_links = len(L["mass"])
        for a_ in range(-1, n_links):
            for b_ in range(a_ + 1, n_links):
                p.setCollisionFilterPair(
                    self.robot, self.robot, a_, b_, 0, physicsClientId=self.c
                )
        for j in (chain.index("gripper"), chain.index("moving_jaw_so101_v1")):
            p.setCollisionFilterPair(
                self.robot, self.cube, j, -1, 0, physicsClientId=self.c
            )
        p.setJointMotorControlArray(
            self.robot,
            self.joints,
            p.VELOCITY_CONTROL,
            forces=[0.0] * 6,
            physicsClientId=self.c,
        )
        for li, inert in enumerate(self.inertia):
            p.changeDynamics(
                self.robot,
                li,
                localInertiaDiagonal=inert,
                lateralFriction=1.0,
                spinningFriction=1.0,
                physicsClientId=self.c,
            )
        # (a link-level call that also sets damping is rejected whole by Bullet; damping is per multibody)
        p.changeDynamics(self.robot, -1, linearDamping=0, angularDamping=0, physicsClientId=self.c)
        self.link_mass = np.array(L["mass"])
        self.link_axis = [np.array(a_) / np.linalg.norm(a_) for a_ in L["axis"]]
        self.gravity = np.array(m.opt.gravity)
        parent = [pi - 1 for pi in L["parent"]]  # -1 = base
        self.subtree = []
        for j in range(len(L["mass"])):
            sub = []
            for k in range(len(L["mass"])):
                a_ = k
                while a_ >= 0 and a_ != j:
                    a_ = parent[a_]
                if a_ == j:
                    sub.append(k)
            self.subtree.append(sub)
        act = [m.actuator(n).id for n in JOINTS]
        self.kp = m.actuator_gainprm[act, 0].copy()
        self.flim = m.actuator_forcerange[act].copy()
        dof = [m.jnt_dofadr[m.joint(n).id] for n in JOINTS]
        self.damp, self.floss = (
            m.dof_damping[dof].copy(),
            m.dof_frictionloss[dof].copy(),
        )

    def _shapes(self, geoms: list[int], collision: bool) -> int:
        p, m = self.p, self.scene.model
        if not geoms:
            return -1
        kw = {
            k: []
            for k in (
                "shapeTypes",
                "radii",
                "halfExtents",
                "lengths",
                "fileNames",
                "meshScales",
                "planeNormals",
            )
        }
        fp, fo, cols = [], [], []
        for g in geoms:
            t, s = mujoco.mjtGeom(m.geom_type[g]), m.geom_size[g]
            st = {
                mujoco.mjtGeom.mjGEOM_BOX: p.GEOM_BOX,
                mujoco.mjtGeom.mjGEOM_CYLINDER: p.GEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_MESH: p.GEOM_MESH,
                mujoco.mjtGeom.mjGEOM_PLANE: p.GEOM_PLANE,
            }[t]
            kw["shapeTypes"].append(st)
            kw["radii"].append(float(s[0]))
            kw["halfExtents"].append(
                s.tolist() if t != mujoco.mjtGeom.mjGEOM_PLANE else [3, 3, 0.1]
            )
            kw["lengths"].append(float(2 * s[1]))
            kw["fileNames"].append(
                self.mesh_file[m.geom_dataid[g]]
                if t == mujoco.mjtGeom.mjGEOM_MESH
                else ""
            )
            kw["meshScales"].append([1, 1, 1])
            kw["planeNormals"].append([0, 0, 1])
            same_frame = m.geom_bodyid[g] in (0, m.body("bowl").id) or g in self.pads
            fp.append([0, 0, 0] if same_frame else m.geom_pos[g].tolist())
            fo.append([0, 0, 0, 1] if same_frame else _xyzw(m.geom_quat[g]))
            cols.append(_rgba(m, g))
        if collision:
            return p.createCollisionShapeArray(
                **kw,
                collisionFramePositions=fp,
                collisionFrameOrientations=fo,
                physicsClientId=self.c,
            )
        return p.createVisualShapeArray(
            **kw,
            rgbaColors=cols,
            visualFramePositions=fp,
            visualFrameOrientations=fo,
            physicsClientId=self.c,
        )

    def reset(self, q: np.ndarray, cube_xyz) -> None:
        p = self.p
        for j, v in zip(self.joints, q):
            p.resetJointState(self.robot, j, float(v), 0.0, physicsClientId=self.c)
        p.resetBasePositionAndOrientation(
            self.cube,
            [float(cube_xyz[0]), float(cube_xyz[1]), float(cube_xyz[2])],
            [0, 0, 0, 1],
            physicsClientId=self.c,
        )
        p.resetBaseVelocity(self.cube, [0, 0, 0], [0, 0, 0], physicsClientId=self.c)

    def step(self, target: np.ndarray) -> None:
        """One MuJoCo-length step (1/300 s): the servo law is re-evaluated on every Bullet substep."""
        p = self.p
        for _ in range(self.substeps):
            st = p.getJointStates(self.robot, self.joints, physicsClientId=self.c)
            q = np.array([s[0] for s in st])
            qd = np.array([s[1] for s in st])
            tau = np.clip(self.kp * (target - q), self.flim[:, 0], self.flim[:, 1])
            tau = tau - self.damp * qd - self.floss * np.tanh(qd / 0.01) + self._gravcomp()
            p.setJointMotorControlArray(self.robot, self.joints, p.TORQUE_CONTROL, forces=tau.tolist(), physicsClientId=self.c)
            p.stepSimulation(physicsClientId=self.c)

    def _gravcomp(self) -> np.ndarray:
        """Joint torques that cancel gravity on every arm link (MuJoCo's gravcomp=1), summed by hand from the link
        centres of mass: Bullet's inverse-dynamics tree refuses the armature-augmented inertias."""
        p = self.p
        ls = p.getLinkStates(self.robot, list(range(len(self.link_mass))), computeForwardKinematics=1, physicsClientId=self.c)
        com = np.array([s_[0] for s_ in ls])
        g = self.gravity
        tau = np.zeros(len(self.joints))
        for j in self.joints:
            o = np.array(ls[j][4])
            R = np.array(p.getMatrixFromQuaternion(ls[j][5])).reshape(3, 3)
            z = R @ self.link_axis[j]
            for k in self.subtree[j]:
                tau[j] -= z @ np.cross(com[k] - o, self.link_mass[k] * g)
        return tau

    def cube_pos(self) -> np.ndarray:
        return np.array(
            self.p.getBasePositionAndOrientation(self.cube, physicsClientId=self.c)[0]
        )

    def render(self, w: int, h: int, eye, target, fovy: float) -> np.ndarray:
        p = self.p
        view = p.computeViewMatrix(eye, target, [0, 0, 1], physicsClientId=self.c)
        proj = p.computeProjectionMatrixFOV(
            fovy, w / h, 0.01, 5.0, physicsClientId=self.c
        )
        img = p.getCameraImage(
            w,
            h,
            view,
            proj,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=[0.4, 0.6, 1.0],
            shadow=1,
            physicsClientId=self.c,
        )
        return np.reshape(np.array(img[2], np.uint8), (h, w, 4))[:, :, :3]


def replay_bullet(b: Bullet, state0, actions, cube0, on_frame=None) -> dict:
    """Open-loop replay of the recorded 30 fps action stream (held for SETTLE_FRAMES at the end)."""
    b.reset(from_lerobot_units(state0), cube0)
    lifted = False
    tgt = [from_lerobot_units(a) for a in actions]
    for k in range(len(tgt) + SETTLE_FRAMES):
        t = tgt[min(k, len(tgt) - 1)]
        for _ in range(STEPS_PER_FRAME):
            b.step(t)
        c = b.cube_pos()
        lifted |= bool(c[2] > CUBE_HALF + 0.02)
        if on_frame is not None:
            on_frame(k)
    c = b.cube_pos()
    return {"success": bool(lifted and in_bowl(c)), "lifted": lifted, "cube_end": c.round(4).tolist()}


def replay_mujoco(scene: Scene, state0, actions, cube0, on_frame=None) -> dict:
    """The same open-loop replay in MuJoCo (ctrl = the recorded 30 fps actions, zero-order hold)."""
    m, d = scene.model, scene.data
    q0 = from_lerobot_units(state0)
    scene.reset(q0[:5], q0[5], cube0[:2], 0.0)
    lifted = False
    for k in range(len(actions) + SETTLE_FRAMES):
        d.ctrl[:] = from_lerobot_units(actions[min(k, len(actions) - 1)])
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(m, d)
        c = scene.cube_pos()
        lifted |= bool(c[2] > CUBE_HALF + 0.02)
        if on_frame is not None:
            on_frame(k)
    c = scene.cube_pos()
    return {"success": bool(lifted and in_bowl(c)), "lifted": lifted, "cube_end": c.round(4).tolist()}


def film(run_dir: Path, ep: dict, w: int = 960, h: int = 720) -> dict:
    """Film one episode's replay in both engines from the same viewpoint (the film's hero camera): MuJoCo renders
    with its own OpenGL renderer, PyBullet with its own TinyRenderer. Writes crosscheck/film_{mujoco,bullet}.mp4."""
    from .media import Encoder, SimFilm, hero_cam
    from .scene import load

    cam = hero_cam()
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    target = np.array(cam.lookat)
    eye = target - cam.distance * fwd
    scene = load(task="pick")
    fovy = float(scene.model.vis.global_.fovy)
    out = run_dir / "crosscheck"
    sf = SimFilm(scene, out / "film_mujoco.mp4", w=w, h=h, camera=cam)
    rm = replay_mujoco(scene, ep["state"][0], ep["action"], ep["env"][0], on_frame=sf.frame)
    sf.close()
    b = Bullet(load(task="pick"), out / "meshes")
    enc = Encoder(out / "film_bullet.mp4", w, h, 30)
    rb = replay_bullet(b, ep["state"][0], ep["action"], ep["env"][0],
                       on_frame=lambda k: enc.add(b.render(w, h, eye.tolist(), target.tolist(), fovy)))
    enc.close()
    return {"episode": ep["episode"], "clip": ep["clip"], "mujoco": rm, "bullet": rb, "eye": eye.round(4).tolist(),
            "target": target.round(4).tolist(), "fovy": fovy}


def episodes(ds_root: Path) -> list[dict]:
    from .dataset import load_arrays

    meta = json.loads((ds_root / "rohub_episodes.json").read_text())
    arr = load_arrays(ds_root, meta["info"]["repo_id"])
    out = []
    for e in meta["episodes"]:
        k = arr["episode"] == e["episode_index"]
        out.append(
            {
                "episode": int(e["episode_index"]),
                "clip": e["clip"],
                "origin": e.get("kind", e.get("origin", "")),
                "state": arr["state"][k],
                "action": arr["action"][k],
                "env": arr["env"][k],
            }
        )
    return out


def main() -> None:
    from .scene import load

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--film", type=int, default=None, help="film this episode in both engines instead")
    a = ap.parse_args()
    import pybullet

    if a.film is not None:
        ep = episodes(a.run_dir / "lerobot")[a.film]
        info = film(a.run_dir, ep)
        (a.run_dir / "crosscheck" / "film.json").write_text(json.dumps(info, indent=1))
        print(f"[film] episode {a.film}: MuJoCo {info['mujoco']['success']}, PyBullet {info['bullet']['success']}")
        return
    scene = load(task="pick")
    b = Bullet(load(task="pick"), a.run_dir / "crosscheck" / "meshes")
    eps = episodes(a.run_dir / "lerobot")[: a.limit]
    print(
        f"[crosscheck] {len(eps)} accepted episodes from {a.run_dir / 'lerobot'}; engines: MuJoCo {mujoco.__version__}, "
        f"PyBullet {pybullet.__version__ if hasattr(pybullet, '__version__') else 'API ' + str(pybullet.getAPIVersion())}",
        flush=True,
    )
    rows, t0 = [], time.time()
    for e in eps:
        rm = replay_mujoco(scene, e["state"][0], e["action"], e["env"][0])
        rb = replay_bullet(b, e["state"][0], e["action"], e["env"][0])
        gap = float(np.linalg.norm(np.array(rm["cube_end"]) - np.array(rb["cube_end"])))
        rows.append(
            {
                "episode": e["episode"],
                "clip": e["clip"],
                "frames": int(len(e["action"])),
                "mujoco": rm,
                "bullet": rb,
                "agree": rm["success"] == rb["success"],
                "end_gap_m": round(gap, 4),
            }
        )
        print(
            f"[crosscheck] episode {e['episode']:>2} ({e['clip']})  MuJoCo {'IN BOWL' if rm['success'] else 'miss   '}  "
            f"PyBullet {'IN BOWL' if rb['success'] else 'miss   '}  final cube {100 * gap:.1f} cm apart",
            flush=True,
        )
    km = sum(r["mujoco"]["success"] for r in rows)
    kb = sum(r["bullet"]["success"] for r in rows)
    both = sum(r["mujoco"]["success"] and r["bullet"]["success"] for r in rows)
    agree = sum(r["agree"] for r in rows)
    print(
        f"[crosscheck] MuJoCo {km}/{len(rows)} in the bowl, PyBullet {kb}/{len(rows)}, both {both}/{len(rows)}, "
        f"engines agree on {agree}/{len(rows)}; median final-cube gap {100 * np.median([r['end_gap_m'] for r in rows]):.1f} cm "
        f"({time.time() - t0:.0f} s)",
        flush=True,
    )
    out = a.run_dir / "crosscheck" / "crosscheck.json"
    out.write_text(
        json.dumps(
            {
                "episodes": len(rows),
                "mujoco_in_bowl": km,
                "bullet_in_bowl": kb,
                "both": both,
                "agree": agree,
                "seconds": round(time.time() - t0, 1),
                "rows": rows,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
