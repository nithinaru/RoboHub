"""Instant MuJoCo preview: any sentence -> a small plan -> a scripted SO-101 motion -> a short mp4.

This is NOT a learned policy. The arm follows a hand-written script: the pick expert's grasp IK (pick_expert.GraspIK,
jaws straight down, closing axis on a horizontal heading) tracks eased end-effector waypoints, and the objects are
mocap bodies moved kinematically (a grasped object rides with the gripper frame, a pushed one slides ahead of the
closed jaws, a released one drops to its rest height, a book cover turns about its hinge with the held edge). No
physics steps run, so a preview renders in a few seconds on this Mac and never fails on a slipped grasp.

Plan (JSON, also what the page shows):
  {"family": "pick_place" | "push" | "stack" | "tower" | "lift" | "hinge_open" | "hinge_close",
   "objects": [{"name": "o0", "kind": "block"|"ball"|"cup"|"bowl"|"plate"|"pad"|"box"|"book", "color": "red",
                "xy": [x, y]}, ...],
   "steps": [{"op": "pick_place", "obj": "o0", "target": "o1"}, {"op": "push", "obj": "o0", "target": "o1"},
             {"op": "lift", "obj": "o0"}, {"op": "hinge", "obj": "o0", "open": true}],
   "note": "what was substituted, if anything"}

parse(sentence) is a deterministic keyword parser; parse_with_claude() asks headless `claude -p` for the same JSON
when the keywords find nothing (10 s timeout, falls back to the keyword plan).

CLI: python -m rohub.preview "put the red block in the bowl" [--out f.mp4]
     python -m rohub.preview --plan plan.json --out f.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from .media import Encoder, hero_cam
from .pick_expert import CENTRE_GRASP, CENTRE_OPEN, GRIP_OPEN, GraspIK, site_for_cube
from .scene import GRIPPER_CLOSED, HOME, Scene, build_spec

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "web-runs" / "_previews"
W, H, FPS = 480, 270, 25
CAPTION = "MuJoCo preview · scripted"

# ---------------------------------------------------------------- vocabulary

COLORS = {
    "red": (0.90, 0.15, 0.12),
    "green": (0.15, 0.70, 0.25),
    "blue": (0.15, 0.35, 0.85),
    "yellow": (0.95, 0.80, 0.10),
    "orange": (0.95, 0.50, 0.10),
    "purple": (0.55, 0.25, 0.75),
    "pink": (0.95, 0.45, 0.65),
    "white": (0.93, 0.93, 0.90),
    "black": (0.12, 0.12, 0.13),
    "gray": (0.55, 0.55, 0.57),
    "grey": (0.55, 0.55, 0.57),
    "brown": (0.50, 0.32, 0.18),
}
NOUNS = {  # word -> kind
    "block": "block",
    "cube": "block",
    "brick": "block",
    "blocks": "block",
    "cubes": "block",
    "ball": "ball",
    "sphere": "ball",
    "orange": "ball",
    "apple": "ball",
    "marble": "ball",
    "cup": "cup",
    "mug": "cup",
    "glass": "cup",
    "bowl": "bowl",
    "plate": "plate",
    "dish": "plate",
    "tray": "plate",
    "saucer": "plate",
    "square": "pad",
    "pad": "pad",
    "mat": "pad",
    "target": "pad",
    "tape": "pad",
    "zone": "pad",
    "circle": "pad",
    "box": "box",
    "bin": "box",
    "basket": "box",
    "container": "box",
    "book": "book",
    "lid": "book",
    "laptop": "book",
    "notebook": "book",
    "cover": "book",
}
MOVABLE = {"block", "ball"}
CONTAINERS = {"bowl", "cup", "plate", "pad", "box"}
DEFAULT_COLOR = {
    "block": "red",
    "ball": "orange",
    "cup": "white",
    "bowl": "white",
    "plate": "white",
    "pad": "blue",
    "box": "brown",
    "book": "blue",
}
SRC_SPOTS = [(0.16, 0.10), (0.15, -0.11), (0.23, 0.13), (0.22, -0.14)]
TARGET_SPOT = (0.25, 0.0)
UNKNOWN_THING = re.compile(r"\b(the|a|an|my|this|that)\s+(?:(\w+)\s+)?(\w+)")


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def _mentions(words: list[str]) -> list[tuple[str, str | None, str]]:
    """(kind, colour or None, word) for every object noun, in sentence order; a colour binds to the next noun."""
    out, col = [], None
    for w in words:
        if w in COLORS:
            col = w
            continue
        if w in NOUNS:
            out.append((NOUNS[w], col, w))
            col = None
    return out


def parse(sentence: str) -> dict:
    """Deterministic keyword parser: sentence -> plan (always returns one; `note` says what was approximated)."""
    s = _norm(sentence)
    words = s.split()
    # "orange" is a colour when a noun follows it
    fixed = []
    for i, w in enumerate(words):
        if w == "orange" and i + 1 < len(words) and words[i + 1] in NOUNS:
            fixed.append("orange")  # colour
        elif w == "orange":
            fixed.append("ball_orange")
        else:
            fixed.append(w)
    words = ["ball" if w == "ball_orange" else w for w in fixed]
    ments = _mentions(words)
    note = ""
    has = lambda *ks: any(re.search(rf"\b{k}\b", s) for k in ks)  # noqa: E731

    def obj(kind, color=None):
        return {"kind": kind, "color": color or DEFAULT_COLOR[kind]}

    # hinge: open / close a book or lid
    if has("open", "close", "shut", "flip open", "lift the lid"):
        if any(k == "book" for k, _, _ in ments) or has(
            "lid", "book", "laptop", "cover", "box"
        ):
            col = next((c for k, c, _ in ments if k in ("book", "box") and c), None)
            opening = not has("close", "shut")
            note = (
                ""
                if has("book", "lid", "cover", "laptop", "notebook")
                else "a hinged book stands in for the lid"
            )
            return _layout(
                "hinge_open" if opening else "hinge_close",
                [obj("book", col)],
                [{"op": "hinge", "obj": 0, "open": opening}],
                note,
            )

    blocks = [(c, w) for k, c, w in ments if k in MOVABLE]
    targets = [(k, c, w) for k, c, w in ments if k in CONTAINERS]

    # tower of three / stack all
    if has("tower", "three", "3 blocks", "all the blocks", "all blocks"):
        cols = [c for c, _ in blocks if c] or []
        for c in ("blue", "red", "green", "yellow"):
            if len(cols) >= 3:
                break
            if c not in cols:
                cols.append(c)
        # bottom first: the last named colour is usually the base ("stack red on green on blue")
        base, mid, top = cols[2], cols[1], cols[0]
        return _layout(
            "tower",
            [obj("block", base), obj("block", mid), obj("block", top)],
            [
                {"op": "pick_place", "obj": 1, "target": 0},
                {"op": "pick_place", "obj": 2, "target": 1},
            ],
            note,
        )

    # stack X on Y (two movable things, or "stack"/"on top")
    if (
        has("stack", "on top") or (len(blocks) >= 2 and has("on", "onto"))
    ) and not targets:
        a = blocks[0] if blocks else ("red", "block")
        b = (
            blocks[1]
            if len(blocks) > 1
            else (("blue" if a[0] != "blue" else "green"), "block")
        )
        ka = "ball" if a[1] in ("ball", "sphere", "apple", "marble") else "block"
        return _layout(
            "stack",
            [obj(ka, a[0]), obj("block", b[0])],
            [{"op": "pick_place", "obj": 0, "target": 1}],
            note,
        )

    # the movable thing
    if blocks:
        mcol, mword = blocks[0]
        mkind = "ball" if NOUNS.get(mword) == "ball" else "block"
    else:
        mcol, mkind = None, "block"
        unknown = [
            m
            for m in UNKNOWN_THING.findall(s)
            if m[2] not in NOUNS and m[2] not in COLORS
        ]
        if unknown:
            thing = unknown[0][2]
            mcol = unknown[0][1] if unknown[0][1] in COLORS else None
            note = f"a block stands in for the {thing}"
    tkind, tcol = (targets[0][0], targets[0][1]) if targets else (None, None)

    if has("push", "slide", "shove", "nudge", "sweep", "drag"):
        tkind = tkind if tkind in ("pad", "plate") else "pad"
        if targets and targets[0][0] not in ("pad", "plate"):
            note = (
                note + "; " if note else ""
            ) + f"pushed onto a pad (the {targets[0][2]} is not pushable into)"
        return _layout(
            "push",
            [obj(mkind, mcol), obj(tkind, tcol)],
            [{"op": "push", "obj": 0, "target": 1}],
            note,
        )

    if (
        has(
            "put",
            "place",
            "move",
            "drop",
            "set",
            "transfer",
            "bring",
            "carry",
            "pick",
            "grab",
            "take",
            "load",
            "insert",
            "throw",
            "toss",
            "give",
            "hand",
        )
        or targets
    ):
        if tkind is None:
            if has("pick up", "lift", "grab", "take", "raise", "hold") and not has(
                "put", "place", "move"
            ):
                return _layout(
                    "lift", [obj(mkind, mcol)], [{"op": "lift", "obj": 0}], note
                )
            tkind = "pad"
            note = (note + "; " if note else "") + "no target named, placed on a pad"
        return _layout(
            "pick_place",
            [obj(mkind, mcol), obj(tkind, tcol)],
            [{"op": "pick_place", "obj": 0, "target": 1}],
            note,
        )

    if has("lift", "raise", "hold", "pick"):
        return _layout("lift", [obj(mkind, mcol)], [{"op": "lift", "obj": 0}], note)

    note = (
        note + "; " if note else ""
    ) + "no known verb, shown as the nearest primitive (pick and place)"
    return _layout(
        "pick_place",
        [obj(mkind, mcol), obj("pad", None)],
        [{"op": "pick_place", "obj": 0, "target": 1}],
        note,
        fallback=True,
    )


def _layout(family, objs, steps, note="", fallback=False) -> dict:
    """Name the objects and place them on the table (movables at the source spots, the target at the far spot)."""
    src = iter(SRC_SPOTS)
    for i, o in enumerate(objs):
        o["name"] = f"o{i}"
    if family == "tower":
        objs[0]["xy"] = [0.21, 0.0]
        objs[1]["xy"], objs[2]["xy"] = list(SRC_SPOTS[0]), list(SRC_SPOTS[1])
    elif family.startswith("hinge"):
        objs[0]["xy"] = [0.21, 0.0]
    else:
        for o in objs:
            is_target = any(st.get("target") == int(o["name"][1:]) for st in steps)
            o["xy"] = list(TARGET_SPOT) if is_target else list(next(src))
    for st in steps:
        for k in ("obj", "target"):
            if isinstance(st.get(k), int):
                st[k] = f"o{st[k]}"
    plan = {"family": family, "objects": objs, "steps": steps, "note": note}
    if fallback:
        plan["fallback"] = True
    return plan


CLAUDE_PROMPT = """Turn this robot-arm task into a JSON plan for a scripted tabletop demo. Reply with JSON only.
Schema: {"family": one of pick_place|push|stack|tower|lift|hinge_open|hinge_close,
 "objects": [{"kind": one of block|ball|cup|bowl|plate|pad|box|book, "color": one of red|green|blue|yellow|orange|purple|pink|white|black|gray|brown}],
 "steps": [{"op": "pick_place"|"push"|"lift"|"hinge", "obj": <index>, "target": <index, omit for lift/hinge>, "open": <bool, hinge only>}],
 "note": "<what you substituted, e.g. 'a block stands in for the banana'>"}
Only block and ball can be carried or pushed. Use at most 3 objects. Task: """


def parse_with_claude(sentence: str, timeout: float = 10.0) -> dict | None:
    exe = shutil.which(
        "claude",
        path="/usr/local/bin:/opt/homebrew/bin:" + str(Path.home() / ".local/bin"),
    )
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "-p", "--model", "haiku", CLAUDE_PROMPT + json.dumps(sentence)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        m = re.search(r"\{.*\}", r.stdout, re.S)
        p = json.loads(m.group(0)) if m else None
        objs = [
            {"kind": o["kind"], "color": o.get("color") or DEFAULT_COLOR[o["kind"]]}
            for o in p["objects"][:3]
            if o.get("kind") in NOUNS.values()
            and (o.get("color") in COLORS or not o.get("color"))
        ]
        steps = []
        for st in p["steps"][:3]:
            if st.get("op") not in (
                "pick_place",
                "push",
                "lift",
                "hinge",
            ) or not 0 <= int(st["obj"]) < len(objs):
                continue
            st = {k: st[k] for k in ("op", "obj", "target", "open") if k in st}
            if "target" in st and not 0 <= int(st["target"]) < len(objs):
                continue
            steps.append(st)
        if not objs or not steps or p.get("family") not in FAMILIES:
            return None
        return _layout(
            p["family"], objs, steps, str(p.get("note", ""))[:120] + " (plan by Claude)"
        )
    except Exception:  # noqa: BLE001 - any failure falls back to the keyword plan
        return None


FAMILIES = ("pick_place", "push", "stack", "tower", "lift", "hinge_open", "hinge_close")


def plan_for(sentence: str, use_claude: bool = True) -> dict:
    p = parse(sentence)
    if p.get("fallback") and use_claude:
        c = parse_with_claude(sentence)
        if c:
            return c
    return p


def plan_key(plan: dict) -> str:
    core = {k: plan[k] for k in ("family", "objects", "steps")}
    return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def caption(plan: dict) -> str:
    def name(o):
        return f"{o['color']} {o['kind']}"

    by = {o["name"]: o for o in plan["objects"]}
    parts = []
    for st in plan["steps"]:
        o = by[st["obj"]]
        if st["op"] == "hinge":
            parts.append(f"{'open' if st.get('open', True) else 'close'} the {name(o)}")
        elif st["op"] == "lift":
            parts.append(f"lift the {name(o)}")
        else:
            t = by[st["target"]]
            prep = "into" if t["kind"] in ("bowl", "cup", "box") else "onto"
            parts.append(
                f"{'push' if st['op'] == 'push' else 'place'} the {name(o)} {prep} the {name(t)}"
            )
    return ", then ".join(parts)


# ---------------------------------------------------------------- scene

HALF = 0.015  # block half size / ball radius
SIZES = {  # kind -> resting half height of the object's top surface used for placing ON it, rest-on height
    "block": 2 * HALF,
    "ball": 2 * HALF,
    "pad": 0.001,
    "plate": 0.006,
    "bowl": 0.004,
    "cup": 0.004,
    "box": 0.004,
}
RIM = {"bowl": 0.028, "cup": 0.042, "box": 0.035}
BOOK_L, BOOK_W, BOOK_T = 0.06, 0.03, 0.008  # book half length (x), half width (y), pages half thickness
COVER_OPEN = 1.75  # rad the arm turns the cover to (past vertical); gravity lays it flat the rest of the way


def _rgba(color):
    return [*COLORS.get(color, (0.6, 0.6, 0.6)), 1.0]


def build(plan: dict) -> tuple[Scene, dict]:
    spec = build_spec(task="push")
    for g in list(spec.geoms):
        if g.name.startswith("tape"):
            spec.delete(g)
    spec.delete(next(b for b in spec.bodies if b.name == "cube"))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = (
        max(W, 640),
        max(H, 480),
    )
    world = spec.worldbody
    vis = {"contype": 0, "conaffinity": 0}
    for o in plan["objects"]:
        k, n = o["kind"], o["name"]
        spec.add_material(name=f"{n}_mat", rgba=_rgba(o["color"]))
        m = f"{n}_mat"
        x, y = o["xy"]
        if k == "block":
            b = world.add_body(name=n, pos=[x, y, HALF], mocap=True)
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, material=m, **vis
            )
        elif k == "ball":
            b = world.add_body(name=n, pos=[x, y, HALF], mocap=True)
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[HALF, 0, 0], material=m, **vis
            )
        elif k == "pad":
            b = world.add_body(name=n, pos=[x, y, 0.0], mocap=True)
            t, w = 0.04, 0.006
            for px, py, sx, sy in [
                (t, 0, w, t + w),
                (-t, 0, w, t + w),
                (0, t, t + w, w),
                (0, -t, t + w, w),
            ]:
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[sx, sy, 0.0006],
                    pos=[px, py, 0.0006],
                    material=m,
                    **vis,
                )
        elif k == "plate":
            b = world.add_body(name=n, pos=[x, y, 0.0], mocap=True)
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.06, 0.003, 0],
                pos=[0, 0, 0.003],
                material=m,
                **vis,
            )
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.066, 0.0015, 0],
                pos=[0, 0, 0.0045],
                rgba=[0.8, 0.8, 0.78, 1],
                **vis,
            )
        elif k in ("bowl", "cup"):
            r, h, t = (
                (0.05, RIM["bowl"], 0.006)
                if k == "bowl"
                else (0.033, RIM["cup"], 0.005)
            )
            b = world.add_body(name=n, pos=[x, y, 0.0], mocap=True)
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r + t, 0.002, 0],
                pos=[0, 0, 0.002],
                material=m,
                **vis,
            )
            segs = 20
            half = (r + t / 2) * np.tan(np.pi / segs) * 1.08
            for i in range(segs):
                a = 2 * np.pi * i / segs
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[t / 2, half, h / 2],
                    pos=[(r + t / 2) * np.cos(a), (r + t / 2) * np.sin(a), h / 2],
                    quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                    material=m,
                    **vis,
                )
            if k == "cup":  # handle
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.004, 0.012, 0.014],
                    pos=[0, -(r + 0.016), h * 0.55],
                    material=m,
                    **vis,
                )
        elif k == "box":
            hb, h, t = 0.05, RIM["box"], 0.005
            b = world.add_body(name=n, pos=[x, y, 0.0], mocap=True)
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hb, hb, 0.002],
                pos=[0, 0, 0.002],
                material=m,
                **vis,
            )
            for px, py, sx, sy in [
                (hb, 0, t, hb),
                (-hb, 0, t, hb),
                (0, hb, hb, t),
                (0, -hb, hb, t),
            ]:
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[sx, sy, h / 2],
                    pos=[px, py, h / 2],
                    material=m,
                    **vis,
                )
        elif k == "book":
            # pages block (fixed) + a cover (mocap) hinged along the book's -y edge; the tab sits on the +y edge
            L, Wd, T = BOOK_L, BOOK_W, BOOK_T
            b = world.add_body(name=n, pos=[x, y, 0.0], mocap=True)
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[L, Wd, T],
                pos=[0, 0, T],
                rgba=[0.95, 0.93, 0.85, 1],
                **vis,
            )
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[L, Wd, 0.0015],
                pos=[0, 0, 0.0015],
                material=m,
                **vis,
            )
            c = world.add_body(
                name=f"{n}_cover", pos=[x, y - Wd, 2 * T + 0.002], mocap=True
            )
            c.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[L, Wd, 0.002],
                pos=[0, Wd, 0],
                material=m,
                **vis,
            )
            c.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.012, 0.006, 0.004],
                pos=[0, 2 * Wd + 0.004, 0],
                rgba=[0.95, 0.8, 0.2, 1],
                **vis,
            )
    model = spec.compile()
    sc = Scene(model, mujoco.MjData(model))
    mid = {
        bd.name: model.body(bd.name).mocapid[0]
        for bd in spec.bodies
        if bd.name and model.body(bd.name).mocapid[0] >= 0
    }
    return sc, mid


# ---------------------------------------------------------------- script


def _ease(s):
    return s * s * (3 - 2 * s)


def _quat_z(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


class Animator:
    def __init__(self, sc: Scene, mocap: dict, plan: dict, out: Path):
        self.sc, self.m, self.d, self.mid = sc, sc.model, sc.data, mocap
        self.kind = {o["name"]: o["kind"] for o in plan["objects"]}
        self.ik = GraspIK(sc)
        self.q = HOME.copy()
        self.g = GRIPPER_CLOSED
        self.ee = self.ik.fk(HOME)[0]
        self.home_ee = self.ee.copy()
        self.held: tuple[str, np.ndarray, np.ndarray] | None = None
        self.falling: dict[str, list[float]] = {}  # name -> [rest_z, vz]
        self.push: tuple | None = None
        self.hinge: tuple | None = None
        self.cover_fall: tuple | None = None
        self.renderer = mujoco.Renderer(self.m, H, W)
        self.cam = hero_cam()
        self.enc = Encoder(out, W, H, FPS)
        self.frames = 0
        self.max_err = 0.0
        mujoco.mj_forward(self.m, self.d)

    def pos(self, n):
        return self.d.mocap_pos[self.mid[n]].copy()

    def _apply(self):
        self.d.qpos[self.sc.qadr] = np.append(self.q, self.g)
        mujoco.mj_kinematics(self.m, self.d)
        site = self.m.site("gripperframe").id
        sp, sr = self.d.site_xpos[site], self.d.site_xmat[site].reshape(3, 3)
        if self.held:
            n, rel_p, rel_r = self.held
            self.d.mocap_pos[self.mid[n]] = sp + sr @ rel_p
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (sr @ rel_r).flatten())
            self.d.mocap_quat[self.mid[n]] = q
        for n, st in list(self.falling.items()):
            p = self.d.mocap_pos[self.mid[n]]
            st[1] += 9.8 / FPS
            p[2] = max(st[0], p[2] - st[1] / FPS)
            if p[2] <= st[0]:
                del self.falling[n]
        if self.push:
            n, o0, dirv, reach, goal_off = self.push
            s = float((sp[:2] - o0[:2]) @ dirv) + reach
            off = min(max(0.0, s), goal_off)
            self.d.mocap_pos[self.mid[n]][:2] = o0[:2] + dirv * off
        if self.cover_fall:  # a released cover past vertical falls the rest of the way (0 or pi)
            cid, ang, vel, goal = self.cover_fall
            vel += 12.0 / FPS
            ang = min(goal, ang + vel / FPS) if goal > ang else max(goal, ang - vel / FPS)
            self.cover_fall = None if ang == goal else (cid, ang, vel, goal)
            self.hinge = (cid, ang)
        if self.hinge:
            cid, ang = self.hinge
            self.d.mocap_quat[cid] = [
                np.cos(ang / 2),
                np.sin(ang / 2),
                0,
                0,
            ]  # about +x: the +y edge lifts up and over toward -y
        mujoco.mj_forward(self.m, self.d)

    def frame(self):
        self._apply()
        self.renderer.update_scene(self.d, camera=self.cam)
        self.enc.add(self.renderer.render())
        self.frames += 1

    def move(self, goal=None, dur=0.6, grip=None, path=None):
        """Ease the end effector to goal (or along path(s) for s in 0..1) while ramping the gripper to grip."""
        n = max(1, int(round(dur * FPS)))
        e0, g0 = self.ee.copy(), self.g
        for i in range(1, n + 1):
            s = _ease(i / n)
            self.ee = path(s) if path else e0 + (np.asarray(goal, float) - e0) * s
            self.ee[2] = min(self.ee[2], 0.09)  # the jaws-down reach ceiling (IK error grows above it)
            if grip is not None:
                self.g = g0 + (grip - g0) * s
            self.q, err = self.ik.solve(self.ee, self.q, iters=40)
            self.max_err = max(self.max_err, err)
            self.frame()

    def hold(self, dur):
        for _ in range(int(round(dur * FPS))):
            self.frame()

    # ---- primitives
    def grasp(self, n):
        c = self.pos(n)
        r = np.linalg.norm(c[:2])
        self.ik.heading = np.array([c[0] / r, c[1] / r, 0.0])
        above = site_for_cube(c, self.ik.heading, CENTRE_OPEN)
        self.move([*above[:2], 0.075], 0.8, grip=GRIP_OPEN)
        self.move(above, 0.45)
        self.move(site_for_cube(c, self.ik.heading, CENTRE_GRASP), 0.15)
        self.move(self.ee, 0.3, grip=GRIPPER_CLOSED + 0.35)
        site = self.m.site("gripperframe").id
        sp, sr = (
            self.d.site_xpos[site].copy(),
            self.d.site_xmat[site].reshape(3, 3).copy(),
        )
        orr = np.zeros(9)
        mujoco.mju_quat2Mat(orr, self.d.mocap_quat[self.mid[n]])
        self.held = (n, sr.T @ (c - sp), sr.T @ orr.reshape(3, 3))

    def place(self, n, target):
        t = self.pos(target)
        tk = self.kind[target]
        top = t[2] + (
            SIZES[tk] if tk not in ("block", "ball") else HALF
        )  # target's top surface
        rest = top + HALF
        rim = RIM.get(tk, 0.0)
        release = max(rest + 0.006, rim + HALF + 0.008)
        carry = min(max(0.075, release + 0.015), 0.098)  # jaws-down reach tops out near site z 0.09
        c = self.pos(n)
        off = self.ee - c  # site relative to the held object
        self.move(np.array([*c[:2], carry]) + off, 0.45)
        self.move(np.array([*t[:2], carry]) + off, 0.8)
        self.move(np.array([*t[:2], release]) + off, 0.4)
        self.held = None
        self.falling[n] = [rest, 0.0]
        self.move(self.ee, 0.25, grip=GRIP_OPEN)
        self.move(self.ee + [0, 0, 0.05], 0.35)

    def pick_place(self, n, target):
        self.grasp(n)
        self.place(n, target)

    def lift(self, n):
        self.grasp(n)
        c = self.pos(n)
        off = self.ee - c
        self.move(np.array([*c[:2], 0.095]) + off, 0.7)
        self.hold(0.4)
        self.move(np.array([*c[:2], HALF + 0.004]) + off, 0.6)
        self.held = None
        self.falling[n] = [HALF, 0.0]
        self.move(self.ee, 0.25, grip=GRIP_OPEN)
        self.move(self.ee + [0, 0, 0.05], 0.35)

    def push_to(self, n, target):
        o, t = self.pos(n), self.pos(target)
        dirv = (t - o)[:2]
        dist = float(np.linalg.norm(dirv))
        dirv /= dist
        back = HALF + 0.012  # finger tips this far behind the object centre at contact
        r = np.linalg.norm(o[:2])
        self.ik.heading = np.array([o[0] / r, o[1] / r, 0.0])  # the natural wrist roll; closed jaws push
        z = 0.012
        start = np.array([*(o[:2] - dirv * (back + 0.03)), z])
        end = np.array([*(t[:2] - dirv * back), z])
        self.move([*start[:2], 0.07], 0.8, grip=GRIPPER_CLOSED)
        self.move(start, 0.4)
        self.push = (n, o, dirv, back, dist)
        self.move(end, 1.3)
        self.push = None
        self.move(end + [0, 0, 0.06], 0.4)

    def hinge_book(self, n, opening: bool):
        cid = self.mid[f"{n}_cover"]
        hp = self.d.mocap_pos[
            cid
        ].copy()  # hinge line point (x centre, -y edge, cover height)
        R = 2 * BOOK_W + 0.004  # hinge -> tab
        # opening: lift the free edge past vertical, let go, it falls open flat; closing: the mirror image
        a0, a1 = (0.0, COVER_OPEN) if opening else (np.pi, np.pi - COVER_OPEN)
        self.hinge = (cid, a0)
        self._apply()
        self.ik.heading = np.array([1.0, 0.0, 0.0])  # jaws close across the tab along x

        def tab(a):
            return hp + np.array([0.0, R * np.cos(a), R * np.sin(a)])

        grip_pt = lambda a: tab(a) + np.array([0, 0, -0.004])  # noqa: E731 - site slightly below the tab centre
        self.move(grip_pt(a0) + [0, 0, 0.05], 0.8, grip=GRIP_OPEN)
        self.move(grip_pt(a0), 0.4)
        self.move(self.ee, 0.25, grip=GRIPPER_CLOSED + 0.4)

        def path(s):
            a = a0 + (a1 - a0) * s
            self.hinge = (cid, a)
            return grip_pt(a)

        self.move(dur=1.6, path=path)
        self.cover_fall = (cid, a1, 0.0, np.pi if opening else 0.0)
        self.move(self.ee, 0.25, grip=GRIP_OPEN)
        self.move(self.ee + [0, 0, 0.04], 0.3)

    def close(self):
        self.enc.close()
        self.renderer.close()


def render(plan: dict, out: Path) -> dict:
    t0 = time.time()
    sc, mocap = build(plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    a = Animator(sc, mocap, plan, tmp)
    a.q, _ = a.ik.solve(a.home_ee, HOME)
    if plan["family"] == "hinge_close":
        a.hinge = (mocap[f"{plan['objects'][0]['name']}_cover"], np.pi)
    a.hold(0.3)
    for st in plan["steps"]:
        op = st["op"]
        if op == "pick_place":
            a.pick_place(st["obj"], st["target"])
        elif op == "push":
            a.push_to(st["obj"], st["target"])
        elif op == "lift":
            a.lift(st["obj"])
        elif op == "hinge":
            a.hinge_book(st["obj"], bool(st.get("open", True)))
    a.move(a.home_ee, 0.8, grip=GRIPPER_CLOSED)
    a.hold(0.3)
    a.close()
    tmp.replace(out)
    return {
        "frames": a.frames,
        "seconds": round(a.frames / FPS, 2),
        "render_s": round(time.time() - t0, 2),
        "max_ik_err": round(a.max_err, 4),
        "bytes": out.stat().st_size,
    }


def cached(sentence: str, use_claude: bool = False) -> tuple[dict, Path]:
    plan = plan_for(sentence, use_claude=use_claude)
    return plan, CACHE / f"{plan_key(plan)}.mp4"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sentence", nargs="?")
    ap.add_argument("--plan", help="plan JSON file (skips parsing)")
    ap.add_argument("--out")
    ap.add_argument(
        "--claude",
        action="store_true",
        help="ask claude -p when the keywords find nothing",
    )
    a = ap.parse_args(argv)
    plan = (
        json.loads(Path(a.plan).read_text())
        if a.plan
        else plan_for(a.sentence, use_claude=a.claude)
    )
    out = Path(a.out) if a.out else CACHE / f"{plan_key(plan)}.mp4"
    stats = render(plan, out)
    rec = {"plan": plan, "caption": caption(plan), "file": str(out), **stats}
    out.with_suffix(".json").write_text(json.dumps(rec))
    print(json.dumps(rec))


if __name__ == "__main__":
    sys.exit(main())
