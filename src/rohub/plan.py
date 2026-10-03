"""From a typed task to Runway prompts.

RoboHub targets one embodiment, the SO-101: one 5-DOF arm with a two-finger gripper. A task is accepted only
if a single hand can do it with a pinch grasp of one small rigid object and a place into a container on a table.
Anything needing two hands, tools, deformables, liquids or finger dexterity is refused before any credit is
spent, with the reason.

One task becomes many scenes: every clip varies lighting, table surface, bowl colour and camera angle, so one
prompt yields a spread of demonstrations instead of one.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

REFUSE = [
    (
        r"\b(both|two) hands\b|\bbimanual\b|\bhand to hand\b",
        "needs two hands; the SO-101 is one arm",
    ),
    (
        r"\bpour|\bliquid|\bwater\b|\bcoffee\b|\bjuice\b",
        "liquids and pouring need wrist twist and are not rigid",
    ),
    (
        r"\bfold|\bcloth|\btowel|\bshirt|\bsock|\bstring|\brope|\btie\b",
        "deformable objects; a 2-finger gripper cannot manage cloth or rope",
    ),
    (
        r"\bcut|\bknife|\bscissor|\bhammer|\bscrew|\bdrill|\bpen\b|\bwrite",
        "tool use; out of scope for a single pinch grasp",
    ),
    (
        r"\btype|\bpiano|\bbutton|\bthread|\bneedle",
        "finger dexterity; the SO-101 has two fingers",
    ),
    (
        r"\bopen (the )?(jar|door|drawer|lid)",
        "articulated objects need forces the demo cannot show",
    ),
]
OBJECT_RE = re.compile(
    r"\b(?:the |a |an )?(?P<color>red|blue|green|yellow|orange|purple|white|black)?\s*(?P<obj>block|cube)\b"
)
CONTAINER_RE = re.compile(r"\b(?P<container>bowl|cup|box|basket|tray)\b")
VERB_RE = re.compile(r"\b(put|place|drop|move|pick|set|transfer)\b")


@dataclass(frozen=True)
class Task:
    text: str
    color: str
    obj: str
    container: str


class Infeasible(ValueError):
    pass


def parse_task(text: str) -> Task:
    t = text.lower().strip()
    for pat, why in REFUSE:
        if re.search(pat, t):
            raise Infeasible(f"'{text}': {why}")
    o, c, v = OBJECT_RE.search(t), CONTAINER_RE.search(t), VERB_RE.search(t)
    if not (o and c and v):
        raise Infeasible(
            f"'{text}': RoboHub handles 'put the <colour> block in the <container>' "
            "(one hand, pinch grasp, place). Try 'put the red block in the bowl'."
        )
    color = o.group("color") or "red"
    if color != "red":
        raise Infeasible(
            f"'{text}': this build tracks one red block (the colour tracker and the scene audit are red-only)"
        )
    return Task(
        text=text.strip(),
        color=o.group("color") or "red",
        obj=o.group("obj"),
        container=c.group("container"),
    )


# variation axes (index into each list with the clip number, offset so neighbours differ on every axis)
LIGHTING = [
    "soft natural daylight from a window on the left",
    "warm late-afternoon sunlight with long soft shadows",
    "cool overcast light, flat and even",
    "a warm desk lamp at night, dark background",
    "bright even studio light",
    "mixed daylight with a slight blue tint",
]
TABLE = [
    "a light oak wooden table",
    "a matte white laminate desk",
    "a dark walnut table",
    "a grey concrete worktop",
    "a pale birch plywood workbench",
    "a black matte desk",
]
BOWL = [
    "white ceramic",
    "pale blue ceramic",
    "sage green ceramic",
    "matte black stoneware",
    "cream enamel",
    "light grey ceramic",
]
ANGLE = [
    "a low front view from table height, about 25 degrees down",
    "a front view from slightly to the left, about 35 degrees down",
    "a front view, about 45 degrees down",
]


# ---------------------------------------------------------------------------------------------------------------
# Prompts. Version 2 follows PhyT2V (Xue, Yin, Yang, Gao, "PhyT2V: LLM-Guided Iterative Self-Refinement for
# Physics-Grounded Text-to-Video Generation", CVPR 2025, arXiv:2412.00596). Each PhyT2V round has three steps:
#   Step 1  from the user prompt, list the objects the video must show and the physical rules it must follow;
#   Step 2  caption the generated video and find where it mismatches the prompt;
#   Step 3  rewrite the prompt so it states those rules and resolves those mismatches; the rewrite is the next
#           round's prompt.
# PhyT2V runs every step with an LLM and a video captioner. RoboHub does the same three steps without either:
#   Step 1  scene_spec(): the task grammar is narrow (one hand, one block, one container), so the inventory with
#           exact counts and the rules are written from the parsed task, in positive words (Runway's Gen-4 guide:
#           negative phrasing is not supported and can give the opposite);
#   Step 2  mismatches(): the gates and the Claude scene audit already compare every clip with the task and say
#           what is wrong ("a second red block appears", "Claude counts 2 bowls"); their rejection reasons are
#           the mismatch;
#   Step 3  refine_fixes() + variants(fixes=...): each mismatch maps to one corrective rule, placed first in the prompt that caused it (the
#           first-frame prompt for scene errors, the motion prompt for errors that appear during the motion).
# Its effect on Runway footage is NOT measured yet (no credits were spent on it); PhyT2V's own numbers are for
# the open models in that paper, not for Runway.
# ---------------------------------------------------------------------------------------------------------------

PROMPT_VERSION = 2
PROMPT_MAX_CHARS = 1000  # Runway promptText limit
# The first run's sentence keeps the prompts its cached clips were made with (like its seed, generate.seed_for),
# so replaying it spends nothing. Every other sentence gets the current prompts.
LEGACY_PROMPTS = {"put the red block in the bowl": 1}


def prompt_version_for(task: Task) -> int:
    return LEGACY_PROMPTS.get(" ".join(task.text.lower().split()), PROMPT_VERSION)


@dataclass(frozen=True)
class SceneSpec:
    """PhyT2V Step 1: what the video must contain (with counts) and the physical rules it must follow.
    rules are (topic, sentence); a Step 3 fix on the same topic replaces the sentence."""

    inventory: tuple[tuple[int, str], ...]
    rules: tuple[tuple[str, str], ...]


NUM = {1: "one", 2: "two", 3: "three", 4: "four"}


def scene_spec(task: Task, bowl: str) -> SceneSpec:
    c, o, k = task.color, task.obj, task.container
    return SceneSpec(
        inventory=(
            (1, f"small {c} wooden toy {o} (a 3 cm cube, all six sides equal)"),
            (1, f"shallow {bowl} {k}"),
            (1, "person's right hand"),
        ),
        rules=(
            (
                "rigid",
                f"The {c} {o} is one rigid cube that keeps its size, shape and colour in every frame.",
            ),
            (
                "count",
                f"Exactly one {c} {o} exists in every frame; once lifted, the spot where it stood is bare.",
            ),
            (
                "grip",
                f"The {o} moves only while it is pinched between thumb and index finger.",
            ),
            (
                "release",
                f"When the fingers open above the {k}, that same {o} falls into the {k} and rests there.",
            ),
            ("container", f"The one {k} stays in its place on the table."),
            (
                "hand",
                "The whole hand and wrist stay inside the frame from the first frame to the last.",
            ),
        ),
    )


@dataclass(frozen=True)
class Mismatch:
    """PhyT2V Step 2: one way a generated clip departed from the task, read off a gate's rejection reason."""

    key: str
    target: str  # "frame" (first-frame prompts), "video" (motion prompt) or "both"
    fix: str  # the corrective rule Step 3 puts first in that prompt
    replaces: tuple[
        str, ...
    ] = ()  # topics of base sentences this fix supersedes (kept short: Runway's limit)


# (key, pattern over the gate reason, target, fix, topics it replaces); {c} {o} {k} = colour, object, container.
# First match wins.
MISMATCH_RULES = [
    (
        "extra_container",
        r"counts \d+ red blocks?, ([2-9]|\d{2,}) bowls",
        "frame",
        "Exactly one {k} stands on the table, and it is empty.",
        ("container", "empty"),
    ),
    (
        "duplicate_block",
        r"counts ([2-9]|\d{2,}) red blocks|second red block appears|duplicated the object",
        "both",
        (
            "Exactly one {c} {o} exists in the whole video. After the hand opens, that same {o} lies in the {k}"
            " and the spot where it started is bare table."
        ),
        ("count", "release"),
    ),
    (
        "second_red_object",
        r"second red object",
        "frame",
        (
            "The {c} {o} is the only {c} thing in view; the {k}, the table, the background and the sleeve are"
            " neutral colours."
        ),
        ("only",),
    ),
    (
        "hand_out_of_frame",
        r"hand \+ wrist visible|hand seen on",
        "both",
        "The whole hand and wrist stay inside the frame, near its centre, from the first frame to the last.",
        ("hand",),
    ),
]


def mismatches(reasons: list[str], task: Task) -> list[Mismatch]:
    """Gate rejection reasons -> mismatches (deduplicated, in rule order). Reasons no rule knows are skipped:
    those clips failed on something a prompt cannot fix (frame rate, tracking, robot kinematics)."""
    keys = set()
    for r in reasons:
        for key, pat, *_ in MISMATCH_RULES:
            if re.search(pat, r):
                keys.add(key)
                break
    fmt = {"c": task.color, "o": task.obj, "k": task.container}
    return [
        Mismatch(key, target, fix.format(**fmt), replaces)
        for key, _, target, fix, replaces in MISMATCH_RULES
        if key in keys
    ]


@dataclass(frozen=True)
class Variant:
    clip_id: str
    lighting: str
    table: str
    bowl: str
    angle: str
    image_prompt: str
    video_prompt: str
    restyle_prompt: str = ""
    prompt_version: int = PROMPT_VERSION
    fixes: tuple[str, ...] = ()  # mismatch keys refine() resolved


def _prompts_v1(
    task: Task, lighting: str, table: str, bowl: str, angle: str
) -> tuple[str, str, str]:
    """The prompts the first run's clips were made with, byte for byte (their cache keys)."""
    image_prompt = (
        f"Photorealistic photo, {angle}, of {table} with only two objects on it: exactly one small "
        f"{task.color} wooden toy {task.obj} (a 3 cm cube, all six sides equal) on the left, and one shallow "
        f"{bowl} {task.container} on the right, about 20 cm apart. Nothing else is on the table. A person's "
        f"right hand hovers just above the {task.color} {task.obj}, thumb and index finger open, ready to "
        f"pinch it; the forearm enters from the right edge of the frame. The whole hand, the {task.obj} and "
        f"the {task.container} are fully in frame and unobstructed. {lighting}. Sharp focus, no text."
    )
    video_prompt = (
        f"The right hand pinches the {task.color} {task.obj} between thumb and index finger, lifts it about "
        f"10 cm, carries it to the right and releases it into the {task.container}, then the open hand moves "
        f"back up and away. One continuous smooth motion at natural speed. Static camera, locked off. The "
        f"{task.obj} stays one solid {task.color} cube the whole time and the {task.container} does not move."
    )
    restyle_prompt = (
        f"The same kind of shot as @base: a right hand about to pinch a small red block next to a bowl. "
        f"Change the scene to {table}, a {bowl} bowl, {lighting}, seen from {angle}. "
        f"Keep the whole hand, the block and the bowl fully in frame. Photorealistic, no text."
    )
    return image_prompt, video_prompt, restyle_prompt


def _join(parts: list[tuple[str, str]], fixes: list[Mismatch], target: str) -> str:
    """Step 3: corrective rules first, then the base sentences whose topic no fix replaced. If the result is over
    Runway's limit, optional sentences (topic starting with '~') go first, from the end."""
    fx = [m for m in fixes if m.target in (target, "both")]
    gone = {t for m in fx for t in m.replaces}
    keep = [("fix", m.fix) for m in fx] + [(t, x) for t, x in parts if t not in gone]
    while len(" ".join(x for _, x in keep)) > PROMPT_MAX_CHARS:
        drop = next(
            (i for i in range(len(keep) - 1, -1, -1) if keep[i][0].startswith("~")),
            None,
        )
        if drop is None:
            raise ValueError(
                f"prompt over {PROMPT_MAX_CHARS} characters even without optional sentences"
            )
        keep.pop(drop)
    return " ".join(x for _, x in keep)


def _placement(task: Task) -> tuple[str, str]:
    """Where the sentence puts the object, as the v2 layout sentence and the carry phrase. The sentence's own words
    ("near the back", "on the left") change the scene; without them the layout is the default one."""
    t = task.text.lower()
    o, k = task.obj, task.container
    if re.search(r"\b(back|far side|far end|behind)\b", t):
        return (
            f"The {o} sits on the left near the back of the table, farther from the camera; the {k} sits on the "
            f"right near the front, about 25 cm away.",
            "carries it forward, toward the camera and to the right,",
        )
    if re.search(r"\bon the left\b|\bleft\b", t):
        return (
            f"The {o} sits at the far left of the table and the {k} on the right, about 25 cm apart, both at the "
            "same distance from the camera.",
            "carries it a long way to the right,",
        )
    return (f"The {o} sits on the left and the {k} on the right, about 20 cm apart.", "carries it to the right and")


def _prompts_v2(
    task: Task,
    lighting: str,
    table: str,
    bowl: str,
    angle: str,
    fixes: list[Mismatch] = (),
) -> tuple[str, str, str]:
    """PhyT2V-style prompts: an explicit inventory with counts, the physical rules, and any corrective rules
    from Step 2 placed first (Step 3)."""
    c, o, k = task.color, task.obj, task.container
    spec = scene_spec(task, bowl)
    n = NUM[sum(m for m, _ in spec.inventory)]
    things = ", ".join(f"{NUM[m]} {what}" for m, what in spec.inventory)
    light = lighting[0].upper() + lighting[1:]
    image = [
        ("scene", f"Photorealistic photo, {angle}, of {table}."),
        ("inventory", f"The scene holds exactly {n} things: {things}."),
        ("layout", _placement(task)[0]),
        ("empty", f"The {k} is empty."),
        ("only", f"The {c} {o} is the only {c} object in the picture."),
        (
            "pose",
            (
                f"The right hand hovers just above the {o}, thumb and index finger open, ready to pinch it; "
                "the forearm enters from the right edge."
            ),
        ),
        (
            "hand",
            f"The whole hand and wrist, the {o} and the {k} sit fully inside the frame, clear of its edges.",
        ),
        ("light", f"{light}."),
        ("~style", "Sharp focus, plain clean surfaces."),
    ]
    video = [
        (
            "motion",
            (
                f"The right hand pinches the one {c} {o} between thumb and index finger, lifts it about 10 cm, "
                f"{_placement(task)[1]} opens its fingers above the {k}; the {o} drops into the {k} and "
                "rests there, and the open hand moves back up."
            ),
        ),
        ("~pace", "One continuous smooth motion at natural speed."),
        ("~rules", "Physical rules:"),
        *spec.rules,
        ("camera", "Static camera, locked off."),
    ]
    restyle = [
        (
            "scene",
            (
                f"The same kind of shot as @base: one right hand about to pinch one small {c} {o} next to one "
                f"{k}. Change the scene to {table}, a {bowl} {k}, {lighting}, seen from {angle}. "
                f"{_placement(task)[0]}"
            ),
        ),
        (
            "inventory",
            f"The scene holds exactly {n} things: one {c} {o}, one empty {k}, one hand.",
        ),
        ("only", f"The {o} is the only {c} object."),
        ("hand", f"The whole hand and wrist, the {o} and the {k} stay fully in frame."),
        ("~style", "Photorealistic, plain clean surfaces."),
    ]
    return (
        _join(image, fixes, "frame"),
        _join(video, fixes, "video"),
        _join(restyle, fixes, "frame"),
    )


def variants(
    task: Task, n: int, fixes: dict[str, list[Mismatch]] | None = None
) -> list[Variant]:
    """n scene variations of the task. fixes (clip id -> mismatches, from refine_fixes) rewrites those clips'
    prompts: PhyT2V Step 3. A refined take always uses the current prompt version (a legacy sentence included)."""
    fixes = {k: v for k, v in (fixes or {}).items() if v}
    version = PROMPT_VERSION if fixes else prompt_version_for(task)
    out = []
    for i in range(n):
        cid = f"v{i + 1:02d}"
        lighting = LIGHTING[i % len(LIGHTING)]
        table = TABLE[(i * 5 + 1) % len(TABLE)] if i else TABLE[0]
        bowl = BOWL[(i * 7 + 2) % len(BOWL)] if i else BOWL[0]
        angle = ANGLE[i % len(ANGLE)]
        fx = fixes.get(cid, fixes.get("*", []))
        if version == 1:
            img, vid, rst = _prompts_v1(task, lighting, table, bowl, angle)
        else:
            img, vid, rst = _prompts_v2(task, lighting, table, bowl, angle, fx)
        out.append(
            Variant(
                cid,
                lighting,
                table,
                bowl,
                angle,
                img,
                vid,
                rst,
                version,
                tuple(m.key for m in fx),
            )
        )
    return out


def refine_fixes(verdicts: list[dict], task: Task) -> dict[str, list[Mismatch]]:
    """A finished data stage's verdicts ({clip_id, accepted, reason}) -> the fixes for the next take.

    PhyT2V refines the prompt that failed. Here all clips share one task, so a mismatch seen on any clip is a
    risk for every clip: each clip gets the fixes from every rejection, and a rejected clip's own fixes come
    first. Key "*" holds the fixes for clips that had no verdict."""
    rejected = [v for v in verdicts if not v.get("accepted")]
    everyone = mismatches([v["reason"] for v in rejected], task)
    out = {"*": everyone}
    for v in verdicts:
        own = mismatches([v["reason"]], task) if not v.get("accepted") else []
        out[v["clip_id"]] = own + [
            m for m in everyone if m.key not in {x.key for x in own}
        ]
    return out


def as_dict(v: Variant) -> dict:
    return asdict(v)
