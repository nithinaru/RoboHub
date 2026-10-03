"""Which Runway Model Router a task's footage goes through, decided from the task text by a transparent rubric.

  demo-best   (optimize for quality)  hard tasks: deformables (cloth, laundry, rope, cable), liquids, two hands,
                                      multi-step or long-horizon tasks, precision or insertion. Motion a cheap model
                                      gets wrong and the physics gates would throw away.
  demo-cheap  (optimize for cost)     easy tasks: one rigid object, pick, place, push, stack. The default.
  demo-fast   (optimize for latency)  only when the request is marked live/urgent (live=True from the UI) and the
                                      task is not hard.

Order: any hard cue -> demo-best; else live -> demo-fast; else demo-cheap.
Optionally the headless Claude CLI refines the pick; it falls back to the rubric on any error or after 10 s.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROUTERS = ("demo-best", "demo-fast", "demo-cheap")

BEST_CUES: list[tuple[str, str]] = [
    (
        r"\b(cloth|towel|shirt|t-shirt|sock|laundry|fabric|napkin|blanket|sheet|fold|folding|crumple)\w*",
        "deformable object (cloth)",
    ),
    (
        r"\b(rope|string|cable|cord|wire|thread|knot|tie|lace)\w*",
        "deformable object (rope or cable)",
    ),
    (r"\b(pour|pouring|water|liquid|juice|milk|coffee|tea|spill|fill)\w*", "liquid"),
    (
        r"\b(both hands|two hands|two-handed|bimanual|hand over|handover)\b",
        "two-handed",
    ),
    (
        r"\b(then|after that|and then|next|finally|sequence|stack all|sort|tidy|clean up|set the table)\b",
        "multi-step, long horizon",
    ),
    (
        r"\b(insert|insertion|plug|peg|hole|slot|thread|screw|key|align|precise|precisely|carefully|exactly)\w*",
        "precision or insertion",
    ),
]
RIGID_CUES = r"\b(pick|place|put|move|push|slide|stack|lift|drop|grab|set)\w*"
RIGID_OBJECTS = r"\b(block|cube|box|ball|cup|mug|can|bottle|toy|brick|lego|marker|pen|bowl|plate)s?\b"

CLAUDE = "/usr/local/bin/claude"


def rubric(text: str, live: bool = False) -> dict:
    t = " ".join(text.lower().split())
    hard = [why for pat, why in BEST_CUES if re.search(pat, t)]
    if hard:
        return {"router": "demo-best", "reasons": hard + ["hard task: pay for quality, fewer clips thrown out by the physics gates"],
                "source": "rubric"}
    reasons = []
    if re.search(RIGID_CUES, t):
        reasons.append("single pick, place, push or stack")
    if m := re.search(RIGID_OBJECTS, t):
        reasons.append(f"one rigid object ({m[1]})")
    if live:
        return {"router": "demo-fast", "reasons": reasons + ["marked live: optimize for turnaround"], "source": "rubric"}
    return {"router": "demo-cheap", "reasons": reasons + ["easy task: the cheapest model is enough"], "source": "rubric"}


def _claude(text: str, first: dict, timeout: float, live: bool = False) -> dict | None:
    prompt = (
        "You route robot-demonstration video generation to one of three Runway Model Routers: demo-best (quality: "
        "hard tasks: deformables, liquids, two hands, multi-step, precision), demo-cheap (cost: easy tasks, one rigid "
        "object pick/place/push/stack), demo-fast (latency: only for live/urgent requests that are not hard). "
        f"Live request: {live}. "
        f"A rubric picked {first['router']} because: {'; '.join(first['reasons'])}. Task: {text!r}. "
        'Answer with only JSON: {"router": "<slug>", "reason": "<under 12 words>"}'
    )
    try:
        out = subprocess.run(
            [CLAUDE, "-p", prompt, "--model", "sonnet"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        m = re.search(r"\{.*\}", out.stdout, re.S)
        j = json.loads(m[0]) if m else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    if not j or j.get("router") not in ROUTERS:
        return None
    return j


def choose(
    text: str, live: bool = False, use_claude: bool = False, timeout: float = 10.0
) -> dict:
    """{router, reasons, source: rubric | claude}. Claude may overrule the rubric; its reason is added."""
    first = rubric(text, live)
    if not use_claude or not Path(CLAUDE).exists():
        return first
    j = _claude(text, first, timeout, live)
    if not j:
        return first | {"claude": "unavailable or over 10 s; rubric kept"}
    if j["router"] == first["router"]:
        return first | {
            "source": "rubric + claude",
            "reasons": first["reasons"] + [f"Claude agrees: {j.get('reason', '')}"],
        }
    return {
        "router": j["router"],
        "reasons": [f"Claude: {j.get('reason', '')}", f"rubric said {first['router']}"],
        "source": "claude",
    }
