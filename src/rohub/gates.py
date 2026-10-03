"""Strict robotics-data gates. docs/DATA-SPEC.md is the human-readable copy of SPEC (a test keeps them in sync).

Every gate returns a row {id, name, passed, value, limit, why}. A clip is accepted only if every gate passes.
Rejections carry the first failing gate's reason, which the site and the demo show.
"""

from __future__ import annotations

import os

import numpy as np

from .pick_expert import BOWL_R
from .retarget import Replay
from .scene import TARGET_XY
from .supabase_client import get_store
from .track import MIN_HAND_SCORE, Demo, Events, Track, duplicate_frames

SPEC = {
    "fps_min": 23.0,
    "fps_max": 31.0,
    "clip_s_min": 3.0,
    "clip_s_max": 12.0,
    "hand_score_min": MIN_HAND_SCORE,  # MediaPipe hand presence (handedness score)
    "visible_frac_min": 0.95,  # of frames in the demonstration window
    "window_tail_s": 0.5,  # window = clip start .. release + 0.5 s
    "red_second_max": 0.30,  # second-largest red blob / largest, first frames
    "dup_small_min": 0.05,  # smaller of the two largest red blobs, fraction of the block's rest area
    "dup_big_min": 0.3,  # larger of the two
    "dup_dist_blocks": 2.0,  # blobs further apart than this many block widths
    "dup_frames_max": 3,  # tolerates a few frames of wood knots or skin read as red
    "couple_max_blocks": 1.0,  # hand-to-block drift while carried, in block widths
    "lift_min_m": 0.03,
    "carry_min_m": 0.05,
    "ik_err_max_m": 0.005,
    "vel_max": 3.0,  # rad/s, arm joints, 30 fps commands (STS3215 no-load ~4.7 rad/s at 7.4 V)
    "acc_max": 60.0,  # rad/s^2
    "jerk_max": 2500.0,  # rad/s^3
    "episode_s_min": 2.0,
    "episode_s_max": 30.0,
}
FPS_OUT = 30


def _row(gid, name, passed, value, limit, why_fail):
    return {
        "id": gid,
        "name": name,
        "passed": bool(passed) if passed is not None else None,
        "value": value,
        "limit": limit,
        "why": "" if passed or passed is None else why_fail,
    }


def clip_gates(tr: Track, ev: Events, audit: dict, demo: Demo | None) -> list[dict]:
    S = SPEC
    rows = []
    dur = tr.n / tr.fps
    rows.append(
        _row(
            "fps",
            "frame rate",
            S["fps_min"] <= tr.fps <= S["fps_max"],
            round(tr.fps, 2),
            f"{S['fps_min']:g}-{S['fps_max']:g}",
            f"{tr.fps:.1f} fps is outside the dataset bounds",
        )
    )
    rows.append(
        _row(
            "length",
            "clip length",
            S["clip_s_min"] <= dur <= S["clip_s_max"],
            round(dur, 2),
            f"{S['clip_s_min']:g}-{S['clip_s_max']:g} s",
            f"{dur:.1f} s is outside the length bounds",
        )
    )
    if audit.get("available"):
        ok = (
            audit["red_blocks_first"] == 1
            and audit["bowls_first"] == 1
            and audit["hands_first"] == 1
        )
        rows.append(
            _row(
                "scene",
                "scene audit (Claude)",
                ok,
                f"{audit['red_blocks_first']} red block, {audit['bowls_first']} bowl, {audit['hands_first']} hand",
                "exactly 1 / 1 / 1",
                f"Claude counts {audit['red_blocks_first']} red blocks, {audit['bowls_first']} bowls, "
                f"{audit['hands_first']} hands in the first frame",
            )
        )
    else:
        rows.append(
            _row(
                "scene",
                "scene audit (Claude)",
                None,
                "not run",
                "exactly 1 / 1 / 1",
                "",
            )
        )
    red2 = float(np.median(tr.red_ratio[:6]))
    rows.append(
        _row(
            "one_object",
            "one red object (pixels)",
            red2 <= S["red_second_max"],
            round(red2, 2),
            f"<= {S['red_second_max']}",
            f"a second red object {red2:.0%} the size of the block; which one is the target?",
        )
    )
    if ev.block_px > 0:
        dup = duplicate_frames(tr, ev.block_px, S["dup_small_min"], S["dup_big_min"], S["dup_dist_blocks"])
        rows.append(
            _row(
                "no_duplicate",
                "no second red object appears",
                len(dup) <= S["dup_frames_max"],
                int(len(dup)),
                f"<= {S['dup_frames_max']} frames",
                f"a second red block appears on {len(dup)} frames (first at {dup[0] / tr.fps:.1f} s): the video duplicated the object"
                if len(dup)
                else "",
            )
        )
    t_end = (ev.t_release if ev.ok else tr.n - 1) + int(
        round(S["window_tail_s"] * tr.fps)
    )
    w = slice(0, min(tr.n, t_end + 1))
    vis = float(np.mean(tr.hand_score[w] >= S["hand_score_min"]))
    rows.append(
        _row(
            "hand_visible",
            "hand + wrist visible",
            vis >= S["visible_frac_min"],
            round(vis, 3),
            f">= {S['visible_frac_min']} of frames at score >= {S['hand_score_min']}",
            f"hand seen on {vis:.0%} of the demonstration frames",
        )
    )
    blk = float(np.mean(~np.isnan(tr.block[w, 0])))
    rows.append(
        _row(
            "block_visible",
            "block tracked",
            blk >= S["visible_frac_min"],
            round(blk, 3),
            f">= {S['visible_frac_min']} of frames",
            f"block tracked on only {blk:.0%} of frames",
        )
    )
    rows.append(
        _row(
            "events",
            "grasp, lift, release found",
            ev.ok,
            ev.why or "ok",
            "all three, in order",
            ev.why,
        )
    )
    if ev.ok:
        cb = ev.max_couple_px / ev.block_px
        rows.append(
            _row(
                "coupled",
                "block moves with the hand",
                cb <= S["couple_max_blocks"],
                round(cb, 2),
                f"<= {S['couple_max_blocks']} block widths",
                f"the block drifts {cb:.1f} block widths from the fingers while 'carried' (it moves on its own)",
            )
        )
    if demo is not None:
        rows.append(
            _row(
                "lift",
                "lift height",
                demo.lift_height >= S["lift_min_m"],
                round(demo.lift_height, 3),
                f">= {S['lift_min_m']} m",
                f"lifted only {demo.lift_height * 100:.1f} cm",
            )
        )
        rows.append(
            _row(
                "carry",
                "carry distance",
                demo.carry_dist >= S["carry_min_m"],
                round(demo.carry_dist, 3),
                f">= {S['carry_min_m']} m",
                f"the block moved only {demo.carry_dist * 100:.1f} cm",
            )
        )
    if audit.get("available"):
        ok = bool(audit["block_in_bowl_last"]) and audit.get("red_blocks_last", 1) == 1
        rows.append(
            _row(
                "video_in_bowl",
                "video ends with block in bowl (Claude)",
                ok,
                bool(audit["block_in_bowl_last"]),
                "true",
                "in the last frame the block is not in the bowl"
                if not audit["block_in_bowl_last"]
                else f"{audit.get('red_blocks_last')} red blocks in the last frame",
            )
        )
    return rows


def robot_gates(rep: Replay) -> list[dict]:
    S = SPEC
    rows = []
    lim = int(rep.at_limit.sum())
    err = float(rep.ik_err.max())
    rows.append(
        _row(
            "ik",
            "IK inside SO-101 joint limits",
            lim == 0 and err <= S["ik_err_max_m"],
            f"{lim} ticks at a limit, max error {err * 1000:.1f} mm",
            f"0 ticks, <= {S['ik_err_max_m'] * 1000:g} mm",
            f"IK hits a joint limit on {lim} control ticks (max error {err * 1000:.0f} mm): the SO-101 cannot reach that pose",
        )
    )
    A = rep.actions[:, :5]
    dt = 1.0 / FPS_OUT
    v = np.diff(A, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    for gid, name, x, key, unit in (
        ("vel", "joint velocity", v, "vel_max", "rad/s"),
        ("acc", "joint acceleration", a, "acc_max", "rad/s^2"),
        ("jerk", "joint jerk", j, "jerk_max", "rad/s^3"),
    ):
        mx = float(np.abs(x).max())
        rows.append(
            _row(
                gid,
                name,
                mx <= S[key],
                round(mx, 1),
                f"<= {S[key]:g} {unit}",
                f"peak {name} {mx:.1f} {unit} exceeds {S[key]:g}",
            )
        )
    ok = rep.close_frame >= 0 and (
        rep.lift_frame < 0 or rep.close_frame < rep.lift_frame
    )
    rows.append(
        _row(
            "close_before_lift",
            "gripper closes before lift",
            ok and rep.lift_frame >= 0,
            f"close @{rep.close_frame}, lift @{rep.lift_frame}",
            "close < lift",
            "the cube never left the table"
            if rep.lift_frame < 0
            else "the cube moved up before the gripper closed",
        )
    )
    if rep.cube_at_open is not None:
        dxy = float(np.linalg.norm(rep.cube_at_open[:2] - TARGET_XY))
        rows.append(
            _row(
                "open_over_bowl",
                "gripper opens over the bowl",
                dxy < BOWL_R,
                round(dxy, 3),
                f"< {BOWL_R} m from bowl centre",
                f"the gripper opens with the cube {dxy * 100:.1f} cm from the bowl centre",
            )
        )
    else:
        rows.append(
            _row(
                "open_over_bowl",
                "gripper opens over the bowl",
                False,
                "never opened",
                "opens",
                "the gripper never opened",
            )
        )
    end = np.array(rep.extra["cube_end"])
    rows.append(
        _row(
            "sim_success",
            "cube ends in bowl (MuJoCo replay)",
            rep.success and rep.lifted,
            f"cube at {end.round(3).tolist()}",
            "in bowl, after a lift",
            "in the physics replay the cube does not end in the bowl",
        )
    )
    n = len(rep.self_contacts)
    rows.append(
        _row(
            "self_collision",
            "no self-collision",
            n == 0,
            n,
            "0 contacts",
            f"{n} frames of arm self-contact ({rep.self_contacts[0][1]} / {rep.self_contacts[0][2]})"
            if n
            else "",
        )
    )
    secs = len(rep.states) / FPS_OUT
    rows.append(
        _row(
            "episode_len",
            "episode length",
            S["episode_s_min"] <= secs <= S["episode_s_max"],
            round(secs, 2),
            f"{S['episode_s_min']:g}-{S['episode_s_max']:g} s",
            f"{secs:.1f} s episode is outside the bounds",
        )
    )
    return rows


def verdict(rows: list[dict]) -> tuple[bool, str]:
    for r in rows:
        if r["passed"] is False:
            return False, f"{r['name']}: {r['why']}"
    return True, "accepted"


def publish(
    rows: list[dict],
    *,
    task_id: str | None = None,
    clip_id: str = "",
    video_storage_path: str | None = None,
    dataset_path: str | None = None,
    trajectory_vector: list[float] | None = None,
) -> dict:
    """Write this clip's gate JSON to Supabase (or the offline log when no project is configured)."""
    ok, why = verdict(rows)
    tid = task_id or os.environ.get("ROBOHUB_TASK_ID") or "local-unscoped"
    audit = {"clip_id": clip_id, "accepted": ok, "reason": why, "gates": rows}
    return get_store().log_gate_audit(
        tid,
        audit,
        passed=ok,
        video_storage_path=video_storage_path,
        dataset_path=dataset_path,
        trajectory_vector=trajectory_vector,
    )
