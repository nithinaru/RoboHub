"""Stage 3: from per-frame landmarks to a human demonstration in the robot's frame.

Input: the pose JSON written by src/pose/extract.py (pixels). Output: a Demo with, per video frame,
the grasp-point position in metres in the SO-101 table frame and the gripper command.

How the 1:1 mapping works (details and assumptions in README.md):
  * Ruler: the block is a 3 cm cube, so metres per pixel = 0.03 / sqrt(block area at rest).
  * Events come from the object, not from finger aperture: the grasp is the last frame before the block starts
    to move with the hand, the release is the first frame the block is back at rest. Finger aperture from a
    single view is too noisy on generated hands to switch a gripper (measured in docs/POSE-COMPARE.md).
  * Grasp-anchored: the pinch-point trajectory is shifted so that at the grasp frame it sits on the block centre.
  * Axes: image right -> robot +y (as seen from the sim's front camera), image up -> robot +z. Depth is not
    observable from one view; the demonstration is assumed to stay in the plane through the block and the bowl.
  * The bowl is fixed in the sim at TARGET_XY; the cube starts at the bowl plus the block-to-bowl offset measured
    in the clip, so the carry distance is the human's, 1:1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scene import CUBE_HALF, TARGET_XY

BLOCK_M = 0.03
MIN_HAND_SCORE = 0.5
VISIBLE_AREA = 0.5  # of the block's rest area


@dataclass
class Track:
    fps: float
    n: int
    width: int
    height: int
    pinch: np.ndarray  # [n, 2] px, nan where the hand is missing
    hand_score: np.ndarray  # [n], 0 where missing
    aperture: np.ndarray  # [n], nan where missing
    block: np.ndarray  # [n, 2] px, nan where missing
    block_area: np.ndarray  # [n]
    red_ratio: np.ndarray  # [n] second-largest red blob area / largest (0 if one blob)
    wrist_vis: np.ndarray  # [n] pose-model wrist visibility, 0 if missing
    second_area: np.ndarray = None  # [n] runner-up red blob area, 0 if none
    second_c: np.ndarray = None  # [n, 2] runner-up red blob centroid, nan if none
    hand_pts: list = field(repr=False, default_factory=list)


def load_track(path: Path) -> Track:
    d = json.loads(Path(path).read_text())
    F = d["frames"]
    n = len(F)
    nan2 = np.full((n, 2), np.nan)
    pinch, block = nan2.copy(), nan2.copy()
    score, ap, area, ratio, wv = (np.zeros(n) for _ in range(5))
    ap[:] = np.nan
    pts = []
    sa, sc2 = np.zeros(n), nan2.copy()
    for i, f in enumerate(F):
        h, b = f.get("hand"), f.get("block")
        if h:
            pinch[i] = h["pinch"]
            score[i] = h["score"]
            ap[i] = h["aperture"]
            pts.append(h["pts"])
        else:
            pts.append(None)
        if b:
            block[i] = b["c"]
            area[i] = b["area"]
            ratio[i] = b.get("second_ratio", 0.0)
            if b.get("second_c"):
                sa[i] = b.get("second_area", 0)
                sc2[i] = b["second_c"]
        arm = f.get("arm")
        if arm:
            # the demonstrating side is the one whose wrist is closest to the hand
            cands = [arm.get("wrist"), arm.get("l_wrist")]
            cands = [c for c in cands if c]
            if h and cands:
                c = min(
                    cands,
                    key=lambda c: np.hypot(
                        c[0] - h["pts"][0][0], c[1] - h["pts"][0][1]
                    ),
                )
            else:
                c = max(cands, key=lambda c: c[2]) if cands else [0, 0, 0]
            wv[i] = c[2]
    return Track(
        d["fps"],
        n,
        d["width"],
        d["height"],
        pinch,
        score,
        ap,
        block,
        area,
        ratio,
        wv,
        second_area=sa,
        second_c=sc2,
        hand_pts=pts,
    )


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over interior nan runs of at most max_gap frames (per column)."""
    x = x.copy()
    n = len(x)
    idx = np.arange(n)
    cols = x if x.ndim == 2 else x[:, None]
    for c in range(cols.shape[1]):
        col = cols[:, c]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        i = 0
        while i < n:
            if ok[i]:
                i += 1
                continue
            j = i
            while j < n and not ok[j]:
                j += 1
            if i > 0 and j < n and j - i <= max_gap:
                col[i:j] = np.interp(idx[i:j], [i - 1, j], [col[i - 1], col[j]])
            i = j
    return x


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Centred Gaussian smoothing along axis 0, nan-aware (nan frames stay nan)."""
    if sigma <= 0:
        return x.copy()
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    x2 = x if x.ndim == 2 else x[:, None]
    out = np.full_like(x2, np.nan, dtype=float)
    ok = ~np.isnan(x2[:, 0])
    for i in range(len(x2)):
        if not ok[i]:
            continue
        lo, hi = max(0, i - r), min(len(x2), i + r + 1)
        w = k[lo - i + r : hi - i + r] * ok[lo:hi]
        out[i] = (w[:, None] * np.nan_to_num(x2[lo:hi])).sum(0) / w.sum()
    return out if x.ndim == 2 else out[:, 0]


@dataclass
class Events:
    ok: bool
    why: str = ""
    block_px: float = 0.0
    b0: np.ndarray | None = None  # block centre at rest before the grasp, px
    b1: np.ndarray | None = None  # block centre at rest after the release, px
    t_grasp: int = -1
    t_lift: int = -1
    t_release: int = -1
    hand_first: int = -1
    hand_last: int = -1
    max_couple_px: float = 0.0  # worst pinch-to-block distance while carried, px


def find_events(tr: Track) -> Events:
    blk = smooth(fill_gaps(tr.block, 4), 1.0)
    have = ~np.isnan(blk[:, 0])
    hand = tr.hand_score >= MIN_HAND_SCORE
    if hand.sum() < 5:
        return Events(False, "no hand found")
    hand_first, hand_last = (
        int(np.argmax(hand)),
        int(len(hand) - 1 - np.argmax(hand[::-1])),
    )
    if have[:6].sum() < 4 or have[-6:].sum() < 4:
        return Events(False, "block not tracked at the start and end")
    b0 = np.nanmedian(blk[:6], axis=0)
    b1 = np.nanmedian(blk[-6:], axis=0)
    rest_area = float(np.median(tr.block_area[:6][tr.block_area[:6] > 0]))
    block_px = float(np.sqrt(rest_area))
    moved = np.linalg.norm(blk - b0, axis=1) > 0.25 * block_px
    moved[~have] = False
    # lift: first frame of a run of 3 moved frames
    t_lift = -1
    for i in range(len(moved) - 2):
        if moved[i] and moved[i + 1] and moved[i + 2]:
            t_lift = i
            break
    if t_lift < 0:
        return Events(
            False,
            "the block never moves",
            block_px,
            b0,
            b1,
            hand_first=hand_first,
            hand_last=hand_last,
        )
    pinch = smooth(fill_gaps(tr.pinch, 3), 1.0)
    # grasp: the first frame, in the 1 s before the lift, where the fingers are as close to the block as they get
    # (within 5% of a block width of the closest approach); strictly before the lift
    lo = max(0, t_lift - int(round(1.0 * tr.fps)))
    d = np.linalg.norm(pinch[lo:t_lift] - blk[lo:t_lift], axis=1)
    d[tr.hand_score[lo:t_lift] < MIN_HAND_SCORE] = (
        np.nan
    )  # a grasp needs a measured hand
    if len(d) == 0 or np.all(np.isnan(d)):
        return Events(
            False,
            "no hand at the moment of the grasp",
            block_px,
            b0,
            b1,
            t_lift=t_lift,
        )
    near = np.where(d <= np.nanmin(d) + 0.05 * block_px)[0]
    t_grasp = lo + int(near[0])
    # release: first frame after lift from which the block stays within 0.25 block of its final rest
    at_rest = np.linalg.norm(blk - b1, axis=1) < 0.25 * block_px
    t_release = -1
    for i in range(t_lift + 1, len(at_rest)):
        if np.all(at_rest[i:]) or (
            i + 6 <= len(at_rest) and np.all(at_rest[i : i + 6])
        ):
            t_release = i
            break
    if t_release < 0:
        return Events(
            False,
            "the block never comes to rest after the lift",
            block_px,
            b0,
            b1,
            t_grasp,
            t_lift,
        )
    carry = slice(t_grasp, t_release + 1)
    dc = np.linalg.norm(
        (pinch[carry] - pinch[t_grasp]) - (blk[carry] - blk[t_grasp]), axis=1
    )
    # a mostly hidden block (behind the fingers or the bowl rim) has a centroid that is not the block's centre:
    # only frames where at least half of the block is visible count as measurements
    visible = tr.block_area[carry] >= VISIBLE_AREA * rest_area
    if visible.mean() < 0.5:
        dc[:] = np.inf
    else:
        dc[~visible] = np.nan
    return Events(
        True,
        "",
        block_px,
        b0,
        b1,
        t_grasp,
        t_lift,
        t_release,
        hand_first,
        hand_last,
        float(np.nanmax(dc)) if np.any(~np.isnan(dc)) else float("inf"),
    )


@dataclass
class Demo:
    """A human demonstration in the robot frame, one row per video frame over [start, end]."""

    fps: float
    start: int
    end: int
    xyz: (
        np.ndarray
    )  # [T, 3] grasp point (the block centre when held), metres, robot frame
    closed: np.ndarray  # [T] bool, gripper command
    cube_start: np.ndarray  # [3]
    m_per_px: float
    grasp_i: int  # index into xyz of the grasp frame
    release_i: int
    lift_height: float
    carry_dist: float


def to_robot(
    tr: Track, ev: Events, bowl_u: float, sigma: float = 1.5, tail_s: float = 0.5
) -> Demo:
    s = BLOCK_M / ev.block_px
    pinch = smooth(fill_gaps(tr.pinch, 3), sigma)
    off = (
        pinch[ev.t_grasp] - ev.b0 if not np.isnan(pinch[ev.t_grasp, 0]) else np.zeros(2)
    )
    g = pinch - off
    # While the block is held, the grasp point IS the block: use the block's own measured centre (the fingertips
    # slide and get occluded while holding; the block does not). Frames where the block is mostly hidden keep the
    # last visible block-to-pinch offset. After the release the hand path continues from where the block was let go.
    raw_blk = fill_gaps(tr.block, 4)
    seen = tr.block_area >= VISIBLE_AREA * ev.block_px**2
    cur = (
        pinch[ev.t_grasp] - raw_blk[ev.t_grasp]
        if not np.isnan(pinch[ev.t_grasp, 0])
        else np.zeros(2)
    )
    for i in range(ev.t_grasp, ev.t_release):
        if seen[i] and not np.isnan(raw_blk[i, 0]) and not np.isnan(pinch[i, 0]):
            cur = pinch[i] - raw_blk[i]
            g[i] = raw_blk[i]
        elif not np.isnan(pinch[i, 0]):
            g[i] = pinch[i] - cur
    held = slice(ev.t_grasp, ev.t_release)
    g[held] = smooth(g[held], sigma)
    if not np.isnan(pinch[ev.t_release, 0]):
        g[ev.t_release :] = pinch[ev.t_release :] - (
            pinch[ev.t_release] - g[ev.t_release - 1]
        )
    start = ev.hand_first
    end = min(ev.hand_last, ev.t_release + int(round(tail_s * tr.fps)))
    seg = fill_gaps(
        g[start : end + 1], 1000
    )  # interior gaps are gated separately (visibility gate)
    # ends: hold the nearest valid value
    for c in range(2):
        col = seg[:, c]
        ok = ~np.isnan(col)
        if ok.any():
            col[: np.argmax(ok)] = col[np.argmax(ok)]
            last = len(ok) - 1 - np.argmax(ok[::-1])
            col[last + 1 :] = col[last]
    y0 = (ev.b0[0] - bowl_u) * s  # cube start y relative to the bowl
    xyz = np.zeros((len(seg), 3))
    xyz[:, 0] = TARGET_XY[0]
    xyz[:, 1] = TARGET_XY[1] + (seg[:, 0] - bowl_u) * s
    xyz[:, 2] = CUBE_HALF + (ev.b0[1] - seg[:, 1]) * s
    closed = np.zeros(len(seg), bool)
    gi, ri = ev.t_grasp - start, ev.t_release - start
    closed[gi:ri] = True
    # The still hold between the grasp and the lift (the block does not move, so nothing is demonstrated) is
    # dropped: in the sim a closed gripper squeezing a cube that rests on the table slowly walks up the cube
    # (measured on v01 and v06: 8 to 12 mm over 1 s), and then the lift grips only the bottom edge.
    li = ev.t_lift - start
    if li - 1 > gi + 1:
        keep = np.r_[0 : gi + 1, li - 1 : len(xyz)]
        cut = len(xyz) - len(keep)
        xyz, closed = xyz[keep], closed[keep]
        ri -= cut
    return Demo(
        tr.fps,
        start,
        end,
        xyz,
        closed,
        np.array([TARGET_XY[0], TARGET_XY[1] + y0, CUBE_HALF]),
        s,
        gi,
        ri,
        float(xyz[gi:ri, 2].max() - CUBE_HALF) if ri > gi else 0.0,
        float(abs(ev.b1[0] - ev.b0[0]) * s),
    )


def duplicate_frames(
    tr: Track, block_px: float, small_min: float, big_min: float, dist_blocks: float
) -> np.ndarray:
    """Frames where two red objects are visible at once: the two largest red blobs are more than dist_blocks
    block widths apart, the larger is at least big_min of the block's rest area (a whole block, not a wood knot)
    and the smaller at least small_min (a block partly hidden in the bowl still counts). A block split in two by
    fingers or a bowl rim stays within a block width, so it does not count; a generated duplicate block does."""
    if tr.second_area is None:
        return np.zeros(0, int)
    rest = block_px**2
    big = np.maximum(tr.block_area, tr.second_area) >= big_min * rest
    small = tr.second_area >= small_min * rest
    d = np.linalg.norm(tr.second_c - tr.block, axis=1)
    far = np.nan_to_num(d, nan=0.0) > dist_blocks * block_px
    return np.where(big & small & far)[0]
