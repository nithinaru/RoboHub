"""Sim joint radians <-> the units LeRobot's so_follower reports with use_degrees=True.

so_follower normalizes the five arm joints with MotorNormMode.DEGREES, i.e. (ticks - mid) * 360 / 4095: degrees
relative to the middle of the calibrated range. The "new calib" MJCF puts each joint's zero at the middle of its
range (the ranges are symmetric), so degrees = rad * 180 / pi is the documented approximation. The gripper uses
MotorNormMode.RANGE_0_100 over its calibrated range, which in the MJCF is [-0.1745, 1.7453] rad (closed..open).
"""

from __future__ import annotations

import numpy as np

GRIPPER_LO = -0.1745
GRIPPER_SPAN = 1.9199  # 1.7453 - (-0.1745)


def to_lerobot_units(q_rad: np.ndarray) -> np.ndarray:
    """[6] sim joint positions (rad) -> [5 arm joints in degrees, gripper 0-100], float32."""
    q = np.asarray(q_rad, dtype=np.float64)
    out = np.empty(6, dtype=np.float64)
    out[:5] = np.degrees(q[:5])
    out[5] = 100.0 * (q[5] - GRIPPER_LO) / GRIPPER_SPAN
    return out.astype(np.float32)


def from_lerobot_units(v: np.ndarray) -> np.ndarray:
    """Inverse of to_lerobot_units: [5 degrees, gripper 0-100] -> [6] rad."""
    v = np.asarray(v, dtype=np.float64)
    out = np.empty(6, dtype=np.float64)
    out[:5] = np.radians(v[:5])
    out[5] = v[5] / 100.0 * GRIPPER_SPAN + GRIPPER_LO
    return out
