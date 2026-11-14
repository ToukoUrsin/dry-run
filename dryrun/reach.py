"""Closed-form SO-101 reach map used by seeded station generation.

A pose is accepted only when the analytic IK solution is at least 5% inside
all five joint limits (the 95% interior margin map). This gate runs before MJCF
creation, so generated scenes cannot contain unreachable pick or bowl poses.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from .closed_form_ik import JOINT_LIMITS, IKError, IKResult, ik_pad_pose

MARGIN_FRACTION = 0.05
PICK_Z = 0.034
PRESENT_Z = 0.125
APPROACH_PITCH_DEG = 35
MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "reach_map_polar.json"

def pose_margin(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    spans = JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]
    return float(np.min(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q) / spans))

def best_pose(target, *, min_margin=MARGIN_FRACTION):
    """Scan approach pitch 35°..0° and return the safest analytic pose.

    The scan is deterministic and uses only closed-form ``ik_pad_pose``. The
    first valid result is the steepest approach that solves. The returned
    tuple is ``(accepted, margin, IKResult|None)``.
    """
    fallback = None
    for degrees in range(APPROACH_PITCH_DEG, -1, -1):
        try:
            # The rollout targets the open-jaw pad midpoint, so the measured
            # reach map must solve that same point rather than the nearby
            # gripperframe site.
            result = ik_pad_pose(target, approach_pitch=np.deg2rad(degrees), enforce_limits=True)
        except IKError:
            continue
        margin = pose_margin(result.q)
        if result.error < 1e-5:
            if fallback is None:
                fallback = (margin, result)
            if margin >= min_margin:
                return True, margin, result
    if fallback is None:
        return False, -1.0, None
    margin, result = fallback
    return False, margin, result


def reach_margin(target, *, min_margin=MARGIN_FRACTION):
    """Return ``(accepted, margin, q, error)`` for a scanned target."""
    accepted, margin, result = best_pose(target, min_margin=min_margin)
    if result is None:
        return False, margin, None, float("inf")
    return accepted, margin, result.q, result.error

def scene_reachable(x: float, y: float) -> bool:
    # Pickup pad midpoint and a lifted/presentation pose must both be safe.
    return reach_margin((x, y, PICK_Z))[0] and reach_margin((x, y, PRESENT_Z))[0]


def _map_rows() -> tuple[dict, ...]:
    if not MAP_PATH.exists():
        return ()
    return tuple(json.loads(MAP_PATH.read_text()).get("cells", ()))


MEASURED_MAP = _map_rows()


def measured_pick_rate(x: float, y: float, size: str, yaw: float) -> float:
    """Return the measured pick/lift rate for an exact polar-map cell."""
    rows = [row for row in MEASURED_MAP
            if row.get("size") == size
            and abs(float(row.get("x", 1e9)) - float(x)) <= 1e-6
            and abs(float(row.get("y", 1e9)) - float(y)) <= 1e-6]
    if not rows:
        return 0.0
    tested = np.asarray(rows[0].get("tested_yaws_rad", ()), dtype=float)
    if not len(tested):
        return 0.0
    period = math.pi / 2.0 if size == "2x2" else math.pi
    delta = np.abs((float(yaw) - tested + period / 2.0) % period - period / 2.0)
    if float(np.min(delta)) > 1e-6:
        return 0.0
    return float(rows[0].get("pick_rate", 0.0))


def measured_pick_ok(x: float, y: float, size: str, yaw: float, *, threshold: float = .95) -> bool:
    return measured_pick_rate(x, y, size, yaw) >= float(threshold)


def measured_cells(size: str, *, threshold: float = .95) -> tuple[dict, ...]:
    return tuple(row for row in MEASURED_MAP
                 if row.get("size") == size and float(row.get("pick_rate", 0.0)) >= float(threshold))
