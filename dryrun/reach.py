"""Closed-form SO-101 reach map used by seeded station generation.

A pose is accepted only when the analytic IK solution is at least 5% inside
all five joint limits (the 95% interior margin map). This gate runs before MJCF
creation, so generated scenes cannot contain unreachable pick or bowl poses.
"""
from __future__ import annotations
import numpy as np
from .closed_form_ik import JOINT_LIMITS, IKError, IKResult, ik_pose

MARGIN_FRACTION = 0.05

def pose_margin(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    spans = JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]
    return float(np.min(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q) / spans))

def best_pose(target, *, min_margin=MARGIN_FRACTION):
    """Scan approach pitch 35°..0° and return the safest analytic pose.

    The scan is deterministic and uses only closed-form ``ik_pose``.  A pose
    is preferred by joint-limit margin, then by distance from neutral.  The
    returned tuple is ``(accepted, margin, IKResult|None)``.
    """
    candidates = []
    for degrees in range(35, -1, -1):
        try:
            result = ik_pose(target, approach_pitch=np.deg2rad(degrees), enforce_limits=True)
        except IKError:
            continue
        margin = pose_margin(result.q)
        if result.error < 1e-5:
            candidates.append((margin, float(np.linalg.norm(result.q)), result))
    if not candidates:
        return False, -1.0, None
    margin, _, result = max(candidates, key=lambda item: (item[0], -item[1]))
    return bool(margin >= min_margin), margin, result


def reach_margin(target, *, min_margin=MARGIN_FRACTION):
    """Return ``(accepted, margin, q, error)`` for a scanned target."""
    accepted, margin, result = best_pose(target, min_margin=min_margin)
    if result is None:
        return False, margin, None, float("inf")
    return accepted, margin, result.q, result.error

def scene_reachable(x: float, y: float) -> bool:
    # Pickup pad midpoint and a lifted/presentation pose must both be safe.
    return reach_margin((x, y, .034))[0] and reach_margin((x, y, .125))[0]
