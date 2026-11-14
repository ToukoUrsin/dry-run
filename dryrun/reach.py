"""Closed-form SO-101 reach map used by seeded station generation.

A pose is accepted only when the analytic IK solution is at least 5% inside
all five joint limits (the 95% interior margin map). This gate runs before MJCF
creation, so generated scenes cannot contain unreachable pick or bowl poses.
"""
from __future__ import annotations
import numpy as np
from .closed_form_ik import JOINT_LIMITS, IKError, solve_position

MARGIN_FRACTION = 0.05

def pose_margin(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    spans = JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]
    return float(np.min(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q) / spans))

def reach_margin(target, *, min_margin=MARGIN_FRACTION):
    """Return (accepted, margin, q, error) for a gripper target."""
    try:
        result = solve_position(target, enforce_limits=True)
    except IKError:
        return False, -1.0, None, float("inf")
    margin = pose_margin(result.q)
    return bool(margin >= min_margin and result.error < 1e-5), margin, result.q, result.error

def scene_reachable(x: float, y: float) -> bool:
    # Pickup pad midpoint and a common bowl-drop height must both be safe.
    return reach_margin((x, y, .080))[0] and reach_margin((x, y, .120))[0]
