"""Closed form position IK for the vendored SO-101 arm.

The module is deliberately independent of the station generator and the
benchmark.  It models the position of the ``gripperframe`` site in
``assets/SO101/Simulation/SO101/so101_dryrun.xml``.  The solution fixes the
wrist-flex and wrist-roll joints (``q4`` and ``q5``) to supplied values, then
solves the remaining shoulder-pan, shoulder-lift and elbow joints as a
two-link planar problem.  The default ``q4=q5=0`` is the calibration pose used
by the measured station reach checks.

This is position-only IK.  It does not model the gripper opening joint.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


# Joint limits copied from the SO-101 MJCF (radians), in q0..q4 order.
JOINT_LIMITS = np.asarray(
    [
        (-1.9198621771937616, 1.9198621771937634),  # shoulder_pan
        (-1.7453292519943224, 1.7453292519943366),  # shoulder_lift
        (-1.69, 1.69),  # elbow_flex
        (-1.6580628494556928, 1.6580627293335335),  # wrist_flex
        (-2.7438472969992493, 2.841206309382605),  # wrist_roll
    ],
    dtype=float,
)

# Geometry is expressed in the q0=0 radial/tangential/vertical frame.  The
# radial frame rotates by -q0 in world XY because of the shoulder-pan mount.
# The tiny y offset is present in the MJCF shoulder body pose and is retained
# so analytic-vs-MuJoCo checks reach floating-point precision.
SHOULDER = np.asarray((0.0388353, -8.97657e-09, 0.0624), dtype=float)
A_FROM_SHOULDER = np.asarray((0.0303992, -0.0182778, 0.0542), dtype=float)
L1 = np.asarray((0.028, 0.0, 0.11257), dtype=float)
L2 = np.asarray((0.1349, 0.0, 0.0052), dtype=float)
L3 = np.asarray((0.0611, 0.0181, 0.0), dtype=float)
SITE_FROM_GRIPPER = np.asarray((0.0981274, 0.00016655369, -0.007901255384), dtype=float)


class IKError(ValueError):
    """Raised when a target has no solution under the requested branch."""


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position: np.ndarray
    error: float


def _ry(angle: float, v: np.ndarray) -> np.ndarray:
    """MuJoCo's equivalent y-axis rotation for this arm's local frame."""
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray((c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]))


def _rx_minus(angle: float, v: np.ndarray) -> np.ndarray:
    """Rotate by ``-angle`` around the local x axis (the wrist-roll axis)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray((v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2]))


def forward_position(q: Iterable[float]) -> np.ndarray:
    """Return the analytic world position of ``gripperframe`` for q0..q4."""
    q = np.asarray(tuple(q), dtype=float)
    if q.shape != (5,):
        raise ValueError("q must contain shoulder_pan through wrist_roll")
    q0, q1, q2, q3, q4 = q
    v4 = _rx_minus(q4, SITE_FROM_GRIPPER)
    theta = q1 + q2 + q3
    local = A_FROM_SHOULDER + _ry(q1, L1) + _ry(q1 + q2, L2)
    local = local + _ry(theta, L3) + _ry(theta, v4)
    # A_FROM_SHOULDER and the link vectors are in radial/tangential/vertical
    # coordinates.  Shoulder-pan rotates that frame clockwise in world XY.
    c, s = math.cos(q0), math.sin(q0)
    x, y = local[0] * c + local[1] * s, -local[0] * s + local[1] * c
    return SHOULDER + np.asarray((x, y, local[2]))


def _normalise_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def ik_position(
    target: Iterable[float],
    *,
    q4: float = 0.0,
    q5: float = 0.0,
    elbow: int = 1,
    radial_sign: int = 1,
    enforce_limits: bool = True,
) -> IKResult:
    """Solve gripperframe position with a closed-form two-link reduction.

    ``q4`` and ``q5`` are held fixed. ``elbow`` selects the two planar
    branches (``+1`` or ``-1``), and ``radial_sign`` selects the radial branch
    around the shoulder (normally ``+1``).  The returned error is the norm of
    analytic forward position minus ``target``.
    """
    target = np.asarray(tuple(target), dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target must be three finite coordinates")
    if elbow not in (-1, 1) or radial_sign not in (-1, 1):
        raise ValueError("elbow and radial_sign must be +1 or -1")
    if not np.isfinite(q4) or not np.isfinite(q5):
        raise ValueError("q4 and q5 must be finite")

    # q0 only rotates the local radial/tangent pair.  The tangent coordinate is
    # fixed by the wrist-roll angle, so the target determines q0 and radial r.
    h = target[:2] - SHOULDER[:2]
    radius_sq = float(np.dot(h, h))
    v4 = _rx_minus(q5, SITE_FROM_GRIPPER)
    tangent = float(A_FROM_SHOULDER[1] + L3[1] + v4[1])
    radial_sq = radius_sq - tangent * tangent
    if radial_sq < -1e-10:
        raise IKError("target horizontal radius is smaller than fixed tangent offset")
    radial = radial_sign * math.sqrt(max(0.0, radial_sq))
    h_angle = math.atan2(float(h[1]), float(h[0]))
    q0 = _normalise_angle(math.atan2(tangent, radial) - h_angle)

    # With q4 fixed, L2 and the wrist composite vector form a standard two-link
    # chain after q1+q2.  Work in x/z polar angles, where Ry(q) subtracts q.
    # Keep the expression explicit: _ry expects a 3-vector, while the wrist
    # vector's tangential component does not affect this planar solve.
    v34 = np.asarray((L3[0] + v4[0], 0.0, L3[2] + v4[2]))
    w = L2 + _ry(q4, v34)
    l1, l2 = float(np.hypot(L1[0], L1[2])), float(np.hypot(w[0], w[2]))
    p = np.asarray((radial - A_FROM_SHOULDER[0], target[2] - SHOULDER[2] - A_FROM_SHOULDER[2]))
    dist = float(np.hypot(p[0], p[1]))
    cos_eta = (dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if cos_eta < -1.0 - 1e-10 or cos_eta > 1.0 + 1e-10:
        raise IKError("target is outside the two-link position workspace")
    eta = elbow * math.acos(float(np.clip(cos_eta, -1.0, 1.0)))
    psi = math.atan2(float(p[1]), float(p[0]))
    alpha = psi - math.atan2(l2 * math.sin(eta), l1 + l2 * math.cos(eta))
    phi1 = math.atan2(float(L1[2]), float(L1[0]))
    phiw = math.atan2(float(w[2]), float(w[0]))
    q1 = _normalise_angle(phi1 - alpha)
    q12 = _normalise_angle(phiw - (alpha + eta))
    q2 = _normalise_angle(q12 - q1)
    q = np.asarray((q0, q1, q2, q4, q5), dtype=float)
    if enforce_limits:
        for i, value in enumerate(q):
            lo, hi = JOINT_LIMITS[i]
            if value < lo - 1e-8 or value > hi + 1e-8:
                raise IKError(f"joint q{i}={value:.6f} outside [{lo:.6f}, {hi:.6f}]")
    position = forward_position(q)
    return IKResult(q=q, position=position, error=float(np.linalg.norm(position - target)))


def solve_position(
    target: Iterable[float],
    *,
    q4: float = 0.0,
    q5: float = 0.0,
    enforce_limits: bool = True,
) -> IKResult:
    """Try all radial/elbow branches and return the first valid solution.

    Branch enumeration is finite (four closed-form candidates); no numerical
    iteration is used.  Among valid candidates the one nearest the neutral
    arm pose is returned, which keeps the result deterministic at symmetric
    targets.
    """
    candidates: list[IKResult] = []
    for radial_sign in (1, -1):
        for elbow in (1, -1):
            try:
                candidates.append(
                    ik_position(
                        target,
                        q4=q4,
                        q5=q5,
                        elbow=elbow,
                        radial_sign=radial_sign,
                        enforce_limits=enforce_limits,
                    )
                )
            except IKError:
                continue
    if not candidates:
        raise IKError("target has no closed-form solution within joint limits")
    return min(candidates, key=lambda r: float(np.linalg.norm(r.q)))


__all__ = ["IKError", "IKResult", "JOINT_LIMITS", "forward_position", "ik_position", "solve_position"]
