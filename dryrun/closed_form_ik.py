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
# The site and pad midpoint are calibrated against the fingertip collision pads
# in ``so101_dryrun.xml``.  The previous pads sat near the jaw hinge and opened
# only 49 mm centre-to-centre; these values describe the relocated tips.
SITE_FROM_GRIPPER = np.asarray((0.09350864, -0.00786148, 0.02778791), dtype=float)
# The pad midpoint is a fixed-pad offset plus a single hinge sinusoid from the
# moving jaw.  These vectors are calibrated in the gripper body frame from the
# MJCF, so the exact solver can drive both the open and closed jaw targets.
PAD_MID_C = np.asarray((0.0605, -0.00719301, 0.01406665), dtype=float)
PAD_MID_COS = np.asarray((0.0350, 0.00035522, -0.00729135), dtype=float)
PAD_MID_SIN = np.asarray((0.0073, -0.00170311021, 0.0349585389), dtype=float)


def pad_from_gripper(grip_q: float) -> np.ndarray:
    """Return the pad midpoint offset from the gripper body at q=``grip_q``."""

    g = float(grip_q)
    if not np.isfinite(g):
        raise ValueError("grip_q must be finite")
    return PAD_MID_C + PAD_MID_COS * math.cos(g) + PAD_MID_SIN * math.sin(g)


class IKError(ValueError):
    """Raised when a target has no solution under the requested branch."""


@dataclass(frozen=True)
class IKResult:
    """One finite, analytic IK solution.

    ``q`` is in the actuator order used by the SO-101 XML (pan, lift, elbow,
    wrist-flex, wrist-roll).  ``approach_pitch`` is the sum of the three
    planar pitch joints.  It is zero for the calibrated straight-down pose.
    """

    q: np.ndarray
    position: np.ndarray
    error: float
    approach_pitch: float = 0.0
    jaw_yaw: float | None = None


def _ry(angle: float, v: np.ndarray) -> np.ndarray:
    """MuJoCo's equivalent y-axis rotation for this arm's local frame."""
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray((c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]))


def _rx_minus(angle: float, v: np.ndarray) -> np.ndarray:
    """Rotate by ``-angle`` around the local x axis (the wrist-roll axis)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray((v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2]))


def _forward_with_distal(q: Iterable[float], distal: np.ndarray) -> np.ndarray:
    """Forward position with a calibrated vector fixed in the gripper body."""
    q = np.asarray(tuple(q), dtype=float)
    if q.shape != (5,):
        raise ValueError("q must contain shoulder_pan through wrist_roll")
    q0, q1, q2, q3, q4 = q
    v4 = _rx_minus(q4, distal)
    theta = q1 + q2 + q3
    local = A_FROM_SHOULDER + _ry(q1, L1) + _ry(q1 + q2, L2)
    local = local + _ry(theta, L3) + _ry(theta, v4)
    # A_FROM_SHOULDER and the link vectors are in radial/tangential/vertical
    # coordinates.  Shoulder-pan rotates that frame clockwise in world XY.
    c, s = math.cos(q0), math.sin(q0)
    x, y = local[0] * c + local[1] * s, -local[0] * s + local[1] * c
    return SHOULDER + np.asarray((x, y, local[2]))


def forward_position(q: Iterable[float]) -> np.ndarray:
    """Return the analytic world position of ``gripperframe`` for q0..q4."""

    return _forward_with_distal(q, SITE_FROM_GRIPPER)


def forward_pad_midpoint(q: Iterable[float], *, grip_q: float = 1.7) -> np.ndarray:
    """Return the analytic pad midpoint at any gripper opening."""
    q = np.asarray(tuple(q), dtype=float)
    if q.shape != (5,):
        raise ValueError("q must contain shoulder_pan through wrist_roll")
    q0, q1, q2, q3, q4 = q
    distal = _rx_minus(q4, pad_from_gripper(grip_q))
    theta = q1 + q2 + q3
    local = A_FROM_SHOULDER + _ry(q1, L1) + _ry(q1 + q2, L2) + _ry(theta, L3 + distal)
    c, s = math.cos(q0), math.sin(q0)
    return SHOULDER + np.asarray((local[0] * c + local[1] * s,
                                  -local[0] * s + local[1] * c,
                                  local[2]))


def jaw_vector(q: Iterable[float]) -> np.ndarray:
    """Return the fixed-pad to moving-pad vector analytically.

    The vector is useful for selecting an equivalent wrist-roll branch for a
    symmetric brick.  It is derived from the two pad centres in the XML and
    has the same radial-frame convention as :func:`forward_position`.
    """

    q = np.asarray(tuple(q), dtype=float)
    if q.shape != (5,):
        raise ValueError("q must contain shoulder_pan through wrist_roll")
    return jaw_vector_at_grip(q, grip_q=0.6)


# Fixed-to-moving centre vector at the closing opening (gripper q=0.35).
# This is the quantity whose orientation matters for choosing a brick yaw;
# it is separate from PAD_MID_* which describes the midpoint target.
JAW_VECTOR_CLOSED = np.asarray((-0.00818273, 0.01348035, 0.03977816), dtype=float)


def jaw_vector_at_grip(q: Iterable[float], *, grip_q: float = 0.6) -> np.ndarray:
    """Return the fixed-to-moving fingertip vector at a gripper opening."""

    q = np.asarray(tuple(q), dtype=float)
    if q.shape != (5,):
        raise ValueError("q must contain shoulder_pan through wrist_roll")
    q0, q1, q2, q3, q4 = q
    # At q0..q4=0 the calibrated fingertip vector is JAW_VECTOR_CLOSED.  The
    # arm pitch and wrist roll rotate that vector into the world radial frame.
    # The small change over the final squeeze is irrelevant to yaw selection,
    # while keeping this closed-form branch exact and deterministic.
    del grip_q
    base = JAW_VECTOR_CLOSED
    local = _ry(q1 + q2 + q3, _rx_minus(q4, base))
    c, s = math.cos(q0), math.sin(q0)
    return np.asarray((local[0] * c + local[1] * s,
                       -local[0] * s + local[1] * c,
                       local[2]), dtype=float)


def _jaw_yaw(q: np.ndarray) -> float:
    v = jaw_vector(q)
    return math.atan2(float(v[1]), float(v[0]))


def _planar_solution(radial: float, z: float, approach_pitch: float,
                     wrist_roll: float, elbow: int,
                     distal_base: np.ndarray = SITE_FROM_GRIPPER) -> tuple[float, float, float]:
    """Solve q1, q2 and q3 after pan and wrist roll are separated out."""

    v4 = _rx_minus(wrist_roll, distal_base)
    wrist = L3 + v4
    # The wrist-flex axis is the point obtained by backing off the target by
    # the calibrated wrist offset at the requested approach pitch.
    off = _ry(approach_pitch, wrist)
    p = np.asarray((radial - A_FROM_SHOULDER[0] - off[0],
                    z - SHOULDER[2] - A_FROM_SHOULDER[2] - off[2]))
    l1 = float(np.hypot(L1[0], L1[2]))
    l2 = float(np.hypot(L2[0], L2[2]))
    dist = float(np.hypot(p[0], p[1]))
    cos_eta = (dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if cos_eta < -1.0 - 1e-10 or cos_eta > 1.0 + 1e-10:
        raise IKError("target is outside the two-link position workspace")
    eta = int(elbow) * math.acos(float(np.clip(cos_eta, -1.0, 1.0)))
    psi = math.atan2(float(p[1]), float(p[0]))
    alpha1 = psi - math.atan2(l2 * math.sin(eta), l1 + l2 * math.cos(eta))
    phi1 = math.atan2(float(L1[2]), float(L1[0]))
    phi2 = math.atan2(float(L2[2]), float(L2[0]))
    q1 = _normalise_angle(phi1 - alpha1)
    q12 = _normalise_angle(phi2 - (alpha1 + eta))
    q2 = _normalise_angle(q12 - q1)
    q3 = _normalise_angle(approach_pitch - q1 - q2)
    return q1, q2, q3


def ik_pose(
    target: Iterable[float],
    *,
    approach_pitch: float = 0.0,
    wrist_roll: float = 0.0,
    elbow: int | None = None,
    radial_sign: int | None = None,
    enforce_limits: bool = True,
    distal_base: np.ndarray = SITE_FROM_GRIPPER,
) -> IKResult:
    """Solve a target point and planar approach pitch without iteration.

    The pan angle is obtained from ``atan2`` after removing the fixed
    tangential offset.  The wrist-flex point is then found by backing off the
    target along the requested pitch, and the shoulder/elbow pair is the
    ordinary two-link cosine-law solution.  ``elbow`` and ``radial_sign`` can
    be left unset to enumerate both analytic branches and select the one with
    the smallest joint displacement.
    """

    target = np.asarray(tuple(target), dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target must be three finite coordinates")
    if not np.isfinite(approach_pitch) or not np.isfinite(wrist_roll):
        raise ValueError("approach_pitch and wrist_roll must be finite")
    elbows = (int(elbow),) if elbow in (-1, 1) else (1, -1)
    radials = (int(radial_sign),) if radial_sign in (-1, 1) else (1, -1)
    h = target[:2] - SHOULDER[:2]
    v4 = _rx_minus(wrist_roll, distal_base)
    tangent = float(A_FROM_SHOULDER[1] + L3[1] + v4[1])
    radius_sq = float(np.dot(h, h))
    radial_sq = radius_sq - tangent * tangent
    if radial_sq < -1e-10:
        raise IKError("target horizontal radius is smaller than the fixed tangential offset")
    h_angle = math.atan2(float(h[1]), float(h[0]))
    candidates: list[IKResult] = []
    for rs in radials:
        radial = int(rs) * math.sqrt(max(0.0, radial_sq))
        q0 = _normalise_angle(math.atan2(tangent, radial) - h_angle)
        for eb in elbows:
            try:
                q1, q2, q3 = _planar_solution(float(radial), float(target[2]),
                                               float(approach_pitch), float(wrist_roll), eb,
                                               distal_base)
            except IKError:
                continue
            q = np.asarray((q0, q1, q2, q3, wrist_roll), dtype=float)
            if enforce_limits and np.any((q < JOINT_LIMITS[:, 0] - 1e-8) | (q > JOINT_LIMITS[:, 1] + 1e-8)):
                continue
            position = _forward_with_distal(q, distal_base)
            candidates.append(IKResult(q=q, position=position,
                                       error=float(np.linalg.norm(position - target)),
                                       approach_pitch=float(approach_pitch),
                                       jaw_yaw=_jaw_yaw(q)))
    if not candidates:
        raise IKError("target has no closed-form solution within joint limits")
    return min(candidates, key=lambda x: float(np.linalg.norm(x.q)))


def ik_pad_pose(
    target: Iterable[float], *, approach_pitch: float = 0.0,
    wrist_roll: float = 0.0, elbow: int | None = None,
    radial_sign: int | None = None, enforce_limits: bool = True,
    grip_q: float = 1.7,
) -> IKResult:
    """Closed-form pose IK for the open-jaw pad midpoint.

    This is the same solver as :func:`ik_pose`, with the distal calibration
    measured at the requested gripper actuator angle.  It is the target used
    by the physical contact expert during approach and grasp.
    """

    return ik_pose(target, approach_pitch=approach_pitch, wrist_roll=wrist_roll,
                   elbow=elbow, radial_sign=radial_sign,
                   enforce_limits=enforce_limits, distal_base=pad_from_gripper(grip_q))


def symmetry_period(size: str) -> float:
    """Return the jaw-yaw repeat period for a Duplo footprint."""

    if size == "2x2":
        return math.pi / 2.0
    if size == "2x4":
        return math.pi
    raise ValueError("size must be '2x2' or '2x4'")


def nearest_symmetric_yaw(yaw: float, size: str, reference: float | None = None) -> float:
    """Choose the equivalent brick yaw nearest the wrist-roll midpoint."""

    period = symmetry_period(size)
    if reference is None:
        reference = float((JOINT_LIMITS[4, 0] + JOINT_LIMITS[4, 1]) / 2.0)
    k = round((float(reference) - float(yaw)) / period)
    return float(yaw + k * period)


def wrist_roll_for_brick(
    target: Iterable[float], brick_yaw: float, size: str, *,
    grip_q: float = 0.6, approach_pitch_max: float = 35.0,
    reference: float | None = None,
) -> float:
    """Choose a wrist roll whose fingertip line crosses the brick's width.

    The jaw line is evaluated from the calibrated fingertip geometry rather
    than assuming that wrist roll equals the brick yaw.  Equivalent brick
    yaws are compared modulo the footprint's symmetry, so a 2x2 and a 2x4
    remain physically interchangeable while the selected roll stays near the
    middle of its actuator range.
    """

    target = np.asarray(tuple(target), dtype=float)
    if target.shape != (3,):
        raise ValueError("target must be three coordinates")
    if reference is None:
        reference = float((JOINT_LIMITS[4, 0] + JOINT_LIMITS[4, 1]) / 2.0)
    # A 2x4 must be crossed across its short side.  A square has two
    # equivalent jaw lines, so include both before reducing the jaw direction
    # modulo pi; this keeps the wrist away from a hard stop for unlucky yaw.
    if size == "2x2":
        desired_angles = (float(brick_yaw), float(brick_yaw) + math.pi / 2.0)
    elif size == "2x4":
        desired_angles = (float(brick_yaw) + math.pi / 2.0,)
    else:
        raise ValueError("size must be '2x2' or '2x4'")
    rolls = np.linspace(JOINT_LIMITS[4, 0] + 1e-5, JOINT_LIMITS[4, 1] - 1e-5, 161)
    candidates: list[tuple[float, float, float]] = []
    for wrist_roll in rolls:
        for degrees in range(int(round(approach_pitch_max)), -1, -1):
            try:
                result = ik_pad_pose(
                    target, approach_pitch=math.radians(degrees),
                    wrist_roll=float(wrist_roll), grip_q=float(grip_q),
                    enforce_limits=True,
                )
            except IKError:
                continue
            v = jaw_vector_at_grip(result.q, grip_q=grip_q)
            actual = math.atan2(float(v[1]), float(v[0]))
            # A jaw line has no direction: either pad can be the left pad.
            diff = min(
                abs((actual - desired + math.pi / 2.0) % math.pi - math.pi / 2.0)
                for desired in desired_angles
            )
            candidates.append((float(diff), abs(float(wrist_roll) - float(reference)), float(wrist_roll)))
            break
    if not candidates:
        raise IKError("no wrist roll can reach the brick target")
    return min(candidates)[2]


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


__all__ = [
    "IKError", "IKResult", "JOINT_LIMITS", "forward_position", "forward_pad_midpoint", "pad_from_gripper", "jaw_vector",
    "ik_pose", "ik_pad_pose", "ik_position", "solve_position", "symmetry_period",
    "nearest_symmetric_yaw",
]
