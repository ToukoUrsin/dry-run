"""Camera-only perception for the Dry Run top camera.

The routines in this module deliberately operate on an image and station
calibration only.  In particular, brick bodies, labels, contacts and scene
specifications are not consulted.  A caller may supply masks produced from
the robot's joint readings (for the yellow arm) and from the measured bowl
locations.  This keeps the image processing usable with a real camera too.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


BRICK_SIZES_MM = ((32.0, 32.0), (32.0, 64.0))
BRICK_COLOURS = ("red", "blue", "yellow", "green")
PERCEPTION_RESOLUTION_PX = 960
# These are the measured station bowl centres.  They are calibration, not
# labels or simulator state, and can be replaced for another station.
DEFAULT_BOWL_CENTRES_M = ((-0.0093, -0.2657), (-0.0093, 0.2657))
DEFAULT_BOWL_RADIUS_M = 0.095


@dataclass(frozen=True)
class TopCameraCalibration:
    """World/image geometry for a pinhole camera looking straight down."""

    position_m: tuple[float, float, float] = (0.15, 0.0, 1.20)
    right: tuple[float, float, float] = (1.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    forward: tuple[float, float, float] = (0.0, 0.0, -1.0)
    fovy_deg: float = 45.0
    width_px: int = PERCEPTION_RESOLUTION_PX
    height_px: int = PERCEPTION_RESOLUTION_PX

    @property
    def metres_per_pixel(self) -> float:
        # At z=0.  Use ``scale_m`` for a brick's top face so parallax is
        # corrected rather than silently assigning the table plane scale.
        return 2.0 * self.position_m[2] * math.tan(math.radians(self.fovy_deg) / 2.0) / self.height_px

    def scale_m(self, z_m: float = 0.0) -> float:
        return 2.0 * (self.position_m[2] - float(z_m)) * math.tan(math.radians(self.fovy_deg) / 2.0) / self.height_px


def top_camera_calibration(model, data=None, *, width: int = 960, height: int = 960) -> TopCameraCalibration:
    """Read the compiled ``top_view`` camera, without reading any brick state.

    ``data`` is optional.  When provided it is forwarded through MuJoCo's
    camera kinematics, which also handles a camera attached to a body.  Only
    camera arrays are accessed; no body positions, contacts or qpos values
    are read by this function.
    """
    import mujoco

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view")
    if cid < 0:
        raise ValueError("compiled station has no top_view camera")
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    pos = np.asarray(data.cam_xpos[cid], dtype=float)
    mat = np.asarray(data.cam_xmat[cid], dtype=float).reshape(3, 3)
    # MuJoCo stores the camera basis as columns.  Its optical axis is -z.
    right, up, forward = mat[:, 0], mat[:, 1], -mat[:, 2]
    fovy = float(model.cam_fovy[cid]) or 45.0
    return TopCameraCalibration(
        position_m=tuple(float(v) for v in pos),
        right=tuple(float(v) for v in right),
        up=tuple(float(v) for v in up),
        forward=tuple(float(v) for v in forward),
        fovy_deg=fovy,
        width_px=int(width),
        height_px=int(height),
    )


# A shorter alias is convenient for callers and makes the calibration intent
# obvious in notebooks.
calibrate_top_camera = top_camera_calibration


def project_world_to_pixel(point_m: Sequence[float], calibration: TopCameraCalibration, *, z_m: float | None = None) -> tuple[float, float]:
    """Project a world point onto the top-camera image."""
    raw = np.asarray(point_m, dtype=float).reshape(-1)
    if raw.size == 2:
        p = np.array([raw[0], raw[1], 0.0], dtype=float)
    elif raw.size >= 3:
        p = raw[:3].copy()
    else:
        raise ValueError("point must contain x and y, and optionally z")
    z = float(p[2] if z_m is None else z_m)
    cam = np.asarray(calibration.position_m, dtype=float)
    right = np.asarray(calibration.right, dtype=float)
    up = np.asarray(calibration.up, dtype=float)
    d = cam[2] - z
    if d <= 0:
        raise ValueError("point must be below the top camera")
    span = 2.0 * d * math.tan(math.radians(calibration.fovy_deg) / 2.0)
    rel = p - cam
    x = calibration.width_px / 2.0 + float(np.dot(rel, right)) * calibration.height_px / span
    y = calibration.height_px / 2.0 - float(np.dot(rel, up)) * calibration.height_px / span
    return x, y


def world_from_pixel(pixel: Sequence[float], calibration: TopCameraCalibration, *, z_m: float = 0.0) -> tuple[float, float, float]:
    """Back-project an image pixel to a measured horizontal plane."""
    u, v = float(pixel[0]), float(pixel[1])
    span = 2.0 * (calibration.position_m[2] - float(z_m)) * math.tan(math.radians(calibration.fovy_deg) / 2.0)
    sx = span / calibration.height_px
    sy = span / calibration.height_px
    du = (u - calibration.width_px / 2.0) * sx
    dv = (calibration.height_px / 2.0 - v) * sy
    cam = np.asarray(calibration.position_m, dtype=float)
    p = cam + du * np.asarray(calibration.right) + dv * np.asarray(calibration.up)
    p[2] = float(z_m)
    return tuple(float(x) for x in p)


@dataclass(frozen=True)
class BrickDetection:
    colour: str
    centre_px: tuple[float, float]
    centre_m: tuple[float, float, float]
    yaw_rad: float
    size_mm: tuple[float, float]
    area_px: int
    confidence: float
    status: str = "ok"

    # American spelling is useful to consumers while retaining the station's
    # existing colour spelling.
    @property
    def color(self) -> str:
        return self.colour

    @property
    def center_px(self) -> tuple[float, float]:
        return self.centre_px

    @property
    def center_m(self) -> tuple[float, float, float]:
        return self.centre_m


def _colour_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    rgb = np.asarray(image)[..., :3].astype(np.float32)
    if rgb.max(initial=0) <= 1.5:
        rgb *= 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hi = np.maximum(np.maximum(r, g), b)
    lo = np.minimum(np.minimum(r, g), b)
    sat = (hi - lo) / np.maximum(hi, 1.0)
    # Thresholds intentionally use channel ratios rather than fixed RGB
    # values: exposure and white balance vary on real station cameras.
    return {
        "red": (r > 55) & (r > g * 1.45) & (r > b * 1.35) & (sat > .35),
        "blue": (b > 55) & (b > r * 1.35) & (b > g * 1.12) & (sat > .30),
        "yellow": (r > 85) & (g > 55) & (r > b * 1.65) & (g > b * 1.35) & (sat > .28),
        "green": (g > 45) & (g > r * 1.25) & (g > b * 1.12) & (sat > .28),
    }


def _components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Return 8-connected component coordinates without a CV dependency."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out: list[np.ndarray] = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack = [(y, x)]; seen[y, x] = True; coords: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop(); coords.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; stack.append((ny, nx))
        if len(coords) >= min_area:
            out.append(np.asarray(coords, dtype=np.float32))
    return out


def bowl_mask(shape: tuple[int, int], calibration: TopCameraCalibration, *, centres_m: Iterable[Sequence[float]] = DEFAULT_BOWL_CENTRES_M, radius_m: float = DEFAULT_BOWL_RADIUS_M) -> np.ndarray:
    """Mask the two measured bowl discs in image coordinates."""
    h, w = int(shape[0]), int(shape[1]); yy, xx = np.indices((h, w), dtype=np.float32)
    out = np.zeros((h, w), dtype=bool)
    for centre in centres_m:
        px, py = project_world_to_pixel((float(centre[0]), float(centre[1]), 0.0), calibration)
        rad = float(radius_m) / calibration.scale_m()  # square top-camera pixels
        out |= (xx - px) ** 2 + (yy - py) ** 2 <= rad * rad
    return out


def arm_mask_from_kinematics(shape: tuple[int, int], calibration: TopCameraCalibration, model, joint_positions: Sequence[float] | Mapping[str, float], *, radius_px: float = 22.0) -> np.ndarray:
    """Project the parked robot links into an image mask.

    This helper creates a fresh MuJoCo data object and writes only the six
    measured arm hinge readings.  It selects the named robot link bodies
    before reading their positions, so brick bodies and their state never
    enter the perception path.  A conservative thick-line mask is sufficient
    here: the arm is parked and the goal is to prevent yellow pixels from
    becoming brick detections.
    """
    import mujoco

    h, w = int(shape[0]), int(shape[1]); out = np.zeros((h, w), dtype=bool)
    data = mujoco.MjData(model)
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    if isinstance(joint_positions, Mapping):
        values = {str(k): float(v) for k, v in joint_positions.items()}
    else:
        values = {name: float(v) for name, v in zip(names, joint_positions)}
    for name, value in values.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = value
    mujoco.mj_forward(model, data)
    link_names = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
    points: list[tuple[float, float]] = []
    for name in link_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        # This access is made only after filtering against the fixed robot
        # link allow-list above; no brick body id can be selected here.
        points.append(project_world_to_pixel(data.xpos[bid], calibration))
    if not points:
        return out
    yy, xx = np.indices((h, w), dtype=np.float32)
    r2 = float(radius_px) ** 2
    for px, py in points:
        out |= (xx - px) ** 2 + (yy - py) ** 2 <= r2
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1; den = dx * dx + dy * dy
        t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / max(den, 1e-9), 0.0, 1.0)
        out |= (xx - (x1 + t * dx)) ** 2 + (yy - (y1 + t * dy)) ** 2 <= r2
    return out


def detect_bricks(
    image: np.ndarray,
    calibration: TopCameraCalibration | None = None,
    *,
    arm_mask: np.ndarray | None = None,
    bowls_mask: np.ndarray | None = None,
    min_area_px: int = 100,
    brick_top_z_m: float = 0.0224,
) -> list[BrickDetection]:
    """Segment coloured bricks and fit a rotated rectangle to each blob.

    ``arm_mask`` should be generated from the parked arm's measured joint
    readings and kinematic model.  Supplying it is preferable to colour
    heuristics because the arm is yellow like one brick colour.  Ambiguous
    blobs are returned with ``status='unclear'`` so the planner can push and
    look again rather than guessing.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("image must have shape (height, width, 3) or (height, width, 4)")
    h, w = arr.shape[:2]
    cal = calibration or TopCameraCalibration(width_px=w, height_px=h)
    reject = np.zeros((h, w), dtype=bool)
    if arm_mask is not None:
        reject |= np.asarray(arm_mask, dtype=bool)
    if bowls_mask is not None:
        reject |= np.asarray(bowls_mask, dtype=bool)
    detections: list[BrickDetection] = []
    for colour, colour_mask in _colour_masks(arr).items():
        mask = colour_mask & ~reject
        for coords in _components(mask, int(min_area_px)):
            # coords are (row, col); PCA is performed in image x/y units.
            xy = coords[:, ::-1]
            centre = xy.mean(axis=0)
            centred = xy - centre
            cov = np.cov(centred, rowvar=False) if len(xy) > 2 else np.eye(2)
            vals, vecs = np.linalg.eigh(cov)
            major = vecs[:, int(np.argmax(vals))]
            if major[0] < 0:
                major = -major
            minor = np.array([-major[1], major[0]])
            ext = np.column_stack((centred @ major, centred @ minor))
            spans_px = np.ptp(ext, axis=0) + 1.0
            spans_mm = np.sort(spans_px * cal.scale_m(brick_top_z_m) * 1000.0)
            expected = min(BRICK_SIZES_MM, key=lambda s: float(np.linalg.norm(spans_mm - np.sort(s))))
            residual = float(np.linalg.norm(spans_mm - np.sort(expected)))
            # A thin or overlarge component is generally a touching/stacked
            # brick.  Keep it visible to the caller but mark it unclear.
            rectangularity = float(len(xy) / max(1.0, spans_px[0] * spans_px[1]))
            status = "ok" if residual <= 4.0 and rectangularity >= .58 else "unclear"
            # image y increases down; convert the PCA direction to world yaw.
            yaw = math.atan2(-float(major[1]), float(major[0]))
            if spans_mm[0] >= spans_mm[1] - 1e-6:
                yaw = ((yaw + math.pi / 2.0) % math.pi) - math.pi / 2.0
            centre_m = world_from_pixel(centre, cal, z_m=brick_top_z_m)
            confidence = max(0.0, min(1.0, 1.0 - residual / 12.0)) * max(0.0, min(1.0, rectangularity))
            detections.append(BrickDetection(colour, (float(centre[0]), float(centre[1])), centre_m, float(yaw), (float(spans_mm[0]), float(spans_mm[1])), int(len(xy)), float(confidence), status))
    return detections


def evaluate_perception(detections: Sequence[BrickDetection], ground_truth: Sequence[Mapping | object], *, max_error_m: float = .020) -> dict[str, float | int]:
    """Evaluate detections against labelled poses for held-out scenes."""
    remaining = list(detections); errors: list[float] = []; yaw_errors: list[float] = []; matched = 0
    for truth in ground_truth:
        def get(name, default=None):
            return truth.get(name, default) if isinstance(truth, Mapping) else getattr(truth, name, default)
        colour = get("colour", get("color", None)); pos = get("centre_m", get("center_m", get("position_m", None)))
        if pos is None:
            pos = (float(get("x")), float(get("y")), float(get("z", 0.0224)))
        candidates = [(i, d) for i, d in enumerate(remaining) if colour is None or d.colour == colour]
        if not candidates:
            continue
        i, det = min(candidates, key=lambda item: math.hypot(item[1].centre_m[0] - float(pos[0]), item[1].centre_m[1] - float(pos[1])))
        err = math.hypot(det.centre_m[0] - float(pos[0]), det.centre_m[1] - float(pos[1]))
        if err <= max_error_m:
            matched += 1; errors.append(err)
            yaw = get("yaw_rad", get("yaw", None))
            if yaw is not None:
                # Brick symmetry makes pi/2 and pi equivalent for evaluation.
                delta = abs((det.yaw_rad - float(yaw) + math.pi / 2) % math.pi - math.pi / 2)
                yaw_errors.append(delta)
            remaining.pop(i)
    pct = matched / len(ground_truth) if ground_truth else 1.0
    return {
        "ground_truth": int(len(ground_truth)), "detected": int(matched),
        "detection_rate": float(pct),
        "position_median_mm": float(np.median(errors) * 1000.0) if errors else float("nan"),
        "position_p95_mm": float(np.percentile(errors, 95) * 1000.0) if errors else float("nan"),
        "yaw_median_deg": float(np.degrees(np.median(yaw_errors))) if yaw_errors else float("nan"),
        "yaw_p95_deg": float(np.degrees(np.percentile(yaw_errors, 95))) if yaw_errors else float("nan"),
    }


segment_bricks = detect_bricks
mask_bowls = bowl_mask


__all__ = ["PERCEPTION_RESOLUTION_PX", "TopCameraCalibration", "BrickDetection", "top_camera_calibration", "calibrate_top_camera", "project_world_to_pixel", "world_from_pixel", "bowl_mask", "mask_bowls", "arm_mask_from_kinematics", "detect_bricks", "segment_bricks", "evaluate_perception"]
