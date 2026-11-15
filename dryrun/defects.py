"""Seeded, station-camera renders of person-made Duplo defects.

The renderer uses the vendored SO-101 MJCF and the same inspection camera as
the station.  It deliberately keeps the defect labels in a JSONL sidecar; the
pixels passed to the inspector never contain the label.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .mjcf_station import BrickSpec, ROOT, _brick, station_xml

DEFECT_TYPES = ("scribble", "dot", "sticker", "chipped_stud", "missing_stud", "discolouring")
SEEN_FOR_THRESHOLD = ("scribble", "dot", "sticker", "discolouring")
HELD_OUT_TYPES = ("chipped_stud", "missing_stud")
COLORS = ("red", "blue", "yellow", "green")
# Prompt-020's measured presentation point is the brick centre (the TCP is
# several centimetres above/behind it).  Keep this contract identical to the
# station camera and to real expert ``present`` frames.
INSPECTION_PRESENT_POSE = np.array((0.1715, -0.1009, 0.1300), dtype=np.float64)
INSPECTION_CAMERA_POSITION = np.array((0.1194, -0.1430, 0.2296), dtype=np.float64)
INSPECTION_CAMERA_XYAXES = np.array((0.6290, -0.7774, 0.0, 0.6449, 0.5218, 0.5584), dtype=np.float64)
INSPECTION_CAMERA_DISTANCE_M = float(np.linalg.norm(INSPECTION_CAMERA_POSITION - INSPECTION_PRESENT_POSE))
INSPECTION_CAMERA_FOVY_DEG = 40.0
INSPECTION_FRAME_SIZE = 384
INSPECTION_PRESENT_Q = np.asarray((0.5527, -1.2042, 0.8575, 1.0175, -2.3), dtype=np.float64)
_POOLS: dict[tuple[str, str, str, int, int], tuple[mujoco.MjModel, mujoco.MjData, mujoco.Renderer, np.ndarray, np.ndarray]] = {}


def _camera_quat(camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a MuJoCo camera quaternion looking at ``target``."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(camera, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    # Keep the top face upright while avoiding a singular cross product for a
    # future near-vertical station variant.
    if abs(float(np.dot(forward, up))) > 0.97:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    xaxis = np.cross(forward, up)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(-forward, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    rotation = np.column_stack((xaxis, yaxis, -forward))
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return quat


def _xyaxes_quat(xyaxes: np.ndarray) -> np.ndarray:
    """Convert MuJoCo camera ``xyaxes`` (x and y columns) to a quaternion."""
    axes = np.asarray(xyaxes, dtype=np.float64).reshape(2, 3)
    xaxis, yaxis = axes
    xaxis = xaxis / np.linalg.norm(xaxis)
    yaxis = yaxis - xaxis * float(np.dot(xaxis, yaxis))
    yaxis = yaxis / np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)
    rotation = np.column_stack((xaxis, yaxis, zaxis))
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return quat


def configure_inspection_camera(
    model: mujoco.MjModel,
    *,
    target: np.ndarray | None = None,
    camera_position: np.ndarray | None = None,
    fovy: float = INSPECTION_CAMERA_FOVY_DEG,
) -> dict[str, float | list[float]]:
    """Place the station camera at the shared fixed close-up envelope.

    The camera is a body in the MJCF, so moving it after load also moves the
    mount.  Mount geoms are visual station hardware and are disabled for
    collision after the move; otherwise the close camera would become an
    accidental obstacle during the real rollout.  Callers must run
    ``mj_forward`` after this function before reading camera poses.
    """
    target = np.asarray(INSPECTION_PRESENT_POSE if target is None else target, dtype=np.float64)
    camera_position = np.asarray(INSPECTION_CAMERA_POSITION if camera_position is None else camera_position, dtype=np.float64)
    if target.shape != (3,) or camera_position.shape != (3,):
        raise ValueError("inspection target and camera position must be 3-vectors")
    distance = float(np.linalg.norm(camera_position - target))
    if not 0.10 <= distance <= 0.15:
        raise ValueError(f"inspection camera must be 10-15 cm from present pose, got {distance:.4f} m")
    cid = model.camera("inspection_view").id
    mount = model.body("inspection_camera_mount").id
    model.body_pos[mount] = camera_position
    model.cam_pos[cid] = np.zeros(3, dtype=np.float64)
    if target is not None and np.allclose(target, INSPECTION_PRESENT_POSE) and np.allclose(camera_position, INSPECTION_CAMERA_POSITION):
        model.cam_quat[cid] = _xyaxes_quat(INSPECTION_CAMERA_XYAXES)
    else:
        model.cam_quat[cid] = _camera_quat(camera_position, target)
    model.cam_fovy[cid] = float(fovy)
    for geom_id in range(model.body_geomadr[mount], model.body_geomadr[mount] + model.body_geomnum[mount]):
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
    return {
        "present_pose": [float(v) for v in target],
        "camera_position": [float(v) for v in camera_position],
        "distance_m": distance,
        "fovy_deg": float(fovy),
    }


def render_inspection_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    focus_body: int | str | None = None,
) -> np.ndarray:
    """Render the fixed brick crop through the shared station camera path.

    The gripper remains part of the MuJoCo rollout and the top/free cameras,
    but is outside this crop so its wrist housing cannot cover the defect.
    The same crop is used for synthetic frames and rollout frames.
    """
    configure_inspection_camera(model)
    mujoco.mj_forward(model, data)
    # CameraExpert deliberately keeps opaque pick tokens and cannot set the
    # environment's physical active-body field.  At a present event the held
    # brick is the body nearest the measured presentation point; use that
    # geometric focus for the crop while leaving physical rollout state and
    # contact bookkeeping untouched.  Explicit focus_body remains the
    # authoritative path for the measured expert and synthetic renders.
    if focus_body is None:
        brick_ids = [
            body_id
            for body_id in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith("brick_")
        ]
        if brick_ids:
            target_xy = INSPECTION_PRESENT_POSE[:2]
            nearest = min(
                brick_ids,
                key=lambda body_id: float(np.linalg.norm(data.xpos[body_id, :2] - target_xy)),
            )
            if float(np.linalg.norm(data.xpos[nearest, :2] - target_xy)) <= 0.12:
                focus_body = int(nearest)
    old_rgba = model.geom_rgba.copy()
    try:
        _hide_noninspection_geometry(model, focus_body=focus_body)
        renderer.update_scene(data, "inspection_view")
        return renderer.render().copy()
    finally:
        model.geom_rgba[:] = old_rgba


def _hide_noninspection_geometry(
    model: mujoco.MjModel, *, focus_body: int | str | None = None
) -> None:
    """Hide arm/camera hardware for the fixed inspection crop only."""
    # The crop contains the held brick and a neutral table/background only;
    # bowls belong to the top-camera context and otherwise make train and
    # rollout crops depend on the station sentence.
    focus_name = None
    if focus_body is not None:
        if isinstance(focus_body, str):
            focus_name = focus_body
        else:
            focus_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(focus_body)
            )
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        keep = body_name == "table" or geom_name == "table" or geom_name in {"fixed_grip_pad", "moving_grip_pad"}
        if body_name.startswith("brick_") or geom_name.startswith("brick_"):
            keep = focus_name is None or body_name == focus_name
        if not keep:
            model.geom_rgba[geom_id, 3] = 0.0


def _set_inspection_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Place the synthetic arm at the measured Present A configuration."""
    for name, value in zip(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"), INSPECTION_PRESENT_Q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = float(value)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = .60


def _replace_brick(xml: str, spec: BrickSpec) -> str:
    # count=1 means there is exactly one brick body and no nested body tags.
    return re.sub(r'<body name="brick_0".*?</body>', _brick(spec), xml, count=1, flags=re.S)


def render_inspection(seed: int, defect: str = "none", *, size: str | None = None,
                      color: str | None = None, width: int = INSPECTION_FRAME_SIZE,
                      height: int = INSPECTION_FRAME_SIZE) -> np.ndarray:
    """Render one RGB inspection frame from a real MuJoCo station model."""
    if defect != "none" and defect not in DEFECT_TYPES:
        raise ValueError(f"unknown defect: {defect}")
    rng = random.Random(seed)
    size = size or ("2x4" if rng.random() < .42 else "2x2")
    color = color or rng.choice(COLORS)
    # The brick is rendered at the same fixed present pose used by the
    # rollout.  Small offsets and yaw mimic a gripper holding it without
    # changing the camera target or making the camera task easy.
    spec = BrickSpec(0, size, float(INSPECTION_PRESENT_POSE[0]), float(INSPECTION_PRESENT_POSE[1]), defect, color)
    key = (defect, size, color, width, height)
    if key not in _POOLS:
        # The honest loose generator intentionally requires 3--7 bricks.  A
        # single-brick inspection shot therefore loads the smallest legal
        # scene and hides the two non-focused bodies in the shared crop.
        xml = station_xml(0, "good and defective bowls on the sides, inspection camera in front",
                      count=3, robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve()))
        xml = _replace_brick(xml, spec)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height, width)
        configure_inspection_camera(model)
        cid = model.camera("inspection_view").id
        _POOLS[key] = (model, data, renderer, model.cam_pos[cid].copy(), model.light_diffuse.copy())
    model, data, renderer, camera_base, light_base = _POOLS[key]
    body = model.body("brick_0").id
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    yaw = rng.uniform(-math.pi, math.pi)
    data.qpos[adr:adr + 3] = [INSPECTION_PRESENT_POSE[0] + rng.uniform(-.006, .006),
                               INSPECTION_PRESENT_POSE[1] + rng.uniform(-.006, .006),
                               INSPECTION_PRESENT_POSE[2] + rng.uniform(-.006, .010)]
    data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    _set_inspection_pose(model, data)
    # The camera remains at the fixed station pose.  Small key-light jitter is
    # part of the render distribution; camera jitter would make train and
    # rollout crops disagree and is intentionally omitted here.
    cid = model.camera("inspection_view").id
    model.cam_pos[cid] = camera_base
    model.light_diffuse[:] = light_base
    for li in range(model.nlight):
        model.light_diffuse[li] *= rng.uniform(.82, 1.18)
    return render_inspection_frame(model, data, renderer, focus_body="brick_0")


def render_drawing_frame(drawing: str | Path | np.ndarray, seed: int = 0, *, size: str = "2x2",
                         color: str = "red") -> np.ndarray:
    """Put a transparent user drawing on the visible brick face and return RGB.

    Render a segmentation pass and use the projected brick bounds for the
    compositing rectangle.  This keeps drawings on the visible studded face
    despite the few-millimetre grasp jitter and avoids a hard-coded pixel
    location.
    """
    base = Image.fromarray(render_inspection(seed, "none", size=size, color=color)).convert("RGBA")
    if isinstance(drawing, (str, Path)):
        overlay = Image.open(drawing).convert("RGBA")
    else:
        arr = np.asarray(drawing)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4) or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("drawing must be a non-empty HxWx3 or HxWx4 image")
        if not np.issubdtype(arr.dtype, np.uint8):
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        overlay = Image.fromarray(arr, "RGBA" if arr.shape[2] == 4 else "RGB").convert("RGBA")

    # ``render_inspection`` leaves the cached model/data/renderer at the exact
    # seeded pose used for ``base``.  MuJoCo's segmentation buffer stores the
    # visible geom id in channel 0, allowing us to project the brick without
    # guessing where camera jitter put it.
    key = ("none", size, color, base.width, base.height)
    pool = _POOLS.get(key)
    bbox = None
    if pool is not None:
        model, data, renderer, _camera_base, _light_base = pool
        brick_geoms = {
            i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("brick_0_")
        }
        if brick_geoms:
            renderer.enable_segmentation_rendering()
            old_rgba = model.geom_rgba.copy()
            try:
                _hide_noninspection_geometry(model)
                renderer.update_scene(data, "inspection_view")
                segmentation = renderer.render()
            finally:
                model.geom_rgba[:] = old_rgba
                renderer.disable_segmentation_rendering()
            mask = np.isin(segmentation[..., 0], tuple(brick_geoms))
            ys, xs = np.where(mask)
            if len(xs):
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    # A brick should always be visible, but retain a bounded centre fallback
    # for unusual custom station models or a fully occluded pose.
    if bbox is None:
        face_w, face_h = (max(1, base.width // 3), max(1, base.height // 4))
        x0 = (base.width - face_w) // 2; y0 = (base.height - face_h) // 2
        bbox = (x0, y0, x0 + face_w, y0 + face_h)
    x0, y0, x1, y1 = bbox
    overlay = overlay.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.Resampling.LANCZOS)
    base.alpha_composite(overlay, (x0, y0))
    return np.asarray(base.convert("RGB"))


def _write_table_and_sticker_assets(out: Path) -> None:
    """Create deterministic project assets when the image tool is unavailable."""
    out.mkdir(parents=True, exist_ok=True)
    table = Image.new("RGB", (512, 512), (78, 86, 88)); d = ImageDraw.Draw(table)
    rng = random.Random(8080)
    for _ in range(3000):
        x, y = rng.randrange(512), rng.randrange(512); c = rng.choice([(87, 94, 94), (63, 71, 73), (101, 105, 101)])
        d.point((x, y), fill=c)
    table.save(out / "table_surface_seed8080.png")
    sticker = Image.new("RGBA", (256, 256), (0, 0, 0, 0)); d = ImageDraw.Draw(sticker)
    d.rounded_rectangle((20, 20, 236, 236), radius=34, fill=(244, 241, 220, 255), outline=(28, 37, 44, 255), width=7)
    d.polygon([(128, 43), (148, 104), (214, 104), (161, 142), (184, 205), (128, 169), (72, 205), (95, 142), (42, 104), (108, 104)], fill=(232, 71, 57, 255))
    sticker.save(out / "sticker_art_seed8080.png")


def build_dataset(out: Path, *, train_good: int = 128, val_good: int = 160,
                  test_good: int = 400, val_each: int = 50, test_each: int = 300) -> dict:
    """Build train/threshold/test renders. Test has 2,200 images by default."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for p in [out / "good" / "train", out / "good" / "val", out / "good" / "test"]:
        p.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        for typ in DEFECT_TYPES:
            (out / "defects" / split / typ).mkdir(parents=True, exist_ok=True)
    _write_table_and_sticker_assets(out / "assets")
    rows = []
    def emit(path: Path, seed: int, typ: str, split: str):
        cv2.imwrite(str(path), cv2.cvtColor(render_inspection(seed, typ), cv2.COLOR_RGB2BGR))
        rows.append({"path": str(path), "seed": seed, "split": split, "defect": typ, "is_defect": typ != "none"})
    for i in range(train_good): emit(out / "good" / "train" / f"{i:05d}.png", 100000 + i, "none", "train")
    for i in range(val_good): emit(out / "good" / "val" / f"{i:05d}.png", 200000 + i, "none", "val")
    for typ in SEEN_FOR_THRESHOLD:
        for i in range(val_each): emit(out / "defects" / "val" / typ / f"{i:05d}.png", 210000 + 1000 * DEFECT_TYPES.index(typ) + i, typ, "val")
    for i in range(test_good): emit(out / "good" / "test" / f"{i:05d}.png", 300000 + i, "none", "test")
    for typ in DEFECT_TYPES:
        for i in range(test_each): emit(out / "defects" / "test" / typ / f"{i:05d}.png", 310000 + 1000 * DEFECT_TYPES.index(typ) + i, typ, "test")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = {"train_good": train_good, "validation_good": val_good,
              "validation_defects": {t: val_each for t in SEEN_FOR_THRESHOLD},
              "test_good": test_good, "test_defects": {t: test_each for t in DEFECT_TYPES},
              "test_total": test_good + len(DEFECT_TYPES) * test_each,
              "held_out_types": list(HELD_OUT_TYPES), "seed_ranges": {"train": [100000, 100000 + train_good - 1], "test": [300000, 310000 + 1000 * len(DEFECT_TYPES) + test_each - 1]}}
    (out / "dataset.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m2/data")); ap.add_argument("--test-each", type=int, default=300)
    a = ap.parse_args(); print(json.dumps(build_dataset(a.out, test_each=a.test_each), indent=2))


if __name__ == "__main__": main()
