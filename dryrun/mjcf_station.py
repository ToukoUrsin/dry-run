"""Generate and validate a real MuJoCo Dry Run station.

The generator deliberately emits ordinary MJCF rather than moving objects in
Python.  Bricks are free bodies, studs are separate collision geoms, and all
scene choices are deterministic from the sentence and seed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from .closed_form_ik import nearest_symmetric_yaw

ROOT = Path(__file__).resolve().parents[1]
ROBOT_INCLUDE = "SO101/Simulation/SO101/so101_dryrun.xml"
COLORS = ("red", "blue", "yellow", "green")
DEFECTS = ("none", "scribble", "dot", "sticker", "chipped_stud", "missing_stud", "discolouring")
TABLE_TOP = 0.0
BRICK_H = 0.0192  # Duplo plate height, metres (19.2 mm)
STUD_H = 0.0032
STUD_R = 0.0040
PITCH = 0.0160


@dataclass(frozen=True)
class BrickSpec:
    index: int
    size: str
    x: float
    y: float
    defect: str
    color: str
    # The free body's yaw is part of the seeded scene contract. Keep a
    # default for older inspection fixtures that construct a BrickSpec
    # positionally.
    yaw: float = 0.0


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;")


def _quat_z(angle: float) -> str:
    return f' quat="{math.cos(angle/2):.8f} 0 0 {math.sin(angle/2):.8f}"'


def _xyaxes(cam: np.ndarray, target: np.ndarray) -> str:
    """Return MuJoCo camera x/y axes looking from cam towards target."""
    forward = target - cam
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up))) > 0.97:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = np.cross(forward, up)
        x /= np.linalg.norm(x)
    zlocal = -forward
    y = np.cross(zlocal, x)
    y /= np.linalg.norm(y)
    return " ".join(f"{v:.8f}" for v in (*x, *y))


def parse_sentence(sentence: str) -> dict[str, str]:
    s = sentence.lower()
    # Parse the bowl clause independently from the camera clause.  A sentence
    # such as "bowls on the sides, inspection camera in front" previously
    # matched the camera word ``front`` and silently put both bowls together.
    if "bowls behind" in s or "behind the arm" in s:
        bowl_layout = "behind"
    elif "bowls in front" in s or "in front of the arm" in s:
        bowl_layout = "front"
    else:
        bowl_layout = "sides"
    return {
        "layout": "tray" if "tray" in s else ("messy" if "messy" in s else ("pile" if "pile" in s else "grid")),
        "inspection": "left" if "left" in s else ("right" if "right" in s else "front"),
        "bowls": bowl_layout,
    }


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float = .003) -> bool:
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def layout_bricks(seed: int, sentence: str, count: int | None = None) -> list[BrickSpec]:
    """Return the measured loose-tier layout envelope from prompt 019.

    Positions are sampled in polar coordinates about the pan axis.  This is
    deliberately independent of the old reach-map fixture: the recipe's
    planner owns the clear-grasp and IK gate, while this function only emits
    seeded scene geometry for the MJCF.
    """
    opts = parse_sentence(sentence)
    rng = random.Random(seed + {"grid": 0, "messy": 100_003, "pile": 200_009, "tray": 300_007}[opts["layout"]])
    n = int(count) if count is not None else rng.randint(3, 7)
    if n < 3 or n > 7:
        raise ValueError("loose scenes contain 3 to 7 bricks")
    # Prompt 019: radius 0.12--0.24 m and azimuth ±0.85 rad from the pan
    # axis (0.0388, 0). Keep the deterministic candidate lattice dense
    # enough that every seed gets a distinct, well-spaced scene.
    pan_x = 0.0388
    accepted = []
    for radius in np.linspace(.12, .24, 9):
        for azimuth in np.linspace(-.85, .85, 18):
            accepted.append({"x": pan_x + float(radius) * math.cos(float(azimuth)),
                             "y": float(radius) * math.sin(float(azimuth)),
                             "radius": float(radius), "azimuth": float(azimuth),
                             "tested_yaws_rad": (0.0, math.pi / 2.0, math.pi)})
    # The measured bowls are outside the brick envelope; good is -y.
    bowls = ((-.0093, -.2657), (-.0093, .2657))

    def add_row(row: dict, index: int, size: str, used: set[tuple[float, float]], rects: list[tuple[float, float, float, float]], placed: list[BrickSpec]) -> bool:
        x, y = round(float(row["x"]), 4), round(float(row["y"]), 4)
        if (x, y) in used:
            return False
        yaws = list(float(v) for v in row.get("tested_yaws_rad", (0.0,)))
        random.Random(int(seed) * 9176 + index * 131 + int(x * 1000) + int(y * 1000)).shuffle(yaws)
        hx, hy = ((.032, .016) if size == "2x4" else (.016, .016))
        period = math.pi / 2.0 if size == "2x2" else math.pi
        branch_rng = random.Random(int(seed) * 9176 + index * 131 + int(x * 1000) + int(y * 1000) + 17)
        for yaw_base in yaws:
            yaw = float(yaw_base) + branch_rng.randrange(4 if size == "2x2" else 2) * period
            yaw += branch_rng.uniform(-math.pi / 8.0, math.pi / 8.0)
            cy, sy = math.cos(yaw), math.sin(yaw)
            bx, by = abs(hx * cy) + abs(hy * sy), abs(hx * sy) + abs(hy * cy)
            rect = (x - bx, y - by, x + bx, y + by)
            if any((x - cx) ** 2 + (y - cy) ** 2 < .095 ** 2 for cx, cy in bowls):
                continue
            # Measured recipe spacing: 78 mm for two 2x2s and 95 mm when a
            # 2x4 is involved.  Use centre distance, then the visual rectangle
            # check as a second guard against corner contact.
            for old_spec, old in zip(placed, rects):
                minimum = .078 if size == old_spec.size == "2x2" else .095
                if math.hypot(x - old_spec.x, y - old_spec.y) < minimum:
                    break
            else:
                if any(_rects_overlap(rect, old) for old in rects):
                    continue
                # accepted candidate
                pass
            if len(placed) and any(math.hypot(x - old_spec.x, y - old_spec.y) < (.078 if size == old_spec.size == "2x2" else .095) for old_spec in placed):
                continue
            defect_rng = random.Random(int(seed) * 65537 + index * 101)
            defect = "none" if defect_rng.random() < .58 else defect_rng.choice(DEFECTS[1:])
            color = COLORS[(int(seed) + index) % len(COLORS)]
            used.add((x, y)); rects.append(rect)
            placed.append(BrickSpec(index, size, x, y, defect, color, round(yaw, 6)))
            return True
        return False

    for attempt in range(200):
        used: set[tuple[float, float]] = set(); rects: list[tuple[float, float, float, float]] = []; placed: list[BrickSpec] = []
        rows = accepted[:]
        random.Random(int(seed) * 7919 + attempt * 313).shuffle(rows)
        for idx, row in enumerate(rows):
            if len(placed) >= n:
                break
            size = "2x4" if (idx == 0 or (rng.random() < .34 and not any(p.size == "2x4" for p in placed))) else "2x2"
            add_row(row, len(placed), size, used, rects, placed)
        if len(placed) == n and any(p.size == "2x4" for p in placed) and any(p.size == "2x2" for p in placed):
            return placed
    raise ValueError(f"could not place measured {n}-brick scene in seed {seed}")

def _bowl(name: str, x: float, y: float, material: str) -> str:
    # Measured 190 mm inner diameter with a 60 mm wall.  The centre is kept
    # outside the loose brick work area so a top-down grasp has clearance.
    radius = .095
    pieces = [f'<body name="{name}" pos="{x:.4f} {y:.4f} 0">',
              f'<geom name="{name}_floor" type="cylinder" size="0.095 0.004" pos="0 0 0.004" material="{material}" friction="0.6 0.05 0.01"/>']
    for j in range(24):
        a = 2 * math.pi * j / 24
        # Tangential boxes form an open bowl: a brick can be dropped inside
        # and the inspection camera can still see the contents.
        px, py = radius * math.cos(a), radius * math.sin(a)
        pieces.append(f'<geom name="{name}_wall_{j}" type="box" size="0.005 0.0135 0.030" pos="{px:.5f} {py:.5f} 0.030"{_quat_z(a)} material="{material}" friction="0.6 0.05 0.01"/>')
    pieces.append('</body>')
    return "".join(pieces)


def _brick(spec: BrickSpec) -> str:
    hx, hy = ((.032, .016) if spec.size == "2x4" else (.016, .016))
    cols = 4 if spec.size == "2x2" else 8
    bits = [f'<body name="brick_{spec.index}" pos="{spec.x:.4f} {spec.y:.4f} {BRICK_H/2:.5f}"{_quat_z(float(spec.yaw))}><freejoint/>',
            f'<geom name="brick_{spec.index}_body" type="box" size="{hx:.4f} {hy:.4f} {BRICK_H/2:.5f}" material="brick_{spec.color}" mass="0.018" friction="0.5 0.01 0.001" condim="4"/>']
    # Stud centres are at the standard 16 mm Duplo pitch.  Missing/chipped
    # studs are represented by absent/reduced collision geometry, not labels.
    stud_i = 0
    xvals = (-.024, -.008, .008, .024) if spec.size == "2x4" else (-.008, .008)
    yvals = (-.008, .008)
    for xx in xvals:
        for yy in yvals:
            missing = spec.defect == "missing_stud" and stud_i == cols - 1
            if not missing:
                rad = .003 if spec.defect == "chipped_stud" and stud_i == cols - 1 else STUD_R
                hh = .0012 if spec.defect == "chipped_stud" and stud_i == cols - 1 else STUD_H / 2
                bits.append(f'<geom name="brick_{spec.index}_stud_{stud_i}" type="cylinder" size="{rad:.4f} {hh:.4f}" pos="{xx:.4f} {yy:.4f} {BRICK_H/2 + STUD_H/2:.5f}" material="brick_{spec.color}" friction="0.5 0.2 0.05"/>')
            stud_i += 1
    top = BRICK_H / 2 + STUD_H + .0002
    # Stud defects are judged from this top-face shot. Add a visible dark
    # cavity/chip mark at the altered stud so a person and a camera can see
    # the defect; the collision geometry above still models the physical
    # missing/reduced stud.
    if spec.defect in {"missing_stud", "chipped_stud"}:
        mx, my = xvals[-1], yvals[-1]
        radius = .006 if spec.defect == "missing_stud" else .0045
        depth = .00035 if spec.defect == "missing_stud" else .00025
        bits.append(f'<geom name="brick_{spec.index}_{spec.defect}_mark" type="cylinder" size="{radius:.4f} {depth:.4f}" pos="{mx:.4f} {my:.4f} {top:.5f}" material="defect_black" contype="0" conaffinity="0"/>')
    if spec.defect == "sticker":
        bits.append(f'<geom name="brick_{spec.index}_sticker" type="cylinder" size="0.010 0.0005" pos="0 0 {top:.5f}" material="sticker" contype="0" conaffinity="0"/>')
    elif spec.defect == "dot":
        bits.append(f'<geom name="brick_{spec.index}_dot" type="cylinder" size="0.003 0.0006" pos="0 0 {top:.5f}" material="defect_black" contype="0" conaffinity="0"/>')
    elif spec.defect == "discolouring":
        bits.append(f'<geom name="brick_{spec.index}_discolour" type="box" size="0.011 0.008 0.0005" pos="0.008 -0.006 {top:.5f}" material="discolour" contype="0" conaffinity="0"/>')
    elif spec.defect == "scribble":
        for j, (xx, yy, ang) in enumerate(((0, 0, .2), (.006, .003, -.3), (-.006, -.003, .5))):
            bits.append(f'<geom name="brick_{spec.index}_scribble_{j}" type="box" size="0.007 0.0008 0.0005" pos="{xx:.4f} {yy:.4f} {top:.5f}"{_quat_z(ang)} material="defect_black" contype="0" conaffinity="0"/>')
    bits.append('</body>')
    return "".join(bits)


def station_xml(seed: int = 7, sentence: str = "good and defective bowls on the sides, inspection camera in front", count: int | None = None, robot_include: str = ROBOT_INCLUDE) -> str:
    opts = parse_sentence(sentence)
    bricks = layout_bricks(seed, sentence, count)
    # Bowl centres stay inside the SO-101's working envelope.  The sentence
    # changes their arrangement while preserving camera and reach validity.
    # Both bowls are on reachable, separated drop points; the sentence only
    # changes which side is labelled good versus defective.
    # Prompt 019: side bowls at ±0.27 m from the pan axis, with good on -y.
    # Sentence variants may change the camera, but the measured loose station
    # keeps this safe drop arrangement.
    good, bad = (-.0093, -.2657), (-.0093, .2657)
    if opts["inspection"] == "left":
        cam = np.array([-.48, -.23, .34])
    elif opts["inspection"] == "right":
        cam = np.array([.48, -.23, .34])
    else:
        cam = np.array([0.0, -.52, .30])
    target = np.array([0.0, 0.0, .12])
    axes = _xyaxes(cam, target)
    pieces = [f'''<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="dryrun_station_seed_{seed}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="10" noslip_iterations="3" iterations="100" ls_iterations="50"/>
  <include file="{_xml_escape(robot_include)}"/>
  <visual><headlight diffuse="0.8 0.8 0.8" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/><global offwidth="960" offheight="720"/></visual>
  <asset>
    <material name="table_mat" rgba="0.18 0.21 0.24 1"/>
    <material name="good_bowl" rgba="0.08 0.55 0.25 1"/>
    <material name="bad_bowl" rgba="0.75 0.12 0.12 1"/>
    <material name="brick_red" rgba="0.75 0.08 0.04 1"/><material name="brick_blue" rgba="0.05 0.25 0.8 1"/>
    <material name="brick_yellow" rgba="0.95 0.65 0.04 1"/><material name="brick_green" rgba="0.1 0.65 0.2 1"/>
    <material name="sticker" rgba="0.95 0.95 0.95 1"/><material name="defect_black" rgba="0.03 0.03 0.03 1"/><material name="discolour" rgba="0.20 0.08 0.05 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.25 -0.45 1.5" dir="-0.2 0.3 -1" directional="true" castshadow="true"/>
  <geom name="table" type="box" pos="0.15 0 -0.045" size="0.62 0.48 0.045" material="table_mat" friction="0.5 0.1 0.02" solref="0.0002 1" solimp="0.99 0.999 0.001"/>
    <geom name="table_edge" type="box" pos="0.15 0.47 0.04" size="0.62 0.025 0.08" material="table_mat"/>
    {_bowl("good_bowl", *good, "good_bowl")}
    {_bowl("defective_bowl", *bad, "bad_bowl")}
    <!-- Rigid camera mounts: the camera is a physical station component, not a floating site. -->
    <body name="top_camera_mount" pos="-0.40 0.35 1.20"><geom type="box" size="0.025 0.025 0.60" pos="0 0 -0.60" material="table_mat"/><geom type="box" size="0.58 0.025 0.025" pos="0.29 -0.175 0" material="table_mat"/><camera name="top_view" pos="0.55 -0.35 0" xyaxes="1 0 0 0 1 0"/>
    </body>
    <body name="inspection_camera_mount" pos="{cam[0]:.4f} {cam[1]:.4f} {cam[2]:.4f}"><geom type="box" size="0.018 0.018 0.17" pos="0 0 -0.17" material="table_mat"/><geom type="box" size="0.06 0.06 0.018" pos="0 0 -0.34" material="table_mat"/><camera name="inspection_view" pos="0 0 0" xyaxes="{axes}"/></body>
    <camera name="free_view" pos="0.95 -1.15 0.85" xyaxes="0.76 0.65 0 -0.27 0.32 0.91"/>
    {''.join(_brick(b) for b in bricks)}
  </worldbody>
</mujoco>''']
    return "".join(pieces)


def _hinge_ids(model: mujoco.MjModel) -> list[int]:
    names = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}
    return [i for i in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) in names]


def ik_reachable(model: mujoco.MjModel, target: Iterable[float], tolerance: float = .045) -> tuple[bool, float]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if sid < 0:
        return False, float("inf")
    d = mujoco.MjData(model)
    joints = _hinge_ids(model)
    qadr = [model.jnt_qposadr[j] for j in joints]
    target = np.asarray(target, dtype=float)
    for _ in range(160):
        mujoco.mj_forward(model, d)
        err = target - d.site_xpos[sid]
        if np.linalg.norm(err) < tolerance:
            return True, float(np.linalg.norm(err))
        jac = np.zeros((3, model.nv)); mujoco.mj_jacSite(model, d, jac, None, sid)
        j = jac[:, [model.jnt_dofadr[jid] for jid in joints]]
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), err)
        for k, adr in enumerate(qadr):
            d.qpos[adr] = np.clip(d.qpos[adr] + .65 * dq[k], model.jnt_range[joints[k], 0], model.jnt_range[joints[k], 1])
    mujoco.mj_forward(model, d)
    return False, float(np.linalg.norm(target - d.site_xpos[sid]))


def check_station(path: str | Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    brick_ids = [i for i in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("brick_")]
    # At initial rest only table-brick and bowl-floor contacts are expected.
    bad_contacts = []
    bowl_overlap = []
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    brick_geom = set()
    for bi in brick_ids:
        for gi in range(model.body_geomadr[bi], model.body_geomadr[bi] + model.body_geomnum[bi]):
            brick_geom.add(gi)
    bowl_geom = {}
    for bowl_name in ("good_bowl", "defective_bowl"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bowl_name)
        bowl_geom[bowl_name] = set(range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]))
    good_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "good_bowl")
    bad_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "defective_bowl")
    centre_distance = float(np.linalg.norm(data.xpos[good_bid, :2] - data.xpos[bad_bid, :2]))
    if centre_distance < 0.15:  # two 75 mm-radius bowls would overlap
        bowl_overlap.append({"reason": "bowl_centres_too_close", "distance_m": centre_distance})
    for ci in range(data.ncon):
        c = data.contact[ci]
        if c.dist < -1e-6 and ((c.geom1 in brick_geom and c.geom2 != table_id) or (c.geom2 in brick_geom and c.geom1 != table_id)):
            bad_contacts.append((int(c.geom1), int(c.geom2), float(c.dist)))
        if c.dist < -1e-6 and ((c.geom1 in bowl_geom["good_bowl"] and c.geom2 in bowl_geom["defective_bowl"]) or (c.geom2 in bowl_geom["good_bowl"] and c.geom1 in bowl_geom["defective_bowl"])):
            bowl_overlap.append({"geom1": int(c.geom1), "geom2": int(c.geom2), "dist_m": float(c.dist)})
    # Test all brick centres plus both bowl centres with the same damped IK.
    reaches = {}
    for bi in brick_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi)
        reaches[name] = ik_reachable(model, data.xpos[bi] + np.array([0, 0, .05]))
    for name in ("good_bowl", "defective_bowl"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        reaches[name] = ik_reachable(model, data.xpos[bid] + np.array([0, 0, .10]))
    cam_checks = {}
    for name in ("free_view", "top_view", "inspection_view"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        to_target = np.array([.0, 0.0, .10]) - data.cam_xpos[cid]
        forward = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
        cam_checks[name] = {"forward_dot": float(np.dot(forward, to_target / np.linalg.norm(to_target))), "sees_station": bool(np.dot(forward, to_target) > 0)}
    return {"loaded": True, "nq": model.nq, "nv": model.nv, "nu": model.nu, "bodies": model.nbody, "geoms": model.ngeom, "camera_count": model.ncam,
            "brick_count": len(brick_ids), "intersections": bad_contacts, "bowl_overlap": bowl_overlap, "ik": {k: {"ok": v[0], "error_m": v[1]} for k, v in reaches.items()}, "camera_checks": cam_checks,
            "pass": not bad_contacts and not bowl_overlap and all(v[0] for v in reaches.values()) and all(v["sees_station"] for v in cam_checks.values())}


def rejection_reason(result: dict) -> str | None:
    if result.get("intersections"):
        return f"rest_intersections:{len(result['intersections'])}"
    if result.get("bowl_overlap"):
        return f"bowl_overlap:{len(result['bowl_overlap'])}"
    failed_ik = [name for name, value in result.get("ik", {}).items() if not value.get("ok")]
    if failed_ik:
        return "ik_unreachable:" + ",".join(failed_ik)
    failed_cameras = [name for name, value in result.get("camera_checks", {}).items() if not value.get("sees_station")]
    if failed_cameras:
        return "camera_misses:" + ",".join(failed_cameras)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--seed", type=int, default=7); g.add_argument("--sentence", default="good and defective bowls on the sides, inspection camera in front"); g.add_argument("--count", type=int); g.add_argument("--out", default="assets/station.xml")
    c = sub.add_parser("check"); c.add_argument("xml")
    b = sub.add_parser("batch"); b.add_argument("--out-dir", default="/bulk2/runs/dryrun/m1/stations"); b.add_argument("--record", default="/bulk2/runs/dryrun/m1/stations.jsonl"); b.add_argument("--seeds", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "generate":
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(station_xml(a.seed, a.sentence, a.count, robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve())))
        print(json.dumps({"out": str(out), "seed": a.seed, "sentence": a.sentence, "bricks": len(layout_bricks(a.seed, a.sentence, a.count))}, indent=2))
    elif a.cmd == "check":
        print(json.dumps(check_station(a.xml), indent=2))
    else:
        out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        sentences = [
            "messy pile, bowls behind the arm, inspection camera on the left",
            "regular tray, bowls in front, inspection camera on the right",
            "scattered bricks, bowls on the sides, inspection camera in front",
            "a tray instead of a pile, bowls behind the arm, inspection camera on the left",
        ]
        rows = []
        for seed in range(a.seeds):
            sentence = sentences[seed % len(sentences)]
            xml_path = out_dir / f"station_{seed:04d}.xml"
            try:
                xml_path.write_text(station_xml(seed, sentence, robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve())))
                result = check_station(xml_path)
                row = {"seed": seed, "sentence": sentence, "xml": str(xml_path), "pass": result["pass"], "reason": rejection_reason(result), "check": result}
            except Exception as exc:
                row = {"seed": seed, "sentence": sentence, "xml": str(xml_path), "pass": False, "reason": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
        record = Path(a.record); record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        print(json.dumps({"attempted": len(rows), "passed": sum(r["pass"] for r in rows), "rejected": sum(not r["pass"] for r in rows), "record": str(record)}, indent=2))


if __name__ == "__main__":
    main()
