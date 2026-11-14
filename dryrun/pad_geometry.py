"""Physical fingertip-pad clearance checks for the M1 grasp gate.

The check measures the actual MuJoCo boxes rather than trusting their XML
positions.  ``free_gap_m`` is the separation of the pad faces along the line
between their centres, after projecting each oriented box onto that line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import math

from .mjcf_station import ROOT, station_xml

BRICK_WIDTH_M = 0.0318
SAFE_CLOSE_Q_MIN = 0.20
MAX_PAD_LEVEL_ERROR_M = 0.003
MAX_PAD_ALIGNMENT_ERROR_RAD = math.radians(3.0)


def _model(xml: str | Path | None = None) -> mujoco.MjModel:
    if xml is not None:
        return mujoco.MjModel.from_xml_path(str(xml))
    text = station_xml(
        901,
        "single brick, bowls behind the arm, inspection camera in front",
        # The measured recipe uses a 3--7 brick loose scene.  The clearance
        # probe only needs the robot and pads, so load the smallest valid
        # station rather than reviving the withdrawn one-brick fixture.
        count=3,
        robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve()),
    )
    return mujoco.MjModel.from_xml_string(text)


def measure(xml: str | Path | None = None, *, samples: int = 101) -> list[dict[str, float | bool]]:
    """Measure centre distance, face clearance, and level error over q."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    model = _model(xml)
    data = mujoco.MjData(model)
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    qadr = int(model.jnt_qposadr[grip])
    fixed = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_grip_pad")
    moving = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_grip_pad")
    lo, hi = (float(model.jnt_range[grip, 0]), float(model.jnt_range[grip, 1]))
    rows: list[dict[str, float | bool]] = []
    for q in np.linspace(lo, hi, int(samples)):
        data.qpos[:] = 0.0
        data.qpos[qadr] = float(q)
        mujoco.mj_forward(model, data)
        vector = np.asarray(data.geom_xpos[moving] - data.geom_xpos[fixed], dtype=float)
        distance = float(np.linalg.norm(vector))
        direction = vector / distance if distance > 1e-12 else np.zeros(3)
        support = []
        for geom in (fixed, moving):
            rotation = np.asarray(data.geom_xmat[geom], dtype=float).reshape(3, 3)
            support.append(float(np.abs(rotation.T @ direction) @ model.geom_size[geom]))
        free_gap = distance - sum(support)
        level_error = abs(float(vector[2]))
        fixed_axis = np.asarray(data.geom_xmat[fixed], dtype=float).reshape(3, 3)[:, 0]
        moving_axis = np.asarray(data.geom_xmat[moving], dtype=float).reshape(3, 3)[:, 0]
        alignment_error = float(np.arccos(np.clip(abs(np.dot(fixed_axis, moving_axis)), -1.0, 1.0)))
        rows.append({
            "gripper_q": float(q),
            "centre_distance_m": distance,
            "free_gap_m": float(free_gap),
            "pad_level_error_m": level_error,
            "pad_alignment_error_rad": alignment_error,
            "fits_brick": bool(free_gap >= BRICK_WIDTH_M),
            "safe_close": bool(q >= SAFE_CLOSE_Q_MIN and alignment_error <= MAX_PAD_ALIGNMENT_ERROR_RAD and free_gap >= BRICK_WIDTH_M),
        })
    return rows


def safe_opening(rows: list[dict[str, float | bool]]) -> dict[str, float] | None:
    """Return the first level, brick-fitting close opening, if one exists."""
    candidates = [row for row in rows if bool(row["safe_close"])]
    if not candidates:
        return None
    row = min(candidates, key=lambda item: float(item["gripper_q"]))
    return {key: float(row[key]) for key in ("gripper_q", "centre_distance_m", "free_gap_m", "pad_level_error_m", "pad_alignment_error_rad")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--out", type=Path, default=Path("/bulk3/runs/dryrun/m1/pad_geometry.json"))
    args = parser.parse_args()
    rows = measure(args.xml, samples=args.samples)
    result = {
        "brick_width_m": BRICK_WIDTH_M,
        "safe_close_q_min": SAFE_CLOSE_Q_MIN,
        "max_pad_level_error_m": MAX_PAD_LEVEL_ERROR_M,
        "rows": rows,
        "safe_opening": safe_opening(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("brick_width_m", "safe_close_q_min", "max_pad_level_error_m", "safe_opening",)}, indent=2))


if __name__ == "__main__":
    main()
