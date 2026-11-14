"""Analytic SO-101 IK must agree with the loaded MuJoCo model."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from dryrun.closed_form_ik import forward_position, ik_pose, nearest_symmetric_yaw
from dryrun.mjcf_station import ROOT, station_xml


def test_closed_form_matches_mujoco_on_10000_reachable_poses(tmp_path: Path) -> None:
    xml = tmp_path / "ik.xml"
    xml.write_text(
        station_xml(
            0,
            "one calibration brick",
            count=1,
            robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve()),
        )
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qadr = [int(model.jnt_qposadr[joint]) for joint in joints]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    limits = np.asarray([model.jnt_range[joint] for joint in joints], dtype=float)
    # Stay a small distance away from hard stops so both analytic branches are
    # valid and the check measures kinematics rather than clipping policy.
    limits[:, 0] += 0.03
    limits[:, 1] -= 0.03
    rng = np.random.default_rng(90101)
    max_error = 0.0
    for _ in range(10_000):
        q = limits[:, 0] + rng.random(5) * (limits[:, 1] - limits[:, 0])
        target = forward_position(q)
        result = ik_pose(target, approach_pitch=float(q[1] + q[2] + q[3]), wrist_roll=float(q[4]))
        data.qpos[:] = 0.0
        for address, value in zip(qadr, result.q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        max_error = max(max_error, float(np.linalg.norm(data.site_xpos[site] - target)))
    assert max_error < 1e-3, max_error


def test_symmetric_yaw_uses_brick_period() -> None:
    assert abs(nearest_symmetric_yaw(0.1, "2x2") - nearest_symmetric_yaw(0.1 + np.pi / 2, "2x2")) < 1e-12
    assert abs(nearest_symmetric_yaw(0.1, "2x4") - nearest_symmetric_yaw(0.1 + np.pi, "2x4")) < 1e-12
