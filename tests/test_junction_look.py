from __future__ import annotations

import numpy as np

from dryrun.env import StationEnv
from dryrun.mjcf_station import ROOT, station_xml


def test_lab_look_is_the_historical_default() -> None:
    kwargs = {
        "seed": 1001,
        "sentence": "loose flat bricks, bowls on the sides, inspection camera in front",
        "robot_include": str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve()),
    }
    assert station_xml(**kwargs) == station_xml(**kwargs, look="lab")


def test_junction_look_changes_presentation_only() -> None:
    lab = StationEnv(1001, tier="loose", render_size=2, look="lab")
    junction = StationEnv(1001, tier="loose", render_size=2, look="junction")
    try:
        # The look is an explicit asset/material choice and must not alter the
        # seeded layout or the simulator's joint/body trajectory.
        assert lab.look == "lab"
        assert junction.look == "junction"
        assert lab.specs == junction.specs
        lab_q = []
        junction_q = []
        action = np.asarray(lab.observe()["joints"], dtype=np.float32)
        for _ in range(24):
            lab.step(action)
            junction.step(action)
            lab_q.append(lab.data.qpos.copy())
            junction_q.append(junction.data.qpos.copy())
        np.testing.assert_array_equal(np.asarray(lab_q), np.asarray(junction_q))
        assert "#e8642c" not in station_xml(1001, look="lab")
        junction_xml = station_xml(1001, look="junction")
        assert "junction_wood" in junction_xml
        # Junction's measured palette uses a warm orange defective bowl and a
        # cyan good bowl; the table itself is the single un-tinted wood texture.
        assert "0.84 0.3921569 0.172549" in junction_xml
        assert "0.0 0.667 0.707" in junction_xml
    finally:
        lab.close()
        junction.close()
