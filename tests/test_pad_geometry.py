"""Prompt 017 regression checks for the physical fingertip clearance gate."""
from __future__ import annotations

from dryrun.pad_geometry import (
    BRICK_WIDTH_M,
    MAX_PAD_ALIGNMENT_ERROR_RAD,
    measure,
    safe_opening,
)


def test_pad_measurement_covers_the_actuator_range():
    rows = measure(samples=31)
    assert len(rows) == 31
    assert rows[0]["gripper_q"] < rows[-1]["gripper_q"]
    assert all(float(row["centre_distance_m"]) >= 0 for row in rows)
    assert all("free_gap_m" in row and "pad_alignment_error_rad" in row for row in rows)


def test_fingertip_geometry_has_a_level_brick_fitting_opening():
    rows = measure(samples=101)
    opening = safe_opening(rows)
    assert opening is not None
    assert opening["free_gap_m"] >= BRICK_WIDTH_M
    assert opening["pad_alignment_error_rad"] <= MAX_PAD_ALIGNMENT_ERROR_RAD
    # The open pose used before descent has useful clearance beyond the brick
    # width, while the fingertip pads still close level on the brick.
    assert max(float(row["free_gap_m"]) for row in rows) >= 0.08
