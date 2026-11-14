"""The training dataset must not consume the withdrawn loose scene."""
from __future__ import annotations

from dryrun.make_dataset import validate_dataset_request


def test_loose_single_brick_is_rejected_before_rollout():
    try:
        validate_dataset_request("loose", 1)
    except ValueError as exc:
        assert "withdrawn fixed-position" in str(exc)
    else:
        raise AssertionError("loose count=1 must be rejected")


def test_loose_default_uses_seeded_layout():
    validate_dataset_request("loose", None)


def test_calibration_count_remains_available_for_other_tiers():
    validate_dataset_request("messy", 1)
