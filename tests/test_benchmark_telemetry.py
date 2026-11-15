"""The held-out benchmark crosses the shared telemetry boundary."""

from __future__ import annotations

from dryrun.benchmark import _ReplayCapture, _jsonable, _telemetry_samples


def test_public_frames_become_30hz_writer_rows() -> None:
    capture = _ReplayCapture()
    capture.frames = [
        {"t": 0.0, "joints": [1, 2, 3, 4, 5, 6]},
        {"t": 1.0 / 30.0, "joints": [2, 3, 4, 5, 6, 7]},
    ]
    rows = _telemetry_samples(capture, base_timestamp=100.0)
    assert rows == [
        {"ts": 100.0, "qpos": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "targets": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "gripper": 6.0, "gripper_effort": 0.0},
        {"ts": 100.0 + 1.0 / 30.0, "qpos": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "targets": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "gripper": 7.0, "gripper_effort": 0.0},
    ]


def test_benchmark_source_wires_setup_write_and_refresh() -> None:
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "dryrun" / "benchmark.py"
    text = source.read_text()
    assert "telemetry.setup()" in text
    assert "telemetry.record_rollout(" in text
    assert "telemetry.refresh()" in text


def test_event_json_sanitizes_nonfinite_ik_errors() -> None:
    assert _jsonable({"ik_error_m": float("inf"), "nested": [float("nan")]}) == {
        "ik_error_m": None,
        "nested": [None],
    }
