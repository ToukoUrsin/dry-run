"""Regression checks against the rigged one-brick loose benchmark."""
from __future__ import annotations

import ast
from pathlib import Path

from dryrun.env import TIERS
from dryrun.mjcf_station import _brick, layout_bricks, BrickSpec
from dryrun.benchmark import _benchmark_groups, _contact_count


ROOT = Path(__file__).resolve().parents[1]


def test_loose_seed_controls_a_real_multi_brick_scene():
    scenes = [layout_bricks(seed, TIERS["loose"]) for seed in range(20)]
    assert all(3 <= len(scene) <= 8 for scene in scenes)
    signatures = {tuple((b.size, b.x, b.y, b.color, b.defect) for b in scene) for scene in scenes}
    assert len(signatures) >= 18, "loose seeds must not repeat one fixed scene"
    assert len({len(scene) for scene in scenes}) >= 2, "seed must affect brick count"
    assert all({b.size for b in scene} == {"2x2", "2x4"} for scene in scenes)
    assert {b.color for scene in scenes for b in scene} == {"red", "blue", "yellow", "green"}
    assert len({round(b.yaw, 6) for scene in scenes for b in scene}) >= 20
    # Positions must vary with the seed as well as metadata.  A fixed pose
    # with changing colours/defects is still the withdrawn rigged fixture.
    positions = {
        tuple((b.size, round(b.x, 4), round(b.y, 4), round(b.yaw, 6)) for b in scene)
        for scene in scenes
    }
    assert len(positions) >= 18
    # Prompt 019 replaces the withdrawn reach-map fixture with the measured
    # recipe envelope; the expert planner performs the clear-grasp and IK gate.
    for scene in scenes:
        for brick in scene:
            radius = ((brick.x - 0.0388) ** 2 + brick.y ** 2) ** 0.5
            azimuth = __import__("math").atan2(brick.y, brick.x - 0.0388)
            assert 0.12 - 1e-4 <= radius <= 0.24 + 1e-4
            assert abs(azimuth) <= 0.851


def test_every_tier_is_seeded_and_multi_brick():
    for tier, sentence in TIERS.items():
        scenes = [layout_bricks(seed, sentence) for seed in range(8)]
        bounds = {"loose": (3, 7), "messy": (5, 9), "pile": (6, 10)}[tier]
        assert all(bounds[0] <= len(scene) <= bounds[1] for scene in scenes)
        assert len({tuple((b.size, round(b.x, 4), round(b.y, 4), round(b.yaw, 6)) for b in scene)
                    for scene in scenes}) >= 6, tier


def test_brick_yaw_is_emitted_on_the_free_body():
    xml = _brick(BrickSpec(0, "2x4", .26, -.04, "none", "red", .37))
    assert 'name="brick_0"' in xml
    assert 'quat="0.982936' in xml


def test_benchmark_has_no_position_override_or_one_brick_special_case():
    source = (ROOT / "dryrun" / "benchmark.py").read_text()
    tree = ast.parse(source)
    assert "re.sub" not in source
    assert "count=1" not in source
    assert "0.2600" not in source
    assert not any(isinstance(node, ast.Import) and any(alias.name == "re" for alias in node.names) for node in tree.body)


def test_benchmark_uses_the_public_rollout_boundary():
    source = (ROOT / "dryrun" / "benchmark.py").read_text()
    assert "from .rollout import run as rollout_run" in source
    assert "from .render import rollout" not in source


def test_benchmark_uses_fixed_tier_partitions_and_rejects_duplicates():
    assert _benchmark_groups(list(range(200)), "pile")["pile"] == list(range(150, 200))
    assert _benchmark_groups(list(range(150, 155)), "pile")["pile"] == list(range(150, 155))
    try:
        _benchmark_groups(list(range(5)), "pile")
    except ValueError:
        pass
    else:
        raise AssertionError("positional non-pile seeds must be rejected")
    try:
        _benchmark_groups([0, 0], None)
    except ValueError as exc:
        assert "duplicates" in str(exc)
    else:
        raise AssertionError("duplicate benchmark seeds must be rejected")


def test_legacy_contact_rows_are_deduplicated_by_brick():
    events = [
        {"event": "grasp", "brick": 4},
        {"picked_by_contact": True, "brick": 4},
        {"picked_by_contact": True, "brick": 5},
        {"picked_by_contact": True, "brick": 5},
    ]
    assert _contact_count(events) == 2
