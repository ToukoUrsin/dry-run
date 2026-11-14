"""Regression checks against the rigged one-brick loose benchmark."""
from __future__ import annotations

import ast
from pathlib import Path

from dryrun.env import TIERS
from dryrun.mjcf_station import _brick, layout_bricks, BrickSpec


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
