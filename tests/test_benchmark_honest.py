"""Regression checks against the rigged one-brick loose benchmark."""
from __future__ import annotations

import ast
from pathlib import Path

from dryrun.env import TIERS
from dryrun.mjcf_station import layout_bricks


ROOT = Path(__file__).resolve().parents[1]


def test_loose_seed_controls_a_real_multi_brick_scene():
    scenes = [layout_bricks(seed, TIERS["loose"]) for seed in range(20)]
    assert all(3 <= len(scene) <= 8 for scene in scenes)
    signatures = {tuple((b.size, b.x, b.y, b.color, b.defect) for b in scene) for scene in scenes}
    assert len(signatures) >= 18, "loose seeds must not repeat one fixed scene"
    assert len({len(scene) for scene in scenes}) >= 2, "seed must affect brick count"


def test_benchmark_has_no_position_override_or_one_brick_special_case():
    source = (ROOT / "dryrun" / "benchmark.py").read_text()
    tree = ast.parse(source)
    assert "re.sub" not in source
    assert "count=1" not in source
    assert "0.2600" not in source
    assert not any(isinstance(node, ast.Import) and any(alias.name == "re" for alias in node.names) for node in tree.body)
