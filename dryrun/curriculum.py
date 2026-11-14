"""Curriculum batches produced by the Nemotron analyst.

Scene specifications are intentionally kept independent from the policy
trainer.  The policy session can pass the same two lists to
``compare_batches`` and train both from one named checkpoint.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .nemotron import SCENE_FIELDS, compare_curriculum, validate_diagnosis

def _generator_positions(value: Any) -> list[list[float]]:
    """Convert camera-pixel coordinates to the generator's metre envelope."""
    if not isinstance(value, list) or not value:
        raise ValueError("positions must be a non-empty list")
    out = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("each position needs x and y")
        x, y = float(point[0]), float(point[1])
        if max(abs(x), abs(y)) > 1.0:
            x, y = 0.12 + 0.36 * x / 1000.0, y / 1000.0 - 0.5
        if not 0.12 <= x <= 0.52 or not -0.30 <= y <= 0.30:
            raise ValueError(f"position outside generator reach envelope: {(x, y)}")
        out.append([round(x, 4), round(y, 4)])
    return out


def scene_specs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract and validate all Nemotron-generated scene specs."""
    out: list[dict[str, Any]] = []
    for record in records:
        diagnosis = record.get("diagnosis", record.get("analysis", record))
        validate_diagnosis(diagnosis)
        for scene in diagnosis["next_scenes"]:
            row = {k: scene[k] for k in SCENE_FIELDS}
            row["positions"] = _generator_positions(row["positions"])
            out.append(row)
    return out


def random_specs(size: int, seed: int = 7) -> list[dict[str, Any]]:
    """Make a same-size random control batch with generator-compatible fields."""
    if size < 1: raise ValueError("size must be positive")
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        n = rng.randint(3, 12)
        rows.append({"tier": rng.choice(["loose", "messy", "pile"]), "brick_count": n,
                     "spacing": round(rng.uniform(.004, .08), 4), "yaw_spread": round(rng.uniform(0, 3.142), 4),
                     "positions": [[round(rng.uniform(.16, .36), 4), round(rng.uniform(-.15, .15), 4)] for _ in range(n)],
                     "lighting": rng.choice(["key light", "overhead", "low contrast", "side light"])})
    return rows


def load_scenes(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL diagnosis records and return accepted next scenes."""
    p = Path(path)
    if p.suffix == ".jsonl": records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else: records = json.loads(p.read_text())
    if isinstance(records, Mapping): records = [records]
    return scene_specs(records)


def compare_batches(suggested: Sequence[Mapping[str, Any]], random_batch: Sequence[Mapping[str, Any]],
                    checkpoint: str, train_and_score: Callable[[Sequence[Mapping[str, Any]], str], Mapping[str, Any]],
                    output: str | Path | None = None) -> dict[str, Any]:
    """Run both batches through the policy/expert tuning hook."""
    return compare_curriculum(suggested, random_batch, train_and_score, checkpoint, output)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--diagnoses", type=Path, required=True); p.add_argument("--size", type=int, default=16); p.add_argument("--seed", type=int, default=7); p.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m5/curriculum.json")); a = p.parse_args()
    suggested = load_scenes(a.diagnoses)
    suggested = suggested[:a.size]
    if len(suggested) != a.size: raise SystemExit(f"need {a.size} generated scenes, found {len(suggested)}")
    random_batch = random_specs(a.size, a.seed)
    # The policy session replaces this hook.  This smoke evaluator is a stable
    # expert tuning score, explicitly labelled so it cannot be mistaken for a
    # held-out policy result.
    def expert_tuning(batch: Sequence[Mapping[str, Any]], checkpoint: str) -> Mapping[str, Any]:
        score = sum(1.0 if s["tier"] == "loose" else .75 if s["tier"] == "messy" else .5 for s in batch) / len(batch)
        return {"evaluator": "expert_tuning_smoke", "checkpoint": checkpoint, "score": round(score, 6), "scenes": len(batch)}
    print(json.dumps(compare_batches(suggested, random_batch, "expert-tuning-v0", expert_tuning, a.out), indent=2))


if __name__ == "__main__": main()
