"""Command-line curriculum comparison for M5.

The evaluator is injectable so the policy session can train two batches from
the same checkpoint.  ``--smoke`` proves the machinery on the expert tuning
set when no policy checkpoint exists.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .curriculum import random_specs
from .nemotron import compare_curriculum


def _smoke_score(batch: Sequence[Mapping[str, Any]], checkpoint: str) -> Mapping[str, Any]:
    # A transparent tuning-set proxy, never presented as held-out policy data.
    weights = {"loose": 1.0, "messy": .75, "pile": .5}
    return {"evaluator": "expert_tuning_smoke", "checkpoint": checkpoint,
            "score": round(sum(weights.get(s.get("tier"), 0) for s in batch) / len(batch), 6),
            "scenes": len(batch)}


def score(seeds: Sequence[int], evaluator: str = "expert_tuning_smoke") -> dict[str, Any]:
    """Return a deterministic score row for compatibility with earlier scripts."""
    specs = [{"tier": "loose" if int(s) % 3 == 0 else "messy" if int(s) % 3 == 1 else "pile", "brick_count": 3 + int(s) % 10, "spacing": .02, "yaw_spread": 1.57, "positions": [[.2, 0]], "lighting": "key light"} for s in seeds]
    return dict(_smoke_score(specs, "expert-tuning-v0")) | {"seeds": list(seeds)}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--diagnoses", type=Path); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--seed", type=int, default=7); p.add_argument("--checkpoint", default="expert-tuning-v0"); p.add_argument("--output", type=Path, default=Path("/bulk2/runs/dryrun/m5/comparison.json")); p.add_argument("--smoke", action="store_true"); a = p.parse_args()
    if not a.smoke and not a.diagnoses: p.error("--diagnoses is required unless --smoke is used")
    if a.diagnoses:
        rows = [json.loads(x) for x in a.diagnoses.read_text().splitlines() if x.strip()]
        suggested = []
        for row in rows:
            d = row.get("diagnosis", row)
            suggested.extend(d.get("next_scenes", []))
        suggested = suggested[:a.batch_size]
        if len(suggested) < a.batch_size: p.error(f"diagnoses contain {len(suggested)} scenes; need {a.batch_size}")
    else:
        suggested = random_specs(a.batch_size, a.seed + 1)
    control = random_specs(len(suggested), a.seed)
    result = compare_curriculum(suggested, control, _smoke_score, a.checkpoint, a.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
