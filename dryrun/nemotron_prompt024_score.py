"""Score strict Prompt 024 diagnoses against simulator event labels.

The analyst receives only the 448 px sheet.  This command joins its saved
responses to the event-derived labels after the provider run and writes an
ID-aligned prediction file plus stage accuracy and confusion data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .nemotron_real import STAGES, validate


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: row is not an object")
            yield value


def _predictions(root: Path) -> dict[str, dict[str, Any]]:
    """Read shard result files, retaining the highest-attempt valid response."""
    found: dict[str, dict[str, Any]] = {}
    paths = sorted(root.rglob("results.jsonl"))
    # The final merge stores flat shard files so the exact retained provider
    # rows can be copied without recreating worker directories.
    if not paths:
        paths = sorted(root.rglob("*.jsonl"))
    for path in paths:
        for row in _jsonl(path):
            rid = str(row.get("id", ""))
            if not rid:
                continue
            current = found.get(rid)
            if current is None or int(row.get("attempt", 0)) >= int(current.get("attempt", 0)):
                found[rid] = row
    return found


def score(labels_path: Path, predictions_root: Path, out_dir: Path) -> dict[str, Any]:
    labels = list(_jsonl(labels_path))
    raw = _predictions(predictions_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, Any]] = []
    matrix: dict[str, Counter[str]] = {stage: Counter() for stage in STAGES}
    per_stage: dict[str, dict[str, Any]] = {}
    tiers: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "unresolved": 0})
    correct = valid = unresolved = 0
    for label in labels:
        rid = str(label["id"])
        row = raw.get(rid, {})
        diagnosis = row.get("diagnosis") if isinstance(row, dict) else None
        pred: str | None = None
        error: str | None = None
        if isinstance(diagnosis, dict):
            try:
                pred = validate(diagnosis)["failure_stage"]
            except (TypeError, ValueError) as exc:
                error = f"invalid diagnosis: {exc}"
        else:
            error = str(row.get("error", "missing response"))
        actual = str(label["failure_stage"])
        if pred is None:
            unresolved += 1
            matrix[actual]["unresolved"] += 1
        else:
            valid += 1
            matrix[actual][pred] += 1
            if pred == actual:
                correct += 1
        tier = str(label.get("tier", "unknown"))
        tiers[tier]["n"] += 1
        if pred is None:
            tiers[tier]["unresolved"] += 1
        elif pred == actual:
            tiers[tier]["correct"] += 1
        merged.append({"id": rid, "label": label, "prediction": diagnosis if pred else None,
                       "predicted_stage": pred, "error": error})
    for stage in STAGES:
        n = sum(matrix[stage].values())
        c = matrix[stage][stage]
        per_stage[stage] = {"n": n, "correct": c, "accuracy": round(c / n, 6) if n else None,
                            "predicted": dict(matrix[stage])}
    score_data = {
        "schema": "prompt024-448-sheet-v1",
        "n": len(labels),
        "valid_n": valid,
        "unresolved": unresolved,
        "correct": correct,
        "stage_accuracy": round(correct / len(labels), 6) if labels else 0.0,
        "valid_stage_accuracy": round(correct / valid, 6) if valid else 0.0,
        "stages": list(STAGES),
        "per_stage": per_stage,
        "confusion": {stage: {column: matrix[stage][column] for column in (*STAGES, "unresolved")}
                      for stage in STAGES},
        "tiers": dict(tiers),
        "labels": str(labels_path),
        "predictions_root": str(predictions_root),
    }
    (out_dir / "predictions.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in merged))
    (out_dir / "score.json").write_text(json.dumps(score_data, indent=2) + "\n")
    return score_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.labels, args.predictions_root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
