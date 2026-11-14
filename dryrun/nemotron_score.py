"""Reproducible Nemotron scoring command used by status/M5.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .nemotron import labelled_set, score_diagnoses


def _read(path: Path) -> list[dict[str, Any]]:
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(path.read_text())
    if isinstance(raw, dict): raw = raw.get("events", raw.get("rows", raw.get("predictions", [])))
    return list(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=Path, required=True, help="real rollout event log, JSON or JSONL")
    p.add_argument("--predictions", type=Path, required=True, help="saved Nemotron predictions, JSON or JSONL")
    p.add_argument("--labels-out", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--minimum", type=int, default=300)
    a = p.parse_args()
    labels = labelled_set(_read(a.events), minimum=a.minimum)
    predictions = _read(a.predictions)
    result = score_diagnoses(labels, predictions)
    result["labels"] = len(labels); result["events"] = str(a.events); result["predictions"] = str(a.predictions)
    if a.labels_out:
        a.labels_out.parent.mkdir(parents=True, exist_ok=True); a.labels_out.write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in labels) + "\n")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
