"""Sentence-to-station generation and validation.

A sentence is compiled by :func:`dryrun.mjcf_station.station_xml`; this module
runs the same acceptance gate used by the page session and keeps every rejected
station with a machine-readable reason.  Accepted stations can be sent through
the normal contact-physics expert API (``dryrun.render.rollout``).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mujoco

from .mjcf_station import ROOT, check_station, layout_bricks, rejection_reason, station_xml

# Four spatially meaningful phrasings for each of the three layout/bowl/camera
# choices produce 36 distinct natural-language station requests.  The parser
# intentionally remains conservative: unsupported wording is still recorded as
# a rejected request rather than silently changing the requested station.
SENTENCES: tuple[str, ...] = tuple(
    f"{layout}, good and defective bowls {bowl}, inspection camera {camera}"
    for layout in ("a loose grid of bricks", "a regular tray of bricks", "a messy pile of bricks")
    for bowl in ("behind the arm", "in front of the arm", "on the sides")
    for camera in ("on the left", "on the right", "in front", "at the front")
)

# The sentence parser recognizes ``tray``, ``pile``/``messy``, and left/right.
# Keep a stable subset of camera wording that maps to the same physical pose.
# ``at the front`` intentionally exercises the default/front branch.

@dataclass(frozen=True)
class StationAttempt:
    seed: int
    sentence: str
    xml: str
    accepted: bool
    reason: str | None
    check: dict
    expert: dict | None = None


def _robot_xml() -> str:
    return str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve())


def _write_xml(path: Path, seed: int, sentence: str, count: int | None) -> None:
    path.write_text(
        station_xml(seed, sentence, count=count, robot_include=_robot_xml()),
        encoding="utf-8",
    )


def run_expert(xml_path: Path, seed: int, sentence: str, count: int | None = None) -> dict:
    """Run an accepted station through the shared MuJoCo rollout API.

    This is deliberately imported lazily: validation can run on machines that
    do not have a renderer, while ``--expert`` proves the integration in the
    same environment used by the benchmark.
    """
    from .render import rollout

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    specs = layout_bricks(seed, sentence, count)
    started = time.perf_counter()
    result = rollout(model, data, seed, specs=specs)
    result = {
        "seed": int(seed),
        "bricks": int(result.get("bricks", 0)),
        "successes": int(result.get("successes", 0)),
        "score": float(result.get("score", 0.0)),
        "events": len(result.get("events", [])),
        "seconds": round(time.perf_counter() - started, 3),
    }
    return result


def generate_and_check(
    sentences: Sequence[str] = SENTENCES,
    *,
    out_dir: str | Path = "/bulk2/runs/dryrun/m5/stations",
    record: str | Path | None = "/bulk2/runs/dryrun/m5/stations.jsonl",
    seed_start: int = 9000,
    count: int | None = 3,
    expert: bool = False,
) -> list[StationAttempt]:
    """Generate, validate, retain and optionally roll out sentence stations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[StationAttempt] = []
    for offset, sentence in enumerate(sentences):
        seed = seed_start + offset
        xml_path = out / f"station_{offset:03d}.xml"
        check: dict
        accepted = False
        reason: str | None = None
        expert_result: dict | None = None
        try:
            _write_xml(xml_path, seed, sentence, count)
            check = check_station(xml_path)
            accepted = bool(check.get("pass", False))
            reason = rejection_reason(check)
            if not accepted and reason is None:
                reason = "validation_failed"
            # Rejected XML is deliberately kept in the same output directory;
            # the JSONL line is the durable reason for audit/replay.
            if accepted and expert:
                expert_result = run_expert(xml_path, seed, sentence, count)
        except Exception as exc:  # retain a rejected request even on load errors
            check = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
            reason = check["error"]
            # Generation can fail before ``station_xml`` returns (for example,
            # when a requested brick count cannot be placed).  Keep an
            # inspectable artifact for that rejection as well as the JSONL
            # reason; this makes the rejected set auditable and replayable.
            if not xml_path.exists():
                xml_path.write_text(
                    "<!-- rejected station: " + reason.replace("--", "-") + " -->\n",
                    encoding="utf-8",
                )
        rows.append(StationAttempt(seed, sentence, str(xml_path), accepted, reason, check, expert_result))
    if record is not None:
        record_path = Path(record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.__dict__, separators=(",", ":")) + "\n")
    return rows


def summary(rows: Sequence[StationAttempt]) -> dict:
    accepted = [r for r in rows if r.accepted]
    rejected = [r for r in rows if not r.accepted]
    expert_rows = [r.expert for r in accepted if r.expert is not None]
    return {
        "attempted": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "pass_rate": len(accepted) / len(rows) if rows else 0.0,
        "expert_runs": len(expert_rows),
        "expert_bricks": sum(int(x.get("bricks", 0)) for x in expert_rows),
        "expert_successes": sum(int(x.get("successes", 0)) for x in expert_rows),
        "expert_score": (
            sum(int(x.get("successes", 0)) for x in expert_rows)
            / sum(int(x.get("bricks", 0)) for x in expert_rows)
            if sum(int(x.get("bricks", 0)) for x in expert_rows)
            else None
        ),
        "rejection_reasons": {
            reason: sum(1 for r in rejected if r.reason == reason)
            for reason in sorted({r.reason for r in rejected})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate stations from plain sentences")
    parser.add_argument("--out-dir", default="/bulk2/runs/dryrun/m5/stations")
    parser.add_argument("--record", default="/bulk2/runs/dryrun/m5/stations.jsonl")
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only for a smoke run; default is all 36 sentences")
    parser.add_argument("--expert", action="store_true", help="run accepted stations through render.rollout")
    args = parser.parse_args()
    sentences = SENTENCES if args.limit is None else SENTENCES[: max(0, args.limit)]
    rows = generate_and_check(
        sentences,
        out_dir=args.out_dir,
        record=args.record,
        seed_start=args.seed_start,
        count=args.count,
        expert=args.expert,
    )
    output = summary(rows)
    output["record"] = str(args.record)
    output["out_dir"] = str(args.out_dir)
    output["sentences"] = len(sentences)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
