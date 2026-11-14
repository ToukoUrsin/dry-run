"""Record successful expert brick cycles in current LeRobot format.

This entry point is deliberately a thin adapter around the public rollout
API.  The recorder owns episode segmentation; after each scene we pass the
rollout result back to it so only bricks listed in ``result["correct"]`` are
committed.  This matters for a multi-brick scene: a scene-level success flag
must never turn every frame in that scene into a training episode.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from .demos import DemoRecorder
from .rollout import run
from .inspector import Inspector


RUN_ROOT = Path("/bulk3/runs/dryrun")


def validate_dataset_request(tier: str, count: int | None) -> None:
    """Reject the withdrawn one-brick loose scene before any rollout starts.

    ``count`` remains part of :func:`dryrun.rollout.run` for calibration and
    non-benchmark callers.  Training defaults to the seeded tier layout, and a
    caller must never accidentally turn loose training into the fixed scene
    that prompts/016 withdrew.
    """
    if tier == "loose" and count == 1:
        raise ValueError(
            "refusing count=1 for loose training: the withdrawn fixed-position "
            "loose scene is not a dataset source; omit --count for seeded scenes"
        )
    if count is not None and count < 1:
        raise ValueError("count must be positive when supplied")


def validate_output_path(path: str | Path) -> Path:
    """Require generated binaries and metadata to live under the new run root."""

    out = Path(path).expanduser()
    try:
        out.resolve().relative_to(RUN_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"dataset output must be under {RUN_ROOT}: {out}") from exc
    if out.exists():
        raise FileExistsError(
            f"dataset output already exists: {out}; choose a new run directory"
        )
    return out


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _finalize_scene(recorder: DemoRecorder, result: dict) -> None:
    """Finalize one scene with the recorder version currently installed.

    ``finalize_result`` is the multi-brick API.  The fallback keeps this
    command usable with an older checkout while making the compatibility
    boundary explicit; it is not used by the current recorder.
    """

    finalize = getattr(recorder, "finalize_result", None)
    if callable(finalize):
        finalize(result)
        return
    # Older recorder versions represented the whole scene as one cycle.  Keep
    # this only as a compatibility path for local smoke runs; current data
    # generation must use the per-brick finalize_result implementation.
    recorder.cycle_end(bool(result.get("correct")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=RUN_ROOT / "m4" / "lerobot-real-current",
        help="new dataset directory (must be under /bulk3/runs/dryrun)",
    )
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--tier", choices=("loose", "messy", "pile"), default="loose")
    ap.add_argument(
        "--count",
        type=int,
        default=None,
        help="optional calibration brick count; loose training rejects 1",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=3600,
        help="per-scene rollout horizon; seven loose bricks need about 2150 ticks",
    )
    ap.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="stop after this many completed cycles (0 means all bricks)",
    )
    ap.add_argument("--inspector-model", type=Path, default=None)
    ap.add_argument("--inspector-threads", type=int, default=1)
    ap.add_argument(
        "--without-inspector",
        action="store_true",
        help="infrastructure smoke only; verdict feature remains zero",
    )
    a = ap.parse_args()
    try:
        validate_dataset_request(a.tier, a.count)
    except ValueError as exc:
        ap.error(str(exc))
    if a.seeds < 1:
        ap.error("--seeds must be positive")
    if a.max_steps < 1:
        ap.error("--max-steps must be positive")
    if a.max_cycles < 0:
        ap.error("--max-cycles must be non-negative")
    try:
        out = validate_output_path(a.out)
    except (ValueError, FileExistsError) as exc:
        ap.error(str(exc))

    recorder = DemoRecorder(out)
    inspector = (
        None
        if a.without_inspector
        else Inspector(a.inspector_model, threads=a.inspector_threads)
    )
    results = []
    for seed in range(a.start_seed, a.start_seed + a.seeds):
        result = run(
            seed,
            "expert",
            inspector=inspector,
            record=recorder,
            tier=a.tier,
            count=a.count,
            max_steps=a.max_steps,
            max_cycles=(a.max_cycles or None),
        )
        _finalize_scene(recorder, result)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    summary = recorder.close()
    payload = {
        "format": "dryrun-dataset-summary-v2",
        "dataset": str(out),
        "tier": a.tier,
        "count": a.count,
        "requested_seeds": a.seeds,
        "start_seed": a.start_seed,
        "max_steps": a.max_steps,
        "max_cycles": a.max_cycles or None,
        "inspector": None if a.without_inspector else str(a.inspector_model or "env/default"),
        "source_commit": _source_commit(),
        "command": " ".join(__import__("sys").argv),
        "results": results,
        **summary,
    }
    summary_path = out.parent / f"{out.name}-summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
