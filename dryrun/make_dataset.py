"""Record successful single-brick expert cycles in current LeRobot format."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from .demos import DemoRecorder
from .rollout import run
from .inspector import Inspector


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m4/lerobot-real")); ap.add_argument("--seeds", type=int, default=12); ap.add_argument("--start-seed", type=int, default=0); ap.add_argument("--tier", choices=("loose", "messy", "pile"), default="loose"); ap.add_argument("--count", type=int, default=None, help="optional calibration brick count; loose training rejects 1"); ap.add_argument("--without-inspector", action="store_true"); a = ap.parse_args()
    try:
        validate_dataset_request(a.tier, a.count)
    except ValueError as exc:
        ap.error(str(exc))
    if a.out.exists(): shutil.rmtree(a.out)
    recorder = DemoRecorder(a.out); inspector = None if a.without_inspector else Inspector()
    results = []
    for seed in range(a.start_seed, a.start_seed + a.seeds):
        result = run(seed, "expert", inspector=inspector, record=recorder, tier=a.tier, count=a.count); results.append(result); print(json.dumps(result), flush=True)
    summary = recorder.close(); payload = {"dataset": str(a.out), "tier": a.tier, "count": a.count, "requested_seeds": a.seeds, "results": results, **summary}; (a.out.parent / "dataset-summary.json").write_text(json.dumps(payload, indent=2)); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
