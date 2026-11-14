"""Record successful single-brick expert cycles in current LeRobot format."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from .demos import DemoRecorder
from .rollout import run
from .inspector import Inspector


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m4/lerobot-real")); ap.add_argument("--seeds", type=int, default=12); ap.add_argument("--start-seed", type=int, default=0); ap.add_argument("--tier", choices=("loose", "messy", "pile"), default="loose"); ap.add_argument("--without-inspector", action="store_true"); a = ap.parse_args()
    if a.out.exists(): shutil.rmtree(a.out)
    recorder = DemoRecorder(a.out); inspector = None if a.without_inspector else Inspector()
    results = []
    for seed in range(a.start_seed, a.start_seed + a.seeds):
        result = run(seed, "expert", inspector=inspector, record=recorder, tier=a.tier, count=1); results.append(result); print(json.dumps(result), flush=True)
    summary = recorder.close(); payload = {"dataset": str(a.out), "tier": a.tier, "requested_seeds": a.seeds, "results": results, **summary}; (a.out.parent / "dataset-summary.json").write_text(json.dumps(payload, indent=2)); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
