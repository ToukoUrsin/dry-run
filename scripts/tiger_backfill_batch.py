"""Run one uniquely tagged full-rate held-out telemetry batch off-box."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dryrun import telemetry
from dryrun.benchmark import _one


def main() -> None:
    tier = os.environ["BACKFILL_TIER"]
    start = int(os.environ["BACKFILL_START"])
    stop = int(os.environ["BACKFILL_STOP"])
    workers = int(os.environ.get("BACKFILL_WORKERS", "4"))
    max_steps = int(os.environ.get("BACKFILL_MAX_STEPS", "1800"))
    seeds = list(range(start, stop))
    started = time.perf_counter()
    telemetry.setup()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _one,
                seed,
                tier,
                Path("/tmp/dryrun-tiger-backfill"),
                None,
                None,
                "expert",
                max_steps,
            )
            for seed in seeds
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append({key: row[key] for key in ("seed", "tier", "bricks", "successes", "max_steps", "telemetry")})
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    telemetry.refresh()
    summary = {
        "tier": tier,
        "seed_start": start,
        "seed_stop_exclusive": stop,
        "seeds": len(rows),
        "bricks": sum(int(row["bricks"]) for row in rows),
        "successes": sum(int(row["successes"]) for row in rows),
        "seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
