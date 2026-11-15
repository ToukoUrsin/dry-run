"""Run the Prompt 024 analyst with a paced set of independent workers.

The NVIDIA endpoint is slow enough that one sequential client does not reach
the requested throughput. Workers are staggered and each has its own pacing
clock; the aggregate schedule is bounded by ``workers / interval`` requests
per second. Each shard retains the request, raw response, and terminal result
under the caller supplied bulk output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .nemotron_real import Client


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _worker(index: int, rows: list[dict[str, Any]], out: Path, interval: float, retries: int, stagger: float) -> dict[str, Any]:
    if stagger > 0:
        time.sleep(index * stagger)
    shard = out / f"shard-{index:02d}"
    client = Client(shard / "requests", interval=interval, retries=retries)
    result_path = shard / "results.jsonl"
    done: set[str] = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "diagnosis" in row:
                    done.add(str(row["id"]))
    shard.mkdir(parents=True, exist_ok=True)
    valid = errors = 0
    with result_path.open("a") as handle:
        for row in rows:
            rid = str(row["id"])
            if rid in done:
                valid += 1
                continue
            result = client.analyse(row)
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            handle.flush()
            if "diagnosis" in result:
                valid += 1
            else:
                errors += 1
            print(json.dumps({"shard": index, "id": rid, "ok": "diagnosis" in result}), flush=True)
    return {"shard": index, "rows": len(rows), "valid": valid, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--interval", type=float, default=24.0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.interval <= 0 or args.retries < 0:
        parser.error("workers, interval, and retries must be positive")
    rows = _read(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    stagger = args.interval / args.workers
    manifest = {
        "input": str(args.input),
        "input_rows": len(rows),
        "workers": args.workers,
        "interval_per_worker_s": args.interval,
        "aggregate_rate_bound_per_min": round(60 * args.workers / args.interval, 6),
        "stagger_s": stagger,
        "retries": args.retries,
        "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "source_commit": os.popen("git rev-parse HEAD").read().strip(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assignments = [rows[i::args.workers] for i in range(args.workers)]
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, i, assignments[i], args.out, args.interval, args.retries, stagger) for i in range(args.workers)]
        for future in as_completed(futures):
            summaries.append(future.result())
    summaries.sort(key=lambda row: row["shard"])
    (args.out / "workers.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"input_rows": len(rows), "workers": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
