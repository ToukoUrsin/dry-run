"""Small local worker for sentence-to-station requests.

The page treats a sentence as data.  This process is the only component that
turns a queued sentence into MJCF: it calls the deterministic Astra-authored
generator/checker and the same expert rollout API used by the benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import telemetry
from .stations import generate_and_check, summary


def run_once(*, url: str | None = None, out_dir: str | Path = "/bulk2/runs/dryrun/m5/worker") -> dict:
    name = "station-worker"
    telemetry.worker_heartbeat(name, "online", {"pid": __import__("os").getpid()}, url=url)
    claimed = telemetry.claim_station_request(url=url)
    if claimed is None:
        telemetry.worker_heartbeat(name, "idle", {}, url=url)
        return {"status": "idle"}
    request_id, sentence = claimed
    try:
        rows = generate_and_check([sentence], out_dir=Path(out_dir) / str(request_id), record=Path(out_dir) / f"{request_id}.jsonl", seed_start=100000 + request_id, count=3, expert=True)
        result = summary(rows)
        status = "accepted" if result["accepted"] else "rejected"
        telemetry.finish_station_request(request_id, status, result=result, url=url)
        telemetry.worker_heartbeat(name, "online", {"last_request": request_id, "last_status": status}, url=url)
        return {"request_id": request_id, "sentence": sentence, **result}
    except Exception as exc:
        telemetry.finish_station_request(request_id, "error", error=f"{type(exc).__name__}: {exc}", url=url)
        telemetry.worker_heartbeat(name, "error", {"request_id": request_id, "error": str(exc)}, url=url)
        return {"request_id": request_id, "status": "error", "error": str(exc)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--url", default=None)
    p.add_argument("--out-dir", default="/bulk2/runs/dryrun/m5/worker")
    args = p.parse_args()
    while True:
        print(json.dumps(run_once(url=args.url, out_dir=args.out_dir), sort_keys=True), flush=True)
        if args.once:
            break
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    main()
