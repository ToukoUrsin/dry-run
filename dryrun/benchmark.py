"""Held-out benchmark over the shared seeded rollout API."""
from __future__ import annotations
import argparse, gzip, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
import imageio.v2 as imageio
from .inspector import Inspector
from .rollout import run as rollout_run
from .env import TIERS
from .mjcf_station import layout_bricks

TIER_COUNTS = {"loose": 80, "messy": 70, "pile": 50}

class _ReplayCapture:
    """Collect the public rollout observations without reading simulator state."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.thumbnail = None

    def __call__(self, row: dict[str, Any]) -> None:
        if "observation" not in row:
            self.events.append(_jsonable(row))
            return
        obs = row.get("observation", {})
        top = obs.get("top")
        if self.thumbnail is None and top is not None:
            self.thumbnail = top
        self.frames.append({
            "t": float(row.get("t", 0.0)),
            "joints": _jsonable(obs.get("joints", [])),
            "action": _jsonable(row.get("action", [])),
            "verdict": int(row.get("verdict", 0)),
        })


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _write_replay(export_dir: Path, row: dict[str, Any], capture: _ReplayCapture) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row['controller']}-{row['tier']}-{int(row['seed']):04d}"
    payload = {
        "format": "dryrun-rollout-v1",
        "controller": row["controller"],
        "seed": row["seed"],
        "tier": row["tier"],
        "commit": row["commit"],
        "scene": row["scene"],
        "frames": capture.frames,
        "events": capture.events + row.get("events", []),
        "outcome": {
            "bricks": row["bricks"],
            "correct": row["successes"],
            "score": row["score"],
        },
    }
    replay = export_dir / f"{stem}.json.gz"
    with gzip.open(replay, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(payload, fh, separators=(",", ":"))
    thumb = export_dir / f"{stem}-top.png"
    if capture.thumbnail is not None:
        imageio.imwrite(thumb, capture.thumbnail)
    row["replay"] = str(replay)
    row["thumbnail"] = str(thumb) if capture.thumbnail is not None else None


def _one(seed: int, tier: str, workdir: Path, inspector=None, export_dir: Path | None = None, controller: str = "expert"):
    # Keep the benchmark on the same public rollout boundary used by policy
    # evaluation and dataset recording.  In particular, this prevents a
    # benchmark-only scene/controller path from drifting back to the
    # withdrawn fixed-position loose fixture.
    del workdir
    capture = _ReplayCapture() if export_dir is not None else None
    r = rollout_run(int(seed), controller, inspector=inspector, tier=tier, record=capture)
    specs = layout_bricks(int(seed), TIERS[tier])
    row = {
        "tier": tier,
        "seed": int(seed),
        "bricks": r["bricks"],
        "successes": r["successes"],
        "score": r["score"],
        "picked_by_contact": sum(int(e.get("picked_by_contact", False)) for e in r.get("events", [])),
        "verdicts": sum(1 for e in r.get("events", []) if e.get("event") == "verdict"),
        "scene": [asdict(spec) for spec in specs],
        "events": r.get("events", []),
        "controller": controller,
        "commit": _commit(),
    }
    if export_dir is not None:
        _write_replay(export_dir, row, capture)
    return row

def summarize(rows):
    bricks = sum(r["bricks"] for r in rows); successes = sum(r["successes"] for r in rows)
    return {"seeds": len(rows), "bricks": bricks, "successes": successes, "score": successes / bricks if bricks else 0.0, "picked_by_contact": sum(r["picked_by_contact"] for r in rows), "verdicts": sum(r["verdicts"] for r in rows)}

def run(seeds=None, workdir=Path("/bulk3/runs/dryrun/benchmark"), inspector=None, tier=None,
        export_replays: Path | None = None, controller: str = "expert", workers: int = 1):
    started = time.perf_counter(); workdir.mkdir(parents=True, exist_ok=True)
    if seeds is None: seeds = range(200)
    seeds = list(seeds)
    groups = {tier: seeds[:TIER_COUNTS[tier]]} if tier else ({"loose": seeds[:80], "messy": seeds[80:150], "pile": seeds[150:200]} if len(seeds) == 200 else {"loose": seeds})
    jobs = [(seed, name) for name, values in groups.items() for seed in values]
    if workers > 1 and inspector is None and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_one, seed, name, workdir, None, export_replays, controller)
                       for seed, name in jobs]
            rows = [future.result() for future in futures]
    else:
        rows = [_one(seed, name, workdir, inspector=inspector, export_dir=export_replays, controller=controller)
                for seed, name in jobs]
    rows.sort(key=lambda row: (row["tier"], row["seed"]))
    tiers = {name: summarize([r for r in rows if r["tier"] == name]) for name in groups}
    total = summarize(rows)
    return {"benchmark": "mujoco-physics-v3", "tiers": tiers, **total, "wall_seconds": round(time.perf_counter() - started, 3), "rows": rows}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--controller", choices=("expert", "policy"), default="expert"); p.add_argument("--checkpoint", type=Path); p.add_argument("--seeds", type=int, default=200); p.add_argument("--tier", choices=sorted(TIER_COUNTS)); p.add_argument("--workers", type=int, default=8); p.add_argument("--max-steps", type=int, default=1800); p.add_argument("--output", type=Path, default=Path("/bulk3/runs/dryrun/benchmark/physics.json")); p.add_argument("--export-replays", type=Path, help="write one compact replay and top-camera thumbnail per rollout"); p.add_argument("--without-inspector", action="store_true"); a = p.parse_args()
    if a.controller == "policy":
        from .policy_benchmark import evaluate
        if a.checkpoint is None: p.error("--checkpoint is required with --controller policy")
        x = evaluate(a.checkpoint, a.seeds, a.workers, "policy", max_steps=a.max_steps)
        x.update({"benchmark": "mujoco-physics-v3", "controller": "policy", "command": "uv run python -m dryrun.benchmark --controller policy --checkpoint PATH --seeds 200", "max_steps": a.max_steps})
    else:
        inspector = None if a.without_inspector else Inspector(); x = run(range(a.seeds), workdir=a.output.parent, inspector=inspector, tier=a.tier, export_replays=a.export_replays, workers=a.workers)
        x["controller"] = "expert"
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(x, indent=2)); print(json.dumps({k: v for k, v in x.items() if k not in ("rows", "tiers")}, indent=2))

if __name__ == "__main__": main()
