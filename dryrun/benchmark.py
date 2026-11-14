"""Held-out benchmark over the shared seeded rollout API."""
from __future__ import annotations
import argparse, json, time
from dataclasses import asdict
from pathlib import Path
from .inspector import Inspector
from .rollout import run as rollout_run
from .env import TIERS
from .mjcf_station import layout_bricks

TIER_COUNTS = {"loose": 80, "messy": 70, "pile": 50}

def _one(seed: int, tier: str, workdir: Path, inspector=None):
    # Keep the benchmark on the same public rollout boundary used by policy
    # evaluation and dataset recording.  In particular, this prevents a
    # benchmark-only scene/controller path from drifting back to the
    # withdrawn fixed-position loose fixture.
    del workdir
    r = rollout_run(int(seed), "expert", inspector=inspector, tier=tier)
    specs = layout_bricks(int(seed), TIERS[tier])
    return {
        "tier": tier,
        "seed": int(seed),
        "bricks": r["bricks"],
        "successes": r["successes"],
        "score": r["score"],
        "picked_by_contact": sum(int(e.get("picked_by_contact", False)) for e in r.get("events", [])),
        "verdicts": sum(1 for e in r.get("events", []) if e.get("event") == "verdict"),
        "scene": [asdict(spec) for spec in specs],
        "events": r.get("events", []),
    }

def summarize(rows):
    bricks = sum(r["bricks"] for r in rows); successes = sum(r["successes"] for r in rows)
    return {"seeds": len(rows), "bricks": bricks, "successes": successes, "score": successes / bricks if bricks else 0.0, "picked_by_contact": sum(r["picked_by_contact"] for r in rows), "verdicts": sum(r["verdicts"] for r in rows)}

def run(seeds=None, workdir=Path("/bulk3/runs/dryrun/benchmark"), inspector=None, tier=None):
    started = time.perf_counter(); workdir.mkdir(parents=True, exist_ok=True)
    if seeds is None: seeds = range(200)
    seeds = list(seeds)
    groups = {tier: seeds[:TIER_COUNTS[tier]]} if tier else ({"loose": seeds[:80], "messy": seeds[80:150], "pile": seeds[150:200]} if len(seeds) == 200 else {"loose": seeds})
    rows = [_one(seed, name, workdir, inspector=inspector) for name, values in groups.items() for seed in values]
    tiers = {name: summarize([r for r in rows if r["tier"] == name]) for name in groups}
    total = summarize(rows)
    return {"benchmark": "mujoco-physics-v3", "tiers": tiers, **total, "wall_seconds": round(time.perf_counter() - started, 3), "rows": rows}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--controller", choices=("expert", "policy"), default="expert"); p.add_argument("--checkpoint", type=Path); p.add_argument("--seeds", type=int, default=200); p.add_argument("--tier", choices=sorted(TIER_COUNTS)); p.add_argument("--workers", type=int, default=8); p.add_argument("--max-steps", type=int, default=1800); p.add_argument("--output", type=Path, default=Path("/bulk3/runs/dryrun/benchmark/physics.json")); p.add_argument("--without-inspector", action="store_true"); a = p.parse_args()
    if a.controller == "policy":
        from .policy_benchmark import evaluate
        if a.checkpoint is None: p.error("--checkpoint is required with --controller policy")
        x = evaluate(a.checkpoint, a.seeds, a.workers, "policy", max_steps=a.max_steps)
        x.update({"benchmark": "mujoco-physics-v3", "controller": "policy", "command": "uv run python -m dryrun.benchmark --controller policy --checkpoint PATH --seeds 200", "max_steps": a.max_steps})
    else:
        inspector = None if a.without_inspector else Inspector(); x = run(range(a.seeds), workdir=a.output.parent, inspector=inspector, tier=a.tier)
        x["controller"] = "expert"
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(x, indent=2)); print(json.dumps({k: v for k, v in x.items() if k not in ("rows", "tiers")}, indent=2))

if __name__ == "__main__": main()
