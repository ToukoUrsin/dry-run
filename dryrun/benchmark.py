"""Held-out MuJoCo benchmark with fixed loose, messy and pile tiers."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import mujoco
from .env import TIERS
from .mjcf_station import station_xml, ROOT, layout_bricks
from .render import rollout
from .inspector import Inspector

TIER_COUNTS = {"loose": 80, "messy": 70, "pile": 50}

def _one(seed: int, tier: str, workdir: Path, inspector=None):
    sentence = TIERS[tier]; xml = workdir / f"{tier}_{int(seed):04d}.xml"
    xml.write_text(station_xml(int(seed), sentence, robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve())))
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    specs = layout_bricks(int(seed), sentence)
    r = rollout(model, data, int(seed), writer=None, specs=specs, inspector=inspector)
    return {"tier": tier, "seed": int(seed), "bricks": r["bricks"], "successes": r["successes"], "score": r["score"], "picked_by_contact": sum(int(e.get("picked_by_contact", False)) for e in r["events"]), "verdicts": sum(1 for e in r["events"] if e.get("event") == "verdict")}

def summarize(rows):
    bricks = sum(r["bricks"] for r in rows); successes = sum(r["successes"] for r in rows)
    return {"seeds": len(rows), "bricks": bricks, "successes": successes, "score": successes / bricks if bricks else 0.0, "picked_by_contact": sum(r["picked_by_contact"] for r in rows), "verdicts": sum(r["verdicts"] for r in rows)}

def run(seeds=None, workdir=Path("/bulk3/runs/dryrun/benchmark"), inspector=None, tier=None):
    started = time.perf_counter(); workdir.mkdir(parents=True, exist_ok=True)
    if seeds is None: seeds = range(200)
    seeds = list(seeds)
    groups = {tier: seeds} if tier else ({"loose": seeds[:80], "messy": seeds[80:150], "pile": seeds[150:200]} if len(seeds) == 200 else {"loose": seeds})
    rows = [_one(seed, name, workdir, inspector=inspector) for name, values in groups.items() for seed in values]
    tiers = {name: summarize([r for r in rows if r["tier"] == name]) for name in groups}
    total = summarize(rows)
    return {"benchmark": "mujoco-physics-v3", "tiers": tiers, **total, "wall_seconds": round(time.perf_counter() - started, 3), "rows": rows}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, default=200); p.add_argument("--tier", choices=sorted(TIER_COUNTS)); p.add_argument("--output", type=Path, default=Path("/bulk3/runs/dryrun/benchmark/physics.json")); p.add_argument("--without-inspector", action="store_true"); a = p.parse_args()
    inspector = None if a.without_inspector else Inspector(); x = run(range(a.seeds), workdir=a.output.parent, inspector=inspector, tier=a.tier); a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(x, indent=2)); print(json.dumps({k: v for k, v in x.items() if k != "rows"}, indent=2))

if __name__ == "__main__": main()
