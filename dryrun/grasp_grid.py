"""Parallel SO-101 single-brick reach and friction-grasp map."""
from __future__ import annotations
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco
from .grasp_test import make_empty_table, trial

_MODEL = None
_BASE = None

def _init(xml: str):
    global _MODEL, _BASE
    _MODEL = mujoco.MjModel.from_xml_path(xml)
    _BASE = mujoco.MjData(_MODEL)

def _one(args):
    x, y, yaw = args
    row = trial(_MODEL, _BASE, (x, y), yaw)
    row["pass"] = bool(row.get("lifted_10mm"))
    row["reachable"] = row.get("failed_stage") not in {"approach", "lower_approach", "recenter", "lift_ik"}
    return row

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="/bulk2/runs/dryrun/m1/reach_map.xml")
    p.add_argument("--out", default="/bulk2/runs/dryrun/m1/reach_map.jsonl")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    xml = Path(a.xml); xml.parent.mkdir(parents=True, exist_ok=True); make_empty_table(xml)
    poses = [(round(x, 3), round(y, 3), round(yaw, 4))
             for x in [0.12 + i * 0.024 for i in range(15)]
             for y in [-0.22 + i * 0.032 for i in range(15)]
             for yaw in [i * 3.141592653589793 / 4 for i in range(8)]]
    with ProcessPoolExecutor(max_workers=min(a.workers, 32), initializer=_init, initargs=(str(xml),)) as pool:
        rows = list(pool.map(_one, poses, chunksize=1))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    print(json.dumps({"tested": len(rows), "workers": min(a.workers, 32), "reachable": sum(r["reachable"] for r in rows), "passes": sum(r["pass"] for r in rows), "out": str(out)}, indent=2))

if __name__ == "__main__": main()
