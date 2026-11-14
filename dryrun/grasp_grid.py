"""Parallel polar SO-101 pick/lift reach calibration.

Each row is a real MuJoCo physics trial. Closed-form IK is used by the trial
as a fast reach cross-check; the ``pass`` field means both pads contacted and
the brick rose at least 10 mm.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco

from .closed_form_ik import SHOULDER
from .grasp_test import make_empty_table, trial

_MODEL = None
_BASE = None
_SIZE = "2x2"


def _init(xml: str, size: str):
    global _MODEL, _BASE, _SIZE
    _SIZE = size
    _MODEL = mujoco.MjModel.from_xml_path(xml)
    _BASE = mujoco.MjData(_MODEL)


def _one(args):
    x, y, yaw = args
    row = trial(_MODEL, _BASE, (x, y), yaw)
    row.update({
        "size": _SIZE,
        "radius_m": round(math.hypot(x - float(SHOULDER[0]), y - float(SHOULDER[1])), 6),
        "theta_rad": round(math.atan2(y - float(SHOULDER[1]), x - float(SHOULDER[0])), 6),
    })
    row["pass"] = bool(row.get("lifted_10mm"))
    row["reachable"] = row.get("failed_stage") not in {"approach", "lower_approach", "recenter", "lift_ik"}
    return row


def _size_xml(path: Path, size: str) -> None:
    if size != "2x4":
        return
    tree = ET.parse(path)
    body = tree.getroot().find(".//body[@name='brick_0']")
    if body is None:
        raise RuntimeError("calibration station has no brick_0")
    geom = body.find("geom[@name='brick_0_body']")
    if geom is None:
        raise RuntimeError("calibration station has no brick body geom")
    geom.set("size", "0.0320 0.0160 0.00960")
    tree.write(path, encoding="unicode")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="/bulk2/runs/dryrun/m1/reach_map.xml")
    p.add_argument("--out", default="/bulk2/runs/dryrun/m1/reach_map_polar.jsonl")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--radii", type=int, default=15, help="number of radii from 0.08 to 0.36 m")
    p.add_argument("--angles", type=int, default=32, help="number of full-pan angular cells")
    p.add_argument("--sizes", nargs="+", choices=("2x2", "2x4"), default=("2x2", "2x4"))
    a = p.parse_args()
    xml = Path(a.xml)
    xml.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for size in a.sizes:
        make_empty_table(xml)
        _size_xml(xml, size)
        poses = []
        for radius in [0.08 + i * (0.36 - 0.08) / max(1, a.radii - 1) for i in range(a.radii)]:
            for theta in [-1.919862 + i * (3.839724) / max(1, a.angles - 1) for i in range(a.angles)]:
                x = float(SHOULDER[0] + radius * math.cos(theta))
                y = float(SHOULDER[1] + radius * math.sin(theta))
                for yi in range(8):
                    poses.append((round(x, 5), round(y, 5), round(yi * math.pi / 4, 6)))
        with ProcessPoolExecutor(max_workers=min(a.workers, 32), initializer=_init, initargs=(str(xml), size)) as pool:
            rows = list(pool.map(_one, poses, chunksize=1))
        all_rows.extend(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in all_rows))
    print(json.dumps({
        "tested": len(all_rows),
        "workers": min(a.workers, 32),
        "sizes": list(a.sizes),
        "radii": a.radii,
        "angles": a.angles,
        "reachable": sum(r["reachable"] for r in all_rows),
        "passes": sum(r["pass"] for r in all_rows),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
