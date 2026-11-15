"""Modal check for where the final ACT controller stops.

This is prompt-040 diagnostic evidence, not a benchmark.  It uses only the
public rollout observation callback and the seeded layout geometry to measure
the TCP's distance to the nearest brick and to the scene centroid.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


app = modal.App("dryrun-policy-check-stop")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
    .pip_install(
        "mujoco", "numpy", "torch", "torchvision", "lerobot[dataset]==0.7.0",
        "opencv-python-headless", "pillow", "openvino",
    )
    .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
    .add_local_dir("assets", remote_path="/root/repo/assets")
    .add_local_file("data/reach_map_polar.json", remote_path="/root/repo/data/reach_map_polar.json")
    .add_local_file(
        "/bulk3/runs/dryrun/m2_rebuild/models_prompt020/PaDiM/openvino/weights/openvino/dryrun_inspector.xml",
        remote_path="/root/repo/model/dryrun_inspector.xml",
    )
    .add_local_file(
        "/bulk3/runs/dryrun/m2_rebuild/models_prompt020/PaDiM/openvino/weights/openvino/dryrun_inspector.bin",
        remote_path="/root/repo/model/dryrun_inspector.bin",
    )
    .add_local_file(
        "/bulk3/runs/dryrun/m2_rebuild/models_prompt020/PaDiM/openvino/weights/openvino/threshold.json",
        remote_path="/root/repo/model/threshold.json",
    )
)


class StopRecorder:
    def __init__(self, seed: int, tier: str):
        import numpy as np
        from dryrun.env import TIERS
        from dryrun.mjcf_station import layout_bricks

        self.seed = int(seed)
        self.tier = str(tier)
        self._points = np.asarray(
            [
                [float(spec.x), float(spec.y), float(getattr(spec, "z", 0.019))]
                for spec in layout_bricks(self.seed, TIERS[self.tier])
            ],
            dtype=np.float64,
        )
        self._centroid = self._points.mean(axis=0)
        self._rows: list[dict] = []

    def record(self, payload: dict):
        observation = payload.get("observation")
        if not observation:
            return
        import numpy as np
        from dryrun.closed_form_ik import forward_position

        joints = np.asarray(observation.get("joints", []), dtype=np.float64)
        if joints.shape != (6,):
            return
        tcp = np.asarray(forward_position(joints[:5]), dtype=np.float64)
        nearest = np.linalg.norm(self._points - tcp[None, :], axis=1)
        nearest_i = int(np.argmin(nearest))
        self._rows.append(
            {
                "step": int(payload.get("step", len(self._rows))),
                "t": float(payload.get("t", 0.0)),
                "tcp_m": [float(v) for v in tcp],
                "nearest_mm": float(nearest[nearest_i] * 1000.0),
                "nearest_brick": nearest_i,
                "centroid_mm": float(np.linalg.norm(tcp - self._centroid) * 1000.0),
                "height_mm": float(tcp[2] * 1000.0),
            }
        )

    def summary(self) -> dict:
        if not self._rows:
            return {
                "seed": self.seed,
                "tier": self.tier,
                "frames": 0,
                "nearest": None,
                "centroid": None,
                "final": None,
            }
        nearest = min(self._rows, key=lambda row: row["nearest_mm"])
        centroid = min(self._rows, key=lambda row: row["centroid_mm"])
        final = self._rows[-1]
        return {
            "seed": self.seed,
            "tier": self.tier,
            "frames": len(self._rows),
            "nearest": {
                "mm": round(nearest["nearest_mm"], 3),
                "brick": nearest["nearest_brick"],
                "height_mm": round(nearest["height_mm"], 3),
                "step": nearest["step"],
            },
            "centroid": {
                "mm": round(centroid["centroid_mm"], 3),
                "height_mm": round(centroid["height_mm"], 3),
                "step": centroid["step"],
            },
            "final": {
                "nearest_mm": round(final["nearest_mm"], 3),
                "centroid_mm": round(final["centroid_mm"], 3),
                "height_mm": round(final["height_mm"], 3),
                "step": final["step"],
            },
        }


@app.function(image=image, timeout=1800, cpu=4, retries=1, volumes={"/runs": volume})
def check_seed(
    seed: int,
    tier: str = "loose",
    checkpoint: str = "/runs/act-hi224-c6d8196-20261115-1/act-final.pt",
    max_steps: int = 2600,
    run_id: str = "policy-stop-check",
    render_size: int = 224,
    open_loop_ticks: int = 30,
    max_delta: float = 0.06,
):
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import torch
    from dryrun.inspector import Inspector
    from dryrun.policy_benchmark import ACTController
    from dryrun.rollout import run

    torch.set_num_threads(1)
    recorder = StopRecorder(int(seed), str(tier))
    controller = ACTController(
        checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        open_loop_ticks=int(open_loop_ticks),
        max_delta=float(max_delta),
    )
    inspector = Inspector("/root/repo/model/dryrun_inspector.xml", threads=1)
    result = run(
        int(seed), controller, inspector=inspector, record=recorder,
        tier=str(tier), max_steps=int(max_steps), render_size=int(render_size),
    )
    summary = recorder.summary()
    summary.update(
        {
            "run_id": str(run_id),
            "checkpoint": str(checkpoint),
            "max_steps": int(max_steps),
            "render_size": int(render_size),
            "open_loop_ticks": int(open_loop_ticks),
            "max_delta": float(max_delta),
            "bricks": int(result.get("bricks", 0)),
            "successes": int(result.get("successes", 0)),
            "score": float(result.get("score", 0.0)),
            "events": result.get("events", []),
        }
    )
    return summary


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(
    seeds: str = ",".join(str(i) for i in range(20)),
    tier: str = "loose",
    checkpoint: str = "/runs/act-hi224-c6d8196-20261115-1/act-final.pt",
    max_steps: int = 2600,
    run_id: str = "",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/policy-checks/stop",
    render_size: int = 224,
    open_loop_ticks: int = 30,
    max_delta: float = 0.06,
):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    run_id = run_id or time.strftime("stop-%Y%m%d-%H%M%S", time.gmtime())
    rows = list(
        check_seed.map(
            values,
            kwargs={
                "tier": tier,
                "checkpoint": checkpoint,
                "max_steps": int(max_steps),
                "run_id": run_id,
                "render_size": int(render_size),
                "open_loop_ticks": int(open_loop_ticks),
                "max_delta": float(max_delta),
            },
        )
    )
    rows.sort(key=lambda row: row["seed"])
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    nearest = [r["nearest"]["mm"] for r in rows if r.get("nearest")]
    centroid = [r["centroid"]["mm"] for r in rows if r.get("centroid")]
    heights = [r["nearest"]["height_mm"] for r in rows if r.get("nearest")]
    final_heights = [r["final"]["height_mm"] for r in rows if r.get("final")]
    import statistics

    aggregate = {
        "nearest_mm": {
            "median": statistics.median(nearest) if nearest else None,
            "min": min(nearest) if nearest else None,
            "max": max(nearest) if nearest else None,
        },
        "centroid_mm": {
            "median": statistics.median(centroid) if centroid else None,
            "min": min(centroid) if centroid else None,
            "max": max(centroid) if centroid else None,
        },
        "height_at_nearest_mm": {
            "median": statistics.median(heights) if heights else None,
            "min": min(heights) if heights else None,
            "max": max(heights) if heights else None,
        },
        "final_height_mm": {
            "median": statistics.median(final_heights) if final_heights else None,
            "min": min(final_heights) if final_heights else None,
            "max": max(final_heights) if final_heights else None,
        },
        "sentence": (
            "The policy's closest TCP point was "
            f"{statistics.median(nearest):.1f} mm from the nearest brick and "
            f"{statistics.median(centroid):.1f} mm from the scene centroid, "
            f"at {statistics.median(heights):.1f} mm height."
            if nearest and centroid and heights else "No policy frames were recorded."
        ),
    }
    manifest = {
        "format": "dryrun-policy-stop-check-v1",
        "source_commit": _commit(),
        "run_id": run_id,
        "tier": tier,
        "checkpoint": checkpoint,
        "seeds": values,
        "max_steps": int(max_steps),
        "render_size": int(render_size),
        "open_loop_ticks": int(open_loop_ticks),
        "max_delta": float(max_delta),
        "aggregate": aggregate,
        "rows": rows,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (target / "rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


__all__ = ["app", "check_seed"]
