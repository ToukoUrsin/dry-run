"""Prompt 040 check 2: evaluate the final ACT policy on one-brick scenes.

This diagnostic intentionally uses the public rollout boundary and the
``count=1`` scene parameter.  It does not inspect simulator state in the
controller; the nearest-brick distance is a post-rollout diagnostic only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


app = modal.App("dryrun-policy-check-single")
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


def _stage_counts(events: list[dict]) -> dict[str, int]:
    ids = {name: set() for name in ("reached", "grasped", "lifted", "presented", "dropped")}
    names = {
        "reach": "reached", "grasp": "grasped", "lift": "lifted",
        "present": "presented", "drop": "dropped",
    }
    for event in events or []:
        if not isinstance(event, dict) or event.get("brick") is None:
            continue
        stage = names.get(str(event.get("event")))
        if stage is not None:
            try:
                ids[stage].add(int(event["brick"]))
            except (TypeError, ValueError):
                pass
    return {key: len(value) for key, value in ids.items()}


class _Approach:
    def __init__(self, seed: int):
        import numpy as np
        from dryrun.closed_form_ik import forward_position
        from dryrun.mjcf_station import layout_bricks
        from dryrun.env import TIERS

        self._forward = forward_position
        spec = layout_bricks(int(seed), TIERS["loose"], count=1)[0]
        self.center = np.asarray([float(spec.x), float(spec.y), float(getattr(spec, "z", 0.019))], dtype=np.float32)
        self.min_m = float("inf")
        self.min_z = float("inf")

    def __call__(self, payload: dict):
        import numpy as np
        obs = payload.get("observation")
        if not obs:
            return
        joints = np.asarray(obs.get("joints", []), dtype=np.float32)
        if joints.size < 5:
            return
        tcp = np.asarray(self._forward(joints[:5]), dtype=np.float32)
        self.min_m = min(self.min_m, float(np.linalg.norm(tcp - self.center)))
        self.min_z = min(self.min_z, float(tcp[2]))


@app.function(image=image, timeout=1800, cpu=4, retries=2, volumes={"/runs": volume})
def evaluate_one(
    seed: int,
    checkpoint: str = "/runs/act-hi224-c6d8196-20261115-1/act-final.pt",
    max_steps: int = 3600,
    render_size: int = 224,
    max_delta: float = 0.06,
    run_id: str = "single-check",
    source_commit: str = "unknown",
):
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import torch

    torch.set_num_threads(1)
    from dryrun.inspector import Inspector
    from dryrun.policy_benchmark import ACTController
    from dryrun.rollout import run

    controller = ACTController(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu", max_delta=float(max_delta))
    inspector = Inspector("/root/repo/model/dryrun_inspector.xml", threads=1)
    approach = _Approach(int(seed))
    result = run(
        int(seed), controller, inspector=inspector, tier="loose", count=1,
        max_steps=int(max_steps), render_size=int(render_size), record=approach,
    )
    stages = _stage_counts(result.get("events", []))
    return {
        "format": "dryrun-policy-single-check-v1",
        "run_id": str(run_id),
        "source_commit": str(source_commit),
        "seed": int(seed),
        "tier": "loose",
        "count": 1,
        "checkpoint": str(checkpoint),
        "max_steps": int(max_steps),
        "render_size": int(render_size),
        "max_delta": float(max_delta),
        "bricks": int(result.get("bricks", 0)),
        "successes": int(result.get("successes", 0)),
        "score": float(result.get("score", 0.0)),
        "stages": {**stages, "right_bowl": int(result.get("successes", 0))},
        "closest_approach_mm": None if approach.min_m == float("inf") else round(1000.0 * approach.min_m, 3),
        "closest_height_mm": None if approach.min_z == float("inf") else round(1000.0 * approach.min_z, 3),
    }


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(
    seeds: str = ",".join(str(i) for i in range(2000, 2020)),
    checkpoint: str = "/runs/act-hi224-c6d8196-20261115-1/act-final.pt",
    max_steps: int = 3600,
    render_size: int = 224,
    max_delta: float = 0.06,
    run_id: str = "",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/policy-checks",
):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    run_id = run_id or f"single-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    source_commit = _commit()
    rows = list(evaluate_one.map(values, kwargs={
        "checkpoint": checkpoint,
        "max_steps": int(max_steps),
        "render_size": int(render_size),
        "max_delta": float(max_delta),
        "run_id": run_id,
        "source_commit": source_commit,
    }))
    rows.sort(key=lambda row: row["seed"])
    stages = {name: sum(row["stages"][name] for row in rows) for name in ("reached", "grasped", "lifted", "presented", "dropped", "right_bowl")}
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "dryrun-policy-single-check-v1",
        "run_id": run_id,
        "source_commit": source_commit,
        "seeds": values,
        "checkpoint": checkpoint,
        "count": 1,
        "max_steps": int(max_steps),
        "render_size": int(render_size),
        "max_delta": float(max_delta),
        "rows": rows,
        "stage_totals": stages,
        "successes": sum(int(row["successes"]) for row in rows),
        "bricks": sum(int(row["bricks"]) for row in rows),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (target / "rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps(manifest, indent=2))


__all__ = ["app", "evaluate_one"]
