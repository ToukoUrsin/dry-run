"""Map closed-loop policy evaluation over seeded scenes on Modal."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


app = modal.App("dryrun-policy-eval")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
rows_store = modal.Dict.from_name("dryrun-policy-rows", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
    .pip_install("mujoco", "numpy", "torch", "torchvision", "lerobot[dataset]==0.7.0", "opencv-python-headless", "pillow", "openvino")
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
    """Count unique bricks that reached each physical rollout stage."""

    ids: dict[str, set[int]] = {
        "reached": set(),
        "grasped": set(),
        "lifted": set(),
        "presented": set(),
        "dropped": set(),
    }
    names = {
        "reach": "reached",
        "grasp": "grasped",
        "lift": "lifted",
        "present": "presented",
        "drop": "dropped",
    }
    for event in events or []:
        if not isinstance(event, dict):
            continue
        stage = names.get(str(event.get("event")))
        brick = event.get("brick")
        if stage is None or brick is None:
            continue
        try:
            ids[stage].add(int(brick))
        except (TypeError, ValueError):
            continue
    return {stage: len(values) for stage, values in ids.items()}


class _ApproachRecorder:
    """Measure the gripper TCP's nearest seeded brick distance from public frames."""

    def __init__(self, seed: int, tier: str):
        import numpy as np
        from dryrun.closed_form_ik import forward_position
        from dryrun.mjcf_station import layout_bricks
        from dryrun.env import TIERS

        self._forward = forward_position
        self._centers = np.asarray(
            [[float(s.x), float(s.y), float(getattr(s, "z", 0.019))] for s in layout_bricks(int(seed), TIERS[tier])],
            dtype=np.float32,
        )
        self.minimum_m = float("inf")

    def record(self, payload: dict):
        observation = payload.get("observation")
        if not observation or self._centers.size == 0:
            return
        import numpy as np

        joints = np.asarray(observation.get("joints", []), dtype=np.float32)
        if joints.size < 5:
            return
        tcp = np.asarray(self._forward(joints[:5]), dtype=np.float32)
        self.minimum_m = min(self.minimum_m, float(np.linalg.norm(self._centers - tcp[None, :], axis=1).min()))


class _ExpertPrefix:
    """Run the measured expert for the recorder's pre-reach prefix."""

    def __init__(self, policy, ticks: int):
        self.policy = policy
        self.remaining = max(0, int(ticks))

    def __call__(self, env, obs):
        if self.remaining:
            self.remaining -= 1
            return env.expert_action()
        return self.policy.act(obs)


@app.function(image=image, timeout=1800, cpu=4, retries=2, volumes={"/runs": volume})
def evaluate_seed(
    seed: int,
    tier: str = "loose",
    count: int | None = None,
    checkpoint: str = "/runs/act-final-modal.pt",
    max_steps: int = 3600,
    run_id: str = "latest",
    source_commit: str = "unknown",
    inspector_model: str = "/root/repo/model/dryrun_inspector.xml",
    open_loop_ticks: int = 0,
    max_delta: float = 0.06,
    render_size: int = 64,
    crop_top: bool = False,
    prefix_expert_ticks: int = 0,
    demo_jpeg: bool = False,
):
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import torch

    torch.set_num_threads(1)
    from dryrun.policy_benchmark import ACTController
    from dryrun.rollout import run
    from dryrun.inspector import Inspector

    policy_controller = ACTController(
        checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        open_loop_ticks=int(open_loop_ticks),
        max_delta=float(max_delta),
        crop_top=bool(crop_top),
        demo_jpeg=bool(demo_jpeg),
    )
    controller = _ExpertPrefix(policy_controller, int(prefix_expert_ticks)) if int(prefix_expert_ticks) > 0 else policy_controller
    inspector = Inspector(inspector_model, threads=1)
    approach = _ApproachRecorder(int(seed), str(tier))
    result = run(int(seed), controller, count=None if count is None else int(count), inspector=inspector, record=approach, tier=tier, max_steps=int(max_steps), render_size=int(render_size))
    stages = _stage_counts(result.get("events", []))
    row = {
        "format": "dryrun-policy-eval-row-v1",
        "source": "modal-map",
        "run_id": str(run_id),
        "source_commit": str(source_commit),
        "seed": int(seed),
        "tier": str(tier),
        "count": None if count is None else int(count),
        "checkpoint": checkpoint,
        "max_steps": int(max_steps),
        "open_loop_ticks": int(open_loop_ticks),
        "max_delta": float(max_delta),
        "render_size": int(render_size),
        "crop_top": bool(crop_top),
        "prefix_expert_ticks": int(prefix_expert_ticks),
        "demo_jpeg": bool(demo_jpeg),
        "bricks": int(result.get("bricks", 0)),
        "successes": int(result.get("successes", 0)),
        "score": float(result.get("score", 0.0)),
        "stages": {
            **stages,
            "right_bowl": int(result.get("successes", 0)),
        },
        "closest_approach_mm": None if approach.minimum_m == float("inf") else round(1000.0 * approach.minimum_m, 3),
    }
    store_key = f"{run_id}/{tier}/{int(seed):04d}"
    # Concurrent Volume commits are lossy fan-in. The checkpoint still comes
    # from the volume, while result rows use atomic per-seed Dict puts.
    rows_store.put(store_key, json.dumps(row, sort_keys=True))
    row["store_key"] = store_key
    return row


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2",
    tier: str = "loose",
    count: int | None = None,
    checkpoint: str = "/runs/act-final-modal.pt",
    max_steps: int = 3600,
    run_id: str = "",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/policy-map",
    download: bool = False,
    inspector_model: str = "/root/repo/model/dryrun_inspector.xml",
    open_loop_ticks: int = 0,
    max_delta: float = 0.06,
    render_size: int = 64,
    crop_top: bool = False,
    prefix_expert_ticks: int = 0,
    demo_jpeg: bool = False,
):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    dict_key_prefix = f"{run_id}/{tier}"
    source_commit = _commit()
    rows = list(
        evaluate_seed.map(
            values,
            kwargs={
                "tier": tier,
                "count": None if count is None else int(count),
                "checkpoint": checkpoint,
                "max_steps": int(max_steps),
                "run_id": run_id,
                "source_commit": source_commit,
                "inspector_model": inspector_model,
                "open_loop_ticks": int(open_loop_ticks),
                "max_delta": float(max_delta),
                "render_size": int(render_size),
                "crop_top": bool(crop_top),
                "prefix_expert_ticks": int(prefix_expert_ticks),
                "demo_jpeg": bool(demo_jpeg),
            },
        )
    )
    rows.sort(key=lambda row: row["seed"])
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "dryrun-modal-policy-map-v1",
        "source_commit": source_commit,
        "run_id": run_id,
        "tier": tier,
        "count": None if count is None else int(count),
        "checkpoint": checkpoint,
        "max_steps": int(max_steps),
        "open_loop_ticks": int(open_loop_ticks),
        "max_delta": float(max_delta),
        "render_size": int(render_size),
        "crop_top": bool(crop_top),
        "prefix_expert_ticks": int(prefix_expert_ticks),
        "demo_jpeg": bool(demo_jpeg),
        "dict_name": "dryrun-policy-rows",
        "dict_key_prefix": dict_key_prefix,
        "rows": rows,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (target / "rows.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    if download:
        row_dir = target / "rows"
        row_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            key = f"{run_id}/{tier}/{int(row['seed']):04d}"
            blob = rows_store.get(key)
            if blob is None:
                raise RuntimeError(f"mapped policy row missing from Modal Dict: {key}")
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8")
            (row_dir / f"seed-{int(row['seed']):04d}.json").write_text(str(blob) + "\n")
    print(json.dumps(manifest, indent=2))
