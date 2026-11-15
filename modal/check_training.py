"""Prompt 040 check 1: compare the trained ACT on training scenes and data.

This is deliberately a diagnostic entrypoint rather than a benchmark.  It
maps the final high-resolution checkpoint over the twenty training seeds and
also compares one stored LeRobot episode frame by frame with the policy's
actions.  The rollout path remains the public ``dryrun.rollout.run`` API.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


app = modal.App("dryrun-policy-check-training")
runs_volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
shard_volume = modal.Volume.from_name("dryrun-demo-shard-00", create_if_missing=True)

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
    names = {"reach": "reached", "grasp": "grasped", "lift": "lifted", "present": "presented", "drop": "dropped"}
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
    def __init__(self, seed: int, tier: str):
        import numpy as np
        from dryrun.closed_form_ik import forward_position
        from dryrun.env import TIERS
        from dryrun.mjcf_station import layout_bricks

        self._forward = forward_position
        self._centers = np.asarray(
            [[float(s.x), float(s.y), float(getattr(s, "z", 0.019))]
             for s in layout_bricks(int(seed), TIERS[tier])], dtype=np.float32
        )
        self.minimum_m = float("inf")

    def record(self, payload: dict):
        import numpy as np
        obs = payload.get("observation")
        if not obs or not self._centers.size:
            return
        joints = np.asarray(obs.get("joints", []), dtype=np.float32)
        if joints.size < 5:
            return
        tcp = np.asarray(self._forward(joints[:5]), dtype=np.float32)
        self.minimum_m = min(self.minimum_m, float(np.linalg.norm(self._centers - tcp[None, :], axis=1).min()))


@app.function(image=image, timeout=1800, cpu=4, retries=2, volumes={"/runs": runs_volume})
def evaluate_seed(
    seed: int,
    checkpoint: str = "/runs/act-hi224-c6d8196-20261115-1/act-final.pt",
    max_steps: int = 2600,
    render_size: int = 224,
    run_id: str = "policy-check-training",
):
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import torch
    from dryrun.inspector import Inspector
    from dryrun.policy_benchmark import ACTController
    from dryrun.rollout import run

    torch.set_num_threads(1)
    approach = _Approach(int(seed), "loose")
    controller = ACTController(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu", open_loop_ticks=30, max_delta=0.06)
    result = run(int(seed), controller, inspector=Inspector("/root/repo/model/dryrun_inspector.xml", threads=1), record=approach, tier="loose", max_steps=int(max_steps), render_size=int(render_size))
    stages = _stage_counts(result.get("events", []))
    return {
        "seed": int(seed), "tier": "loose", "run_id": run_id,
        "bricks": int(result.get("bricks", 0)), "successes": int(result.get("successes", 0)),
        "score": float(result.get("score", 0.0)), "stages": {**stages, "right_bowl": int(result.get("successes", 0))},
        "closest_approach_mm": None if approach.minimum_m == float("inf") else round(1000.0 * approach.minimum_m, 3),
        "max_steps": int(max_steps), "render_size": int(render_size), "open_loop_ticks": 30, "max_delta": 0.06,
    }


@app.function(image=image, timeout=1800, cpu=4, volumes={"/shard0": shard_volume, "/runs": runs_volume})
def compare_recorded_episode(
    checkpoint: str = "/runs/act-hi224-c6d8196-20261115-1/act-final.pt",
    run_id: str = "policy-check-training",
    max_frames: int = 330,
):
    """Compare policy predictions against the first stored high-res episode."""
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from dryrun.train_policy import load_policy

    torch.set_num_threads(1)
    root = Path("/shard0/loose-200-hi224-c6d8196-20261115-1/shard-dataset-hi224")
    complete = root / "_COMPLETE.json"
    if not complete.exists():
        raise FileNotFoundError(f"dataset shard not complete: {complete}")
    policy = load_policy(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(policy.parameters()).device
    chunk_size = int(policy.config.chunk_size)
    dataset = LeRobotDataset(
        "dryrun-so101", root=root, download_videos=False,
        delta_timestamps={"action": [(i + 1) / 30.0 for i in range(chunk_size)]},
    )
    n = min(int(max_frames), len(dataset))
    action_errors = []
    image_errors = []
    joint_errors = [[] for _ in range(6)]
    verdict_counts = {}
    for i in range(n):
        row = dataset[i]
        top = row["observation.images.top"].float()
        inspection = row["observation.images.inspection"].float()
        state = row["observation.state"].float().reshape(1, 6)
        verdict = row["inspection.verdict"].float().reshape(1, 1)
        if top.ndim == 3:
            top = top.unsqueeze(0)
            inspection = inspection.unsqueeze(0)
        batch = {
            "observation.images.top": top.to(device),
            "observation.images.inspection": inspection.to(device),
            "observation.state": torch.cat([state, verdict], dim=1).to(device),
        }
        with torch.no_grad():
            pred = policy.select_action(batch).detach().cpu().numpy().reshape(6)
        recorded = row["action"][0].detach().cpu().numpy().reshape(6)
        if bool(row["action_is_pad"][0]):
            continue
        err = np.abs(pred - recorded)
        action_errors.append(err)
        for j in range(6):
            joint_errors[j].append(float(err[j]))
        # The test controller receives this exact CHW float image.  Convert it
        # through the HWC uint8 boundary used by rollout and back to quantify
        # any preprocessing loss (including resize when the shapes differ).
        h, w = int(top.shape[-2]), int(top.shape[-1])
        policy_u8 = np.clip(top[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255).round().astype(np.uint8)
        stored_u8 = np.clip(row["observation.images.top"].detach().cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255).round().astype(np.uint8)
        image_errors.append(float(np.abs(policy_u8.astype(np.float32) - stored_u8.astype(np.float32)).mean()))
        key = str(int(row["inspection.verdict"].reshape(-1)[0]))
        verdict_counts[key] = verdict_counts.get(key, 0) + 1
    errors = np.asarray(action_errors, dtype=np.float64)
    return {
        "format": "dryrun-policy-training-check-v1", "run_id": run_id, "checkpoint": checkpoint,
        "dataset": str(root), "frames_compared": n, "episode_frames_capped": int(max_frames),
        "image_size": 224, "verdict_counts": verdict_counts,
        "recorded_vs_policy_action_mae_per_joint": [round(float(x), 8) for x in errors.mean(axis=0)],
        "recorded_vs_policy_action_rmse_per_joint": [round(float(x), 8) for x in np.sqrt((errors ** 2).mean(axis=0))],
        "recorded_vs_policy_action_max_abs_per_joint": [round(float(x), 8) for x in errors.max(axis=0)],
        "recorded_vs_policy_action_mae_all_joints": round(float(errors.mean()), 8),
        "stored_vs_policy_input_image_mae_uint8": round(float(np.mean(image_errors)), 8),
        "stored_vs_policy_input_image_max_mae_uint8": round(float(np.max(image_errors)), 8),
        "device": str(device),
    }


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(
    seeds: str = ",".join(str(i) for i in range(20)),
    checkpoint: str = "/runs/act-hi224-c6d8196-20261115-1/act-final.pt",
    run_id: str = "",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/policy-checks",
    max_steps: int = 2600,
):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    run_id = run_id or f"training-scenes-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    rows = list(evaluate_seed.map(values, kwargs={"checkpoint": checkpoint, "max_steps": int(max_steps), "run_id": run_id}))
    rows.sort(key=lambda row: row["seed"])
    episode = compare_recorded_episode.remote(checkpoint=checkpoint, run_id=run_id)
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "dryrun-policy-training-check-v1", "source_commit": _commit(), "run_id": run_id,
        "checkpoint": checkpoint, "seeds": values, "max_steps": int(max_steps),
        "rows": rows, "recorded_episode": episode,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (target / "rows.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


__all__ = ["app", "evaluate_seed", "compare_recorded_episode"]
