"""Merge successful Modal demo shards into one LeRobot dataset.

Mapped rollout workers deliberately write independent Dict values. This job
is the single writer for LeRobot metadata, so it can safely create episodes
after a large ``.map`` has completed.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import modal


app = modal.App("dryrun-demo-merge")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
shards = modal.Dict.from_name("dryrun-demo-shards", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
    .pip_install("lerobot[dataset]==0.7.0", "numpy", "pillow", "opencv-python-headless")
    .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
)


# A large map can contain hundreds of successful brick episodes.  LeRobot's
# per-episode image-stat pass is intentionally serialized by this single
# writer, so the merge needs more than the 30-minute rollout-worker limit.
@app.function(image=image, timeout=7200, cpu=4, volumes={"/runs": volume})
def merge(
    run_id: str,
    seeds: list[int],
    tier: str = "loose",
    dataset_path: str = "/runs/lerobot-modal",
):
    sys.path.insert(0, "/root/repo")
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from dryrun.demos import FEATURES

    root = Path(dataset_path)
    if root.exists():
        raise FileExistsError(f"dataset path already exists: {root}; choose a new run id")
    root.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(
        "dryrun-so101",
        fps=30,
        features=FEATURES,
        root=root,
        robot_type="so101",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    episodes = 0
    frames = 0
    for seed in sorted(int(value) for value in seeds):
        key = f"{run_id}/{tier}/{seed:04d}"
        blob = shards.get(key)
        if blob is None:
            continue
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8")
        payload = json.loads(blob)
        for cycle in payload.get("cycles", []):
            cycle_frames = cycle.get("frames", [])
            if not cycle_frames:
                continue
            for frame in cycle_frames:
                def decode(name: str) -> np.ndarray:
                    encoded = frame.get(name)
                    if not encoded:
                        return np.zeros((64, 64, 3), dtype=np.uint8)
                    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))

                ds.add_frame(
                    {
                        "observation.images.top": decode("top_jpeg"),
                        "observation.images.inspection": decode("inspection_jpeg"),
                        "observation.state": np.asarray(frame.get("state", [0] * 6), dtype=np.float32),
                        "action": np.asarray(frame.get("action", [0] * 6), dtype=np.float32),
                        "inspection.verdict": np.asarray([int(frame.get("verdict", 0))], dtype=np.int8),
                        "success": np.asarray([True], dtype=np.bool_),
                        "task": str(cycle.get("task", "sort brick")),
                    }
                )
                frames += 1
            ds.save_episode()
            episodes += 1
    ds.finalize()
    volume.commit()
    return {"run_id": run_id, "tier": tier, "seeds": len(seeds), "episodes": episodes, "frames": frames, "dataset": dataset_path}


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(
    run_id: str,
    seeds: str = "1,2",
    tier: str = "loose",
    dataset_path: str = "/runs/lerobot-modal",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/merged",
):
    values = [int(value) for value in seeds.split(",") if value.strip()]
    result = merge.remote(run_id=run_id, seeds=values, tier=tier, dataset_path=dataset_path)
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "dryrun-modal-lerobot-merge-v1", "source_commit": _commit(), **result}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
