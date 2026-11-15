"""Build small LeRobot demo datasets from ten-seed Modal Dict shards.

The rollout map writes one durable JSON value per seed.  This module fans the
values into independent ten-seed datasets, each with its own LeRobot metadata
and statistics.  A separate volume per shard avoids concurrent Volume commit
fan-in races from a single shared dataset.
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import modal


app = modal.App("dryrun-demo-shards")
shards = modal.Dict.from_name("dryrun-demo-shards", create_if_missing=True)
demo_volumes = {
    i: modal.Volume.from_name(f"dryrun-demo-shard-{i:02d}", create_if_missing=True)
    for i in range(20)
}
volume_mounts = {f"/shards/{i:02d}": volume for i, volume in demo_volumes.items()}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
    .pip_install("lerobot[dataset]==0.7.0", "numpy", "pillow", "opencv-python-headless")
    .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
)


def _decode(value: str | None):
    import numpy as np
    from PIL import Image

    if not value:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


@app.function(image=image, timeout=1800, cpu=4, volumes=volume_mounts)
def merge_shard(
    spec: dict[str, Any],
    *,
    run_id: str,
    tier: str = "loose",
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """Merge one ten-seed group into its private volume."""

    import sys

    sys.path.insert(0, "/root/repo")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from dryrun.demos import FEATURES

    index = int(spec["index"])
    if index not in demo_volumes:
        raise ValueError(f"shard index must be 0..19, got {index}")
    seeds = [int(seed) for seed in spec["seeds"]]
    root = Path(f"/shards/{index:02d}") / run_id / dataset_name
    if root.exists():
        raise FileExistsError(f"dataset path already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
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
    missing = []
    for seed in sorted(seeds):
        key = f"{run_id}/{tier}/{seed:04d}"
        blob = shards.get(key)
        if blob is None:
            missing.append(seed)
            continue
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8")
        payload = json.loads(blob)
        for cycle in payload.get("cycles", []):
            cycle_frames = cycle.get("frames", [])
            if not cycle_frames:
                continue
            for frame in cycle_frames:
                import numpy as np

                dataset.add_frame(
                    {
                        "observation.images.top": _decode(frame.get("top_jpeg")),
                        "observation.images.inspection": _decode(frame.get("inspection_jpeg")),
                        "observation.state": np.asarray(frame.get("state", [0] * 6), dtype=np.float32),
                        "action": np.asarray(frame.get("action", [0] * 6), dtype=np.float32),
                        "inspection.verdict": np.asarray([int(frame.get("verdict", 0))], dtype=np.int8),
                        "success": np.asarray([True], dtype=np.bool_),
                        "task": str(cycle.get("task", "sort brick")),
                    }
                )
                frames += 1
            dataset.save_episode()
            episodes += 1
    dataset.finalize()
    # The marker is written only after LeRobot has closed all parquet files.
    # Trainers must require it; metadata can become visible while a shard is
    # still being committed to its private volume.
    complete = root / "_COMPLETE.json"
    complete.write_text(
        json.dumps({"run_id": run_id, "shard": index, "episodes": episodes, "frames": frames})
        + "\n"
    )
    demo_volumes[index].commit()
    return {
        "format": "dryrun-modal-demo-shard-v1",
        "run_id": run_id,
        "tier": tier,
        "shard": index,
        "seeds": seeds,
        "missing": missing,
        "episodes": episodes,
        "frames": frames,
        "dataset": str(root),
    }


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(
    run_id: str,
    seeds: str = ",".join(str(i) for i in range(200)),
    tier: str = "loose",
    dataset_name: str = "dataset",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/demo-shards",
):
    values = [int(value) for value in seeds.split(",") if value.strip()]
    groups = [values[start : start + 10] for start in range(0, len(values), 10)]
    if len(groups) > 20:
        raise ValueError("at most 20 ten-seed shards are supported")
    specs = [{"index": index, "seeds": group} for index, group in enumerate(groups)]
    rows = list(
        merge_shard.map(
            specs,
            kwargs={"run_id": run_id, "tier": tier, "dataset_name": dataset_name},
        )
    )
    rows.sort(key=lambda row: row["shard"])
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "dryrun-modal-demo-shards-v1",
        "source_commit": _commit(),
        "run_id": run_id,
        "tier": tier,
        "dataset_name": dataset_name,
        "rows": rows,
        "episodes": sum(int(row["episodes"]) for row in rows),
        "frames": sum(int(row["frames"]) for row in rows),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (target / "rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


__all__ = ["app", "merge_shard"]
