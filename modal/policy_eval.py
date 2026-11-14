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
    .pip_install("mujoco", "numpy", "torch", "torchvision", "lerobot[dataset]==0.7.0", "opencv-python-headless", "pillow")
    .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
    .add_local_dir("assets", remote_path="/root/repo/assets")
    .add_local_file("data/reach_map_polar.json", remote_path="/root/repo/data/reach_map_polar.json")
)


@app.function(image=image, timeout=1800, cpu=4, gpu="T4", retries=2, volumes={"/runs": volume})
def evaluate_seed(
    seed: int,
    tier: str = "loose",
    checkpoint: str = "/runs/act-final-modal.pt",
    max_steps: int = 3600,
    run_id: str = "latest",
    source_commit: str = "unknown",
):
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import torch

    torch.set_num_threads(1)
    from dryrun.policy_benchmark import ACTController
    from dryrun.rollout import run

    controller = ACTController(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    result = run(int(seed), controller, tier=tier, max_steps=int(max_steps))
    row = {
        "format": "dryrun-policy-eval-row-v1",
        "source": "modal-map",
        "run_id": str(run_id),
        "source_commit": str(source_commit),
        "seed": int(seed),
        "tier": str(tier),
        "checkpoint": checkpoint,
        "max_steps": int(max_steps),
        "bricks": int(result.get("bricks", 0)),
        "successes": int(result.get("successes", 0)),
        "score": float(result.get("score", 0.0)),
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
    checkpoint: str = "/runs/act-final-modal.pt",
    max_steps: int = 3600,
    run_id: str = "",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/policy-map",
    download: bool = False,
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
                "checkpoint": checkpoint,
                "max_steps": int(max_steps),
                "run_id": run_id,
                "source_commit": source_commit,
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
        "checkpoint": checkpoint,
        "max_steps": int(max_steps),
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
