"""Train ACT from independent ten-seed LeRobot datasets on an A10G."""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal


app = modal.App("dryrun-policy-shards")
train_volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
shard_volumes = {
    i: modal.Volume.from_name(f"dryrun-demo-shard-{i:02d}", create_if_missing=True)
    for i in range(20)
}
volume_mounts = {"/runs": train_volume}
volume_mounts.update({f"/shards/{i:02d}": volume for i, volume in shard_volumes.items()})
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "lerobot[dataset]==0.7.0", "numpy", "pyarrow", "pillow")
    .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
)


@app.function(image=image, gpu="A10G", timeout=7200, retries=2, volumes=volume_mounts)
def train_shards(
    run_id: str,
    shard_indices: list[int],
    dataset_name: str = "dataset",
    out_dir: str = "/runs/act-shards",
    epochs: int = 1,
    batch_size: int = 512,
    checkpoint_minutes: float = 15.0,
):
    import sys

    sys.path.insert(0, "/root/repo")
    import torch
    from torch.utils.data import ConcatDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from dryrun.train_policy import act_config

    datasets = []
    roots = []
    for index in sorted({int(value) for value in shard_indices}):
        if index not in shard_volumes:
            raise ValueError(f"shard index must be 0..19, got {index}")
        root = Path(f"/shards/{index:02d}") / run_id / dataset_name
        if not (root / "_COMPLETE.json").exists():
            continue
        datasets.append(LeRobotDataset("dryrun-so101", root=root, download_videos=False))
        roots.append(str(root))
    if not datasets:
        raise FileNotFoundError(f"no completed datasets for run {run_id}: {shard_indices}")
    dataset = ConcatDataset(datasets)

    cfg = act_config(chunk_size=60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    policy = __import__("lerobot.policies.act", fromlist=["ACTPolicy"]).ACTPolicy(cfg).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"
    started = time.perf_counter()
    next_checkpoint = float(checkpoint_minutes) * 60.0
    step = 0

    def batch(rows):
        top = torch.stack([row["observation.images.top"] for row in rows]).float()
        inspection = torch.stack([row["observation.images.inspection"] for row in rows]).float()
        state = torch.stack([row["observation.state"] for row in rows]).float()
        verdict = torch.stack([row["inspection.verdict"] for row in rows]).float().reshape(len(rows), 1)
        action = torch.stack([row["action"] for row in rows]).float()[:, None, :].repeat(1, 60, 1)
        return {
            "observation.images.top": top.to(device),
            "observation.images.inspection": inspection.to(device),
            "observation.state": torch.cat([state, verdict], dim=1).to(device),
            "action": action.to(device),
            "action_is_pad": torch.zeros(len(rows), 60, dtype=torch.bool, device=device),
        }

    rows_log = []
    with log_path.open("w") as log:
        for epoch in range(int(epochs)):
            order = torch.randperm(len(dataset)).tolist()
            for start in range(0, len(order), int(batch_size)):
                examples = [dataset[index] for index in order[start : start + int(batch_size)]]
                if not examples:
                    continue
                loss, pieces = policy(batch(examples))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                elapsed = time.perf_counter() - started
                row = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "l1_loss": float(pieces["l1_loss"]),
                    "seconds": elapsed,
                    "frames": len(dataset),
                    "device": str(device),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                rows_log.append(row)
                if elapsed >= next_checkpoint:
                    torch.save(
                        {
                            "state_dict": policy.state_dict(),
                            "step": step,
                            "architecture": "LeRobot ACT",
                            "chunk_size": 60,
                            "dataset_roots": roots,
                        },
                        output / f"act-step-{step:06d}.pt",
                    )
                    next_checkpoint += float(checkpoint_minutes) * 60.0
    final = output / "act-final.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "step": step,
            "architecture": "LeRobot ACT",
            "chunk_size": 60,
            "dataset_roots": roots,
        },
        final,
    )
    summary = {
        "format": "dryrun-modal-act-shards-v1",
        "run_id": run_id,
        "dataset_roots": roots,
        "shards": sorted(int(value) for value in shard_indices),
        "frames": len(dataset),
        "epochs": int(epochs),
        "steps": step,
        "initial_loss": rows_log[0]["loss"] if rows_log else None,
        "final_loss": rows_log[-1]["loss"] if rows_log else None,
        "checkpoint": str(final),
        "log": str(log_path),
        "device": str(device),
        "gpu": "A10G",
        "seconds": time.perf_counter() - started,
    }
    (output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    train_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str,
    shard_indices: str = ",".join(str(i) for i in range(20)),
    dataset_name: str = "dataset",
    out_dir: str = "/runs/act-shards",
    local_out: str = "/bulk3/runs/dryrun/m4/modal-map/shard-training",
    epochs: int = 1,
    batch_size: int = 512,
    checkpoint_minutes: float = 15.0,
):
    indices = [int(value) for value in shard_indices.split(",") if value.strip()]
    result = train_shards.remote(
        run_id=run_id,
        shard_indices=indices,
        dataset_name=dataset_name,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_minutes=checkpoint_minutes,
    )
    target = Path(local_out) / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


__all__ = ["app", "train_shards"]
