"""Measure ACT one-step action MAE on every frame of a Modal dataset."""
from __future__ import annotations
import json
from pathlib import Path
import modal

app = modal.App("dryrun-action-mae")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "torchvision", "lerobot[dataset]==0.7.0", "numpy", "pyarrow", "pillow")
         .add_local_dir("dryrun", remote_path="/root/repo/dryrun"))

@app.function(image=image, gpu="T4", timeout=1800, volumes={"/runs": volume})
def evaluate(dataset: str, checkpoint: str, batch_size: int = 64):
    import sys
    sys.path.insert(0, "/root/repo")
    import numpy as np, torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from dryrun.train_policy import load_policy
    ds = LeRobotDataset("dryrun-so101", root=dataset, download_videos=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(checkpoint, device=str(device))
    total = np.zeros(6, dtype=np.float64); n = 0
    with torch.no_grad():
        for start in range(0, len(ds), int(batch_size)):
            rows = [ds[i] for i in range(start, min(start + int(batch_size), len(ds)))]
            top = torch.stack([r["observation.images.top"] for r in rows]).float().to(device)
            inspection = torch.stack([r["observation.images.inspection"] for r in rows]).float().to(device)
            state = torch.stack([r["observation.state"] for r in rows]).float().to(device)
            verdict = torch.stack([r["inspection.verdict"] for r in rows]).float().reshape(len(rows), 1).to(device)
            batch = {"observation.images.top": top, "observation.images.inspection": inspection,
                     "observation.state": torch.cat([state, verdict], 1)}
            pred = policy.select_action(batch).detach().cpu().numpy()
            target = np.stack([r["action"].numpy() for r in rows])
            total += np.abs(pred - target).sum(axis=0); n += len(rows)
    return {"dataset": dataset, "checkpoint": checkpoint, "frames": int(n),
            "mae_per_joint": (total / n).tolist(), "mae_all_joints": float(total.sum() / (n * 6)),
            "device": str(device)}

@app.local_entrypoint()
def main(dataset: str, checkpoint: str, out: str = "/bulk3/runs/dryrun/m4/modal-map/onebrick-overfit/training/action-mae.json"):
    result = evaluate.remote(dataset, checkpoint)
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
