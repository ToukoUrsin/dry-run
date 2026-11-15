"""Continue a compact ACT overfit from a published checkpoint."""
from __future__ import annotations
import json, time
from pathlib import Path
import modal

app = modal.App("dryrun-policy-continue")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "torchvision", "lerobot[dataset]==0.7.0", "numpy", "pyarrow", "pillow")
         .add_local_dir("dryrun", remote_path="/root/repo/dryrun"))

@app.function(image=image, gpu="T4", timeout=1800, volumes={"/runs": volume})
def train(dataset: str, checkpoint: str, out_dir: str, epochs: int = 50, batch_size: int = 64, lr: float = 5e-5):
    import sys
    sys.path.insert(0, "/root/repo")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from dryrun.train_policy import act_config
    chunk_size = 30
    ds = LeRobotDataset("dryrun-so101", root=dataset, download_videos=False, delta_timestamps={"action": [(i + 1) / 30.0 for i in range(chunk_size)]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = act_config(chunk_size=chunk_size, image_size=64)
    policy = ACTPolicy(cfg).to(device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(blob["state_dict"]); policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=float(lr))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    log = (out / "training.jsonl").open("w"); started = time.time(); step = 0
    for epoch in range(int(epochs)):
        order = torch.randperm(len(ds)).tolist()
        for start in range(0, len(order), int(batch_size)):
            rows = [ds[i] for i in order[start:start+int(batch_size)]]
            top = torch.stack([r["observation.images.top"] for r in rows]).float().to(device)
            inspection = torch.stack([r["observation.images.inspection"] for r in rows]).float().to(device)
            state = torch.stack([r["observation.state"] for r in rows]).float()
            verdict = torch.stack([r["inspection.verdict"] for r in rows]).float().reshape(len(rows), 1)
            action = torch.stack([r["action"] for r in rows]).float()
            action_is_pad = torch.stack([r["action_is_pad"] for r in rows]).bool()
            batch = {"observation.images.top": top, "observation.images.inspection": inspection,
                     "observation.state": torch.cat([state, verdict], 1).to(device),
                     "action": action.to(device), "action_is_pad": action_is_pad.to(device)}
            loss, info = policy(batch); opt.zero_grad(); loss.backward(); opt.step(); step += 1
            log.write(json.dumps({"step": step, "epoch": epoch+1, "loss": float(loss.detach()), "l1_loss": float(info["l1_loss"]), "seconds": time.time()-started})+"\n"); log.flush()
    log.close(); final = out / "act-final.pt"
    torch.save({"state_dict": policy.state_dict(), "step": step, "architecture": "LeRobot ACT", "chunk_size": chunk_size, "image_size": 64}, final)
    summary = {"dataset": dataset, "checkpoint_in": checkpoint, "checkpoint": str(final), "epochs": int(epochs), "steps": step, "lr": float(lr), "device": str(device), "seconds": time.time()-started}
    (out / "training.json").write_text(json.dumps(summary, indent=2) + "\n"); volume.commit(); return summary

@app.local_entrypoint()
def main(dataset: str, checkpoint: str, out_dir: str = "/runs/onebrick-overfit-20-act-continue-20261115-1", epochs: int = 50, batch_size: int = 64, lr: float = 5e-5, out: str = "/bulk3/runs/dryrun/m4/modal-map/onebrick-overfit/training/continue.json"):
    result = train.remote(dataset, checkpoint, out_dir, int(epochs), int(batch_size), float(lr))
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
