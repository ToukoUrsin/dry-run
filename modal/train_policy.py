"""Modal GPU ACT trainer for Dry Run.

The volume is intentionally named ``dryrun-demos`` and the app
``dryrun-policy`` as specified by the supervisor.  This file is also a
reproducible record of the remote command; local smoke runs do not pretend to
be GPU evidence.
"""
from __future__ import annotations
import modal

app = modal.App("dryrun-policy")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "torchvision", "lerobot[dataset]==0.7.0", "numpy", "pyarrow", "pillow"))


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/runs": volume})
def train(
    dataset: str = "/runs/lerobot-real",
    out_dir: str = "/runs/act-training",
    epochs: int = 2,
    batch_size: int = 8,
    checkpoint_every: int = 25,
):
    """Train ACT remotely and checkpoint into the persistent Modal volume."""
    import json, time, torch
    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act import ACTConfig, ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    features = {"observation.images.top": PolicyFeature(FeatureType.VISUAL, (3,64,64)), "observation.images.inspection": PolicyFeature(FeatureType.VISUAL, (3,64,64)), "observation.state": PolicyFeature(FeatureType.STATE, (7,))}
    chunk_size = 30
    cfg = ACTConfig(input_features=features, output_features={"action": PolicyFeature(FeatureType.ACTION,(6,))}, chunk_size=chunk_size, n_action_steps=1, use_vae=False, pretrained_backbone_weights=None, dim_model=64, n_heads=4, n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=128, n_vae_encoder_layers=1, optimizer_lr=2e-4, push_to_hub=False)
    ds = LeRobotDataset("dryrun-so101", root=dataset, download_videos=False, delta_timestamps={"action": [(i + 1) / 30.0 for i in range(chunk_size)]})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy(cfg).to(device); opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr)
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"; started = time.time()
    with log_path.open("w") as log:
        step = 0
        for epoch in range(epochs):
            for i in range(0, len(ds), int(batch_size)):
                rows = [ds[j] for j in range(i, min(i+int(batch_size), len(ds)))]
                top = torch.stack([r["observation.images.top"] for r in rows]).float().to(device); inspection = torch.stack([r["observation.images.inspection"] for r in rows]).float().to(device); state = torch.cat([torch.stack([r["observation.state"] for r in rows]).float(), torch.stack([r["inspection.verdict"] for r in rows]).float().reshape(len(rows),1)],1).to(device); action = torch.stack([r["action"] for r in rows]).float().to(device); action_is_pad = torch.stack([r["action_is_pad"] for r in rows]).bool().to(device); batch={"observation.images.top":top,"observation.images.inspection":inspection,"observation.state":state,"action":action,"action_is_pad":action_is_pad}
                loss, info = policy(batch); opt.zero_grad(); loss.backward(); opt.step(); step += 1; row={"step":step,"epoch":epoch+1,"loss":float(loss),"l1_loss":float(info["l1_loss"]),"seconds":time.time()-started}; log.write(json.dumps(row)+"\n"); log.flush()
                if step % checkpoint_every == 0:
                    torch.save({"state_dict":policy.state_dict(),"step":step,"architecture":"LeRobot ACT","chunk_size":chunk_size,"image_size":64}, output / f"act-step-{step:06d}.pt")
    final = output / "act-final.pt"
    torch.save({"state_dict":policy.state_dict(),"step":step,"architecture":"LeRobot ACT","chunk_size":chunk_size,"image_size":64}, final)
    summary = {"dataset": dataset, "output": str(output), "steps": step, "checkpoint": str(final), "log": str(log_path), "gpu": "cuda" if device.type == "cuda" else "cpu", "chunk_size":chunk_size, "image_size":64, "seconds": time.time()-started}
    (output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit(); return summary


@app.local_entrypoint()
def main(
    dataset: str = "/runs/lerobot-real",
    out_dir: str = "/runs/act-training",
    epochs: int = 2,
    batch_size: int = 8,
    checkpoint_every: int = 25,
):
    print(train.remote(dataset=dataset, out_dir=out_dir, epochs=epochs, batch_size=batch_size, checkpoint_every=checkpoint_every))
