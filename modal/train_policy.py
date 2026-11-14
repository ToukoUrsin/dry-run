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
         .pip_install("torch", "torchvision", "lerobot[dataset]", "numpy", "pyarrow"))


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/runs": volume})
def train(dataset: str = "/runs/lerobot-real", epochs: int = 2, checkpoint_every: int = 25):
    """Train ACT remotely and checkpoint into the persistent Modal volume."""
    import json, time, torch
    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act import ACTConfig, ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    features = {"observation.images.top": PolicyFeature(FeatureType.VISUAL, (3,64,64)), "observation.images.inspection": PolicyFeature(FeatureType.VISUAL, (3,64,64)), "observation.state": PolicyFeature(FeatureType.STATE, (7,))}
    cfg = ACTConfig(input_features=features, output_features={"action": PolicyFeature(FeatureType.ACTION,(6,))}, chunk_size=4, n_action_steps=1, use_vae=False, pretrained_backbone_weights=None, dim_model=64, n_heads=4, n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=128, n_vae_encoder_layers=1, optimizer_lr=2e-4, push_to_hub=False)
    ds = LeRobotDataset("dryrun-so101", root=dataset, download_videos=False); device = torch.device("cuda"); policy = ACTPolicy(cfg).to(device); opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr); log_path = Path("/runs/modal-training.jsonl"); started = time.time()
    with log_path.open("w") as log:
        step = 0
        for epoch in range(epochs):
            for i in range(0, len(ds), 8):
                rows = [ds[j] for j in range(i, min(i+8, len(ds)))]
                top = torch.stack([r["observation.images.top"] for r in rows]).float().to(device); inspection = torch.stack([r["observation.images.inspection"] for r in rows]).float().to(device); state = torch.cat([torch.stack([r["observation.state"] for r in rows]).float(), torch.stack([r["inspection.verdict"] for r in rows]).float().reshape(len(rows),1)],1).to(device); action = torch.stack([r["action"] for r in rows]).float()[:,None,:].repeat(1,4,1).to(device); batch={"observation.images.top":top,"observation.images.inspection":inspection,"observation.state":state,"action":action,"action_is_pad":torch.zeros(len(rows),4,dtype=torch.bool,device=device)}
                loss, info = policy(batch); opt.zero_grad(); loss.backward(); opt.step(); step += 1; row={"step":step,"epoch":epoch+1,"loss":float(loss),"l1_loss":float(info["l1_loss"]),"seconds":time.time()-started}; log.write(json.dumps(row)+"\n"); log.flush()
                if step % checkpoint_every == 0: torch.save({"state_dict":policy.state_dict(),"step":step,"architecture":"LeRobot ACT"}, f"/runs/act-step-{step:06d}.pt")
    torch.save({"state_dict":policy.state_dict(),"step":step,"architecture":"LeRobot ACT"}, "/runs/act-final-modal.pt"); volume.commit(); return {"steps":step,"checkpoint":"/runs/act-final-modal.pt","log":str(log_path),"gpu":"T4","seconds":time.time()-started}


@app.local_entrypoint()
def main():
    print(train.remote())
