"""Train a compact LeRobot ACT policy and emit resumable checkpoints."""
from __future__ import annotations
from pathlib import Path
import argparse, json, time
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType

ROOT = Path("/bulk2/runs/dryrun/m4")


def act_config() -> ACTConfig:
    features = {
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        "observation.images.inspection": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
    }
    return ACTConfig(
        input_features=features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        chunk_size=4, n_action_steps=1, use_vae=False,
        pretrained_backbone_weights=None, dim_model=64, n_heads=4,
        n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=128,
        n_vae_encoder_layers=1, optimizer_lr=2e-4, push_to_hub=False,
    )


def _batch(rows, device):
    # LeRobot returns images as CHW float tensors in [0,1].  Add the verdict
    # channel to the six measured joints; this is the inspector interface.
    top = torch.stack([r["observation.images.top"] for r in rows]).float()
    inspection = torch.stack([r["observation.images.inspection"] for r in rows]).float()
    state = torch.stack([r["observation.state"] for r in rows]).float()
    verdict = torch.stack([r["inspection.verdict"] for r in rows]).float().reshape(len(rows), 1)
    state = torch.cat([state, verdict], dim=1)
    action = torch.stack([r["action"] for r in rows]).float()
    action = action[:, None, :].repeat(1, 4, 1)
    return {"observation.images.top": top.to(device), "observation.images.inspection": inspection.to(device),
            "observation.state": state.to(device), "action": action.to(device),
            "action_is_pad": torch.zeros(len(rows), 4, dtype=torch.bool, device=device)}


def train(dataset: str | Path, *, out: Path = ROOT, epochs: int = 2, batch_size: int = 8, checkpoint_every: int = 25):
    out.mkdir(parents=True, exist_ok=True); log_path = out / "training.jsonl"
    ds = LeRobotDataset("dryrun-so101", root=dataset, download_videos=False)
    cfg = act_config(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy(cfg).to(device); policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr)
    steps = 0; rows = []; started = time.perf_counter()
    with log_path.open("w") as log:
        for epoch in range(epochs):
            order = torch.randperm(len(ds)).tolist()
            for start in range(0, len(order), batch_size):
                examples = [ds[i] for i in order[start:start + batch_size]]
                if not examples: continue
                loss, pieces = policy(_batch(examples, device)); optimizer.zero_grad(); loss.backward(); optimizer.step()
                steps += 1
                row = {"step": steps, "epoch": epoch + 1, "loss": float(loss.detach().cpu()), "l1_loss": float(pieces["l1_loss"]), "seconds": time.perf_counter() - started}
                log.write(json.dumps(row) + "\n"); log.flush(); rows.append(row); print(json.dumps(row))
                if steps % checkpoint_every == 0:
                    torch.save({"state_dict": policy.state_dict(), "architecture": "LeRobot ACT", "step": steps}, out / f"act-step-{steps:06d}.pt")
    final = out / "act-final.pt"
    torch.save({"state_dict": policy.state_dict(), "architecture": "LeRobot ACT", "step": steps}, final)
    summary = {"dataset": str(dataset), "frames": len(ds), "epochs": epochs, "steps": steps, "initial_loss": rows[0]["loss"] if rows else None, "final_loss": rows[-1]["loss"] if rows else None, "seconds": time.perf_counter() - started, "checkpoint": str(final), "log": str(log_path), "device": str(device), "model": "ACT"}
    (out / "training.json").write_text(json.dumps(summary, indent=2)); return summary


def load_policy(checkpoint: str | Path, device: str = "cpu"):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = ACTPolicy(act_config()).to(device); policy.load_state_dict(blob["state_dict"]); policy.eval(); return policy


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", type=Path, default=ROOT / "lerobot-dataset"); ap.add_argument("--out", type=Path, default=ROOT); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--checkpoint-every", type=int, default=25)
    a = ap.parse_args(); print(json.dumps(train(a.dataset, out=a.out, epochs=a.epochs, batch_size=a.batch_size, checkpoint_every=a.checkpoint_every), indent=2))


if __name__ == "__main__": main()
