"""Train a compact LeRobot ACT policy and emit resumable checkpoints."""
from __future__ import annotations
from pathlib import Path
import argparse, json, time
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType

ROOT = Path("/bulk3/runs/dryrun/m4")


def act_config(chunk_size: int = 30, image_size: int = 64) -> ACTConfig:
    image_size = int(image_size)
    features = {
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, image_size, image_size)),
        "observation.images.inspection": PolicyFeature(FeatureType.VISUAL, (3, image_size, image_size)),
        "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
    }
    return ACTConfig(
        input_features=features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        chunk_size=int(chunk_size), n_action_steps=1, use_vae=False,
        pretrained_backbone_weights=None, dim_model=64, n_heads=4,
        n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=128,
        n_vae_encoder_layers=1, optimizer_lr=2e-4, push_to_hub=False,
    )


def _batch(rows, device, chunk_size: int = 30):
    # LeRobot returns images as CHW float tensors in [0,1].  Add the verdict
    # channel to the six measured joints; this is the inspector interface.
    top = torch.stack([r["observation.images.top"] for r in rows]).float()
    inspection = torch.stack([r["observation.images.inspection"] for r in rows]).float()
    state = torch.stack([r["observation.state"] for r in rows]).float()
    verdict = torch.stack([r["inspection.verdict"] for r in rows]).float().reshape(len(rows), 1)
    state = torch.cat([state, verdict], dim=1)
    # ``LeRobotDataset(delta_timestamps=...)`` supplies the next k targets
    # for the post-step observation at t.  Preserve episode-end padding so
    # ACT never learns to repeat the final target across a new episode.
    action = torch.stack([r["action"] for r in rows]).float()
    if action.ndim == 2:
        raise ValueError("dataset must provide future action windows; pass delta_timestamps")
    action_is_pad = torch.stack([r["action_is_pad"] for r in rows]).bool()
    return {"observation.images.top": top.to(device), "observation.images.inspection": inspection.to(device),
            "observation.state": state.to(device), "action": action.to(device),
            "action_is_pad": action_is_pad.to(device)}


def train(dataset: str | Path, *, out: Path = ROOT, epochs: int = 2, batch_size: int = 8, checkpoint_every: int = 25, chunk_size: int = 30, image_size: int = 64):
    out.mkdir(parents=True, exist_ok=True); log_path = out / "training.jsonl"
    # Rows contain the observation after action t.  Train it to predict the
    # targets applied on ticks t+1 ... t+k, never the action that produced the
    # stored observation and never targets from the next episode.
    fps = 30.0
    ds = LeRobotDataset(
        "dryrun-so101", root=dataset, download_videos=False,
        delta_timestamps={"action": [(i + 1) / fps for i in range(int(chunk_size))]},
    )
    cfg = act_config(chunk_size=chunk_size, image_size=image_size); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy(cfg).to(device); policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr)
    steps = 0; rows = []; started = time.perf_counter()
    with log_path.open("w") as log:
        for epoch in range(epochs):
            order = torch.randperm(len(ds)).tolist()
            for start in range(0, len(order), batch_size):
                examples = [ds[i] for i in order[start:start + batch_size]]
                if not examples: continue
                loss, pieces = policy(_batch(examples, device, cfg.chunk_size)); optimizer.zero_grad(); loss.backward(); optimizer.step()
                steps += 1
                row = {"step": steps, "epoch": epoch + 1, "loss": float(loss.detach().cpu()), "l1_loss": float(pieces["l1_loss"]), "seconds": time.perf_counter() - started}
                log.write(json.dumps(row) + "\n"); log.flush(); rows.append(row); print(json.dumps(row))
                if steps % checkpoint_every == 0:
                    torch.save({"state_dict": policy.state_dict(), "architecture": "LeRobot ACT", "step": steps, "chunk_size": int(chunk_size), "image_size": int(image_size)}, out / f"act-step-{steps:06d}.pt")
    final = out / "act-final.pt"
    torch.save({"state_dict": policy.state_dict(), "architecture": "LeRobot ACT", "step": steps, "chunk_size": int(chunk_size), "image_size": int(image_size)}, final)
    summary = {"dataset": str(dataset), "frames": len(ds), "epochs": epochs, "steps": steps, "initial_loss": rows[0]["loss"] if rows else None, "final_loss": rows[-1]["loss"] if rows else None, "seconds": time.perf_counter() - started, "checkpoint": str(final), "log": str(log_path), "device": str(device), "model": "ACT", "chunk_size": int(chunk_size), "image_size": int(image_size)}
    (out / "training.json").write_text(json.dumps(summary, indent=2)); return summary


def load_policy(checkpoint: str | Path, device: str = "cpu"):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    chunk_size = blob.get("chunk_size")
    if chunk_size is None:
        # Early Modal trainers omitted metadata even though the decoder
        # position embedding still records the actual trained chunk length.
        # Infer it so those checkpoints remain evaluable after the contract
        # moved from fabricated 4-step chunks to real 30-step windows.
        pos = blob.get("state_dict", {}).get("model.decoder_pos_embed.weight")
        chunk_size = int(pos.shape[0]) if pos is not None and getattr(pos, "ndim", 0) >= 1 else 30
    policy = ACTPolicy(act_config(int(chunk_size), int(blob.get("image_size", 64)))).to(device)
    policy.load_state_dict(blob["state_dict"]); policy.eval(); return policy


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", type=Path, default=ROOT / "lerobot-dataset"); ap.add_argument("--out", type=Path, default=ROOT); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--checkpoint-every", type=int, default=25); ap.add_argument("--chunk-size", type=int, default=30); ap.add_argument("--image-size", type=int, default=64)
    a = ap.parse_args(); print(json.dumps(train(a.dataset, out=a.out, epochs=a.epochs, batch_size=a.batch_size, checkpoint_every=a.checkpoint_every, chunk_size=a.chunk_size, image_size=a.image_size), indent=2))


if __name__ == "__main__": main()
