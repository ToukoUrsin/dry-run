"""Two simulator DAgger rounds with expert action labels in reached states."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .demos import FEATURES
from .env import StationEnv
from .policy_benchmark import ACTController
from .train_policy import train


def append_round(source: Path, target: Path, checkpoint: Path, round_no: int, seeds: int = 2):
    if target.exists(): shutil.rmtree(target)
    ds = LeRobotDataset("dryrun-so101", root=source, download_videos=False)
    out = LeRobotDataset.create("dryrun-so101", fps=30, features=FEATURES, root=target, robot_type="so101", use_videos=False, image_writer_processes=0, image_writer_threads=0)
    for i in range(len(ds)):
        row = ds[i]; out.add_frame({"observation.images.top": (row["observation.images.top"].permute(1,2,0).numpy()*255).astype(np.uint8), "observation.images.inspection": (row["observation.images.inspection"].permute(1,2,0).numpy()*255).astype(np.uint8), "observation.state": row["observation.state"].numpy(), "action": row["action"].numpy(), "inspection.verdict": row["inspection.verdict"].numpy().astype(np.int8).reshape(1), "success": row["success"].numpy().astype(bool).reshape(1), "task": "sort brick"})
        if (i + 1) % 320 == 0: out.save_episode()
    policy = ACTController(checkpoint)
    added = 0
    for seed in range(seeds):
        env = StationEnv(seed, "loose"); obs = env.reset(); rows = []
        for _ in range(30):
            policy_action = policy.act({**obs, "verdict": np.asarray([0], np.float32)}); expert_action = env.expert_action(); rows.append({"observation.images.top": obs["top"], "observation.images.inspection": obs["inspection"], "observation.state": obs["joints"], "action": expert_action, "inspection.verdict": np.asarray([0], np.int8), "success": np.asarray([False]), "task": f"dagger round {round_no}"}); obs, _, done, _ = env.step(policy_action); added += 1
            if done: break
        for row in rows: out.add_frame(row)
        out.save_episode(); env.close()
    out.finalize(); return {"source": str(source), "target": str(target), "round": round_no, "expert_labeled_frames": added, "frames": len(LeRobotDataset("dryrun-so101", root=target, download_videos=False))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", type=Path, default=Path("/bulk2/runs/dryrun/m4/lerobot-real")); ap.add_argument("--checkpoint", type=Path, default=Path("/bulk2/runs/dryrun/m4/act-final.pt")); ap.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m4/dagger")); ap.add_argument("--seeds", type=int, default=2); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); rows = []
    source = a.dataset
    for round_no in (1, 2):
        target = a.out / f"round-{round_no}"; row = append_round(source, target, a.checkpoint, round_no, a.seeds); rows.append(row); train(target, out=a.out / f"train-{round_no}", epochs=1, batch_size=8, checkpoint_every=25); source = target
    (a.out / "rounds.json").write_text(json.dumps(rows, indent=2)); print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
