"""Measure Nemotron versus random curriculum batches with real ACT policies.

The local command sends two equal-sized scene lists to one Modal worker.  The
worker records expert image/joint demonstrations for each list, initializes
both ACT policies from the same checkpoint, trains them on their respective
datasets, and evaluates both policies and the expert on the same held-out
scenes.  The result is a comparison artifact, never a benchmark replacement.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import modal


app = modal.App("dryrun-curriculum-compare")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
    .pip_install("mujoco", "numpy", "torch", "torchvision", "lerobot[dataset]==0.7.0", "opencv-python-headless", "pillow", "openvino")
    .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
    .add_local_dir("assets", remote_path="/root/repo/assets")
    .add_local_file("data/reach_map_polar.json", remote_path="/root/repo/data/reach_map_polar.json")
    .add_local_file(
        "/bulk3/runs/dryrun/m2_rebuild/models_prompt020/PaDiM/openvino/weights/openvino/dryrun_inspector.xml",
        remote_path="/root/repo/model/dryrun_inspector.xml",
    )
    .add_local_file(
        "/bulk3/runs/dryrun/m2_rebuild/models_prompt020/PaDiM/openvino/weights/openvino/dryrun_inspector.bin",
        remote_path="/root/repo/model/dryrun_inspector.bin",
    )
    .add_local_file(
        "/bulk3/runs/dryrun/m2_rebuild/models_prompt020/PaDiM/openvino/weights/openvino/threshold.json",
        remote_path="/root/repo/model/threshold.json",
    )
)


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _features(size: int = 64) -> dict[str, dict[str, Any]]:
    return {
        "observation.images.top": {"dtype": "image", "shape": (size, size, 3)},
        "observation.images.inspection": {"dtype": "image", "shape": (size, size, 3)},
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
        "inspection.verdict": {"dtype": "int8", "shape": (1,)},
        "success": {"dtype": "bool", "shape": (1,)},
    }


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/runs": volume})
def compare(
    nemotron_batch: list[dict[str, Any]],
    random_batch: list[dict[str, Any]],
    run_id: str,
    heldout_seeds: list[int] | None = None,
    epochs: int = 2,
) -> dict[str, Any]:
    import base64
    import io
    import os
    import sys

    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import numpy as np
    import torch
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from dryrun.inspector import Inspector
    from dryrun.mjcf_station import scene_specs_from_dict
    from dryrun.policy_benchmark import ACTController
    from dryrun.rollout import run
    from dryrun.train_policy import act_config

    torch.set_num_threads(4)
    heldout_seeds = [int(v) for v in (heldout_seeds or [0, 1, 2])]
    root = Path("/runs") / "curriculum-compare" / str(run_id)
    root.mkdir(parents=True, exist_ok=True)

    class Recorder:
        """Keep only successful brick episodes from the public rollout API."""

        def __init__(self, image_size: int = 64):
            self.image_size = int(image_size)
            self.open: dict[str, Any] | None = None
            self.cycles: list[dict[str, Any]] = []
            self.seen: set[tuple[Any, ...]] = set()

        @staticmethod
        def _key(event: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(sorted((str(k), repr(v)) for k, v in event.items()))

        def record(self, payload: dict[str, Any]) -> None:
            if payload.get("type") == "cycle_start":
                self.open = None
                self.seen.clear()
                return
            if payload.get("type") == "cycle_end":
                if self.open is not None and self.open["frames"]:
                    self.cycles.append(self.open)
                self.open = None
                return
            if "observation" not in payload:
                return
            for event in payload.get("events", []) or []:
                if not isinstance(event, dict):
                    continue
                key = self._key(event)
                if key in self.seen:
                    continue
                self.seen.add(key)
                if event.get("event") == "reach" and event.get("brick") is not None:
                    if self.open is not None and self.open["frames"]:
                        self.cycles.append(self.open)
                    self.open = {"brick": int(event["brick"]), "frames": [], "success": False}
            if self.open is None:
                return
            obs = payload["observation"]
            def encode(value: Any) -> str:
                arr = np.asarray(value, dtype=np.uint8)
                image = Image.fromarray(arr).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                buf = io.BytesIO(); image.save(buf, format="JPEG", quality=78, optimize=True)
                return base64.b64encode(buf.getvalue()).decode("ascii")
            self.open["frames"].append({
                "top": encode(obs["top"]), "inspection": encode(obs["inspection"]),
                "state": np.asarray(obs["joints"], dtype=np.float32).tolist(),
                "action": np.asarray(payload.get("action", [0] * 6), dtype=np.float32).tolist(),
                "verdict": int(payload.get("verdict", 0)),
            })

        def finish(self, result: dict[str, Any]) -> None:
            if self.open is not None and self.open["frames"]:
                self.cycles.append(self.open)
            correct = {int(v) for v in result.get("correct", [])}
            for cycle in self.cycles:
                cycle["success"] = int(cycle["brick"]) in correct

    def decode(value: str) -> np.ndarray:
        image = Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def build_dataset(name: str, batch: list[dict[str, Any]]) -> dict[str, Any]:
        dataset_root = root / name
        dataset = LeRobotDataset.create(
            "dryrun-so101", fps=30, features=_features(64), root=dataset_root,
            robot_type="so101", use_videos=False, image_writer_processes=0,
            image_writer_threads=0,
        )
        inspector = Inspector("/root/repo/model/dryrun_inspector.xml", threads=1)
        scene_rows = []
        for index, scene in enumerate(batch):
            scene = dict(scene)
            seed = int(scene.pop("_demo_seed", 61000 + index))
            tier = str(scene.get("tier", "loose"))
            specs = scene_specs_from_dict(seed, scene)
            recorder = Recorder(64)
            result = run(seed, "expert", tier=tier, inspector=inspector, record=recorder,
                         max_steps=2200, render_size=64, scene_specs=specs)
            recorder.finish(result)
            successful = 0
            for cycle in recorder.cycles:
                if not cycle.get("success"):
                    continue
                frames = cycle.get("frames", [])
                if not frames:
                    continue
                for frame in frames:
                    dataset.add_frame({
                        "observation.images.top": decode(frame["top"]),
                        "observation.images.inspection": decode(frame["inspection"]),
                        "observation.state": np.asarray(frame["state"], dtype=np.float32),
                        "action": np.asarray(frame["action"], dtype=np.float32),
                        "inspection.verdict": np.asarray([frame["verdict"]], dtype=np.int8),
                        "success": np.asarray([True], dtype=np.bool_),
                        "task": f"sort {tier}",
                    })
                dataset.save_episode(); successful += 1
            scene_rows.append({"seed": seed, "tier": tier, "bricks": int(result.get("bricks", 0)),
                               "expert_successes": int(result.get("successes", 0)),
                               "recorded_successful_episodes": successful})
        dataset.finalize()
        marker = dataset_root / "_COMPLETE.json"
        marker.write_text(json.dumps({"name": name, "scenes": scene_rows}) + "\n")
        volume.commit()
        return {"root": str(dataset_root), "frames": len(dataset), "scenes": scene_rows}

    # Both policies start from one exact, persisted initial checkpoint.  It is
    # intentionally a fresh ACT initialization so neither branch sees the
    # other branch's demonstrations.
    torch.manual_seed(1337)
    cfg = act_config(chunk_size=30, image_size=64)
    base = ACTPolicy(cfg)
    base_path = root / "base-act.pt"
    torch.save({"state_dict": base.state_dict(), "architecture": "LeRobot ACT",
                "step": 0, "chunk_size": 30, "image_size": 64}, base_path)
    volume.commit()

    def train_one(name: str, dataset_info: dict[str, Any]) -> dict[str, Any]:
        ds = LeRobotDataset("dryrun-so101", root=dataset_info["root"], download_videos=False,
                            delta_timestamps={"action": [(i + 1) / 30.0 for i in range(30)]})
        policy = ACTPolicy(cfg)
        policy.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=False)["state_dict"])
        policy.train(); opt = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr)
        steps = 0; losses: list[float] = []; started = time.perf_counter()
        order = list(range(len(ds)))
        for epoch in range(int(epochs)):
            # Same order and seed for both branches; only scene data differs.
            generator = torch.Generator().manual_seed(9000 + epoch)
            order = torch.randperm(len(ds), generator=generator).tolist()
            for start in range(0, len(order), 32):
                rows = [ds[i] for i in order[start:start + 32]]
                if not rows:
                    continue
                top = torch.stack([r["observation.images.top"] for r in rows]).float()
                inspection = torch.stack([r["observation.images.inspection"] for r in rows]).float()
                state = torch.cat([torch.stack([r["observation.state"] for r in rows]).float(),
                                   torch.stack([r["inspection.verdict"] for r in rows]).float().reshape(len(rows), 1)], 1)
                action = torch.stack([r["action"] for r in rows]).float()
                mask = torch.stack([r["action_is_pad"] for r in rows]).bool()
                loss, _ = policy({"observation.images.top": top, "observation.images.inspection": inspection,
                                  "observation.state": state, "action": action, "action_is_pad": mask})
                opt.zero_grad(); loss.backward(); opt.step(); steps += 1
                losses.append(float(loss.detach()))
        out = root / f"{name}-act.pt"
        torch.save({"state_dict": policy.state_dict(), "architecture": "LeRobot ACT", "step": steps,
                    "chunk_size": 30, "image_size": 64, "base_checkpoint": str(base_path)}, out)
        volume.commit()
        return {"checkpoint": str(out), "steps": steps, "initial_loss": losses[0] if losses else None,
                "final_loss": losses[-1] if losses else None, "seconds": time.perf_counter() - started}

    datasets = {"nemotron": build_dataset("nemotron", nemotron_batch),
                "random": build_dataset("random", random_batch)}
    training = {name: train_one(name, datasets[name]) for name in ("nemotron", "random")}

    def score_policy(checkpoint: str, seed: int) -> dict[str, Any]:
        controller = ACTController(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu",
                                    max_delta=0.06)
        inspector = Inspector("/root/repo/model/dryrun_inspector.xml", threads=1)
        result = run(int(seed), controller, tier="loose", count=3, inspector=inspector,
                     max_steps=1600, render_size=64)
        return {"seed": int(seed), "bricks": int(result.get("bricks", 0)),
                "successes": int(result.get("successes", 0)), "score": float(result.get("score", 0.0))}

    policy_rows = {name: [score_policy(training[name]["checkpoint"], seed) for seed in heldout_seeds]
                   for name in ("nemotron", "random")}
    expert_rows = []
    for seed in heldout_seeds:
        result = run(int(seed), "expert", tier="loose", count=3,
                     inspector=Inspector("/root/repo/model/dryrun_inspector.xml", threads=1),
                     max_steps=1600, render_size=64)
        expert_rows.append({"seed": int(seed), "bricks": int(result.get("bricks", 0)),
                            "successes": int(result.get("successes", 0)), "score": float(result.get("score", 0.0))})

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        bricks = sum(int(row["bricks"]) for row in rows)
        successes = sum(int(row["successes"]) for row in rows)
        return {"scenes": len(rows), "bricks": bricks, "successes": successes,
                "score": successes / max(1, bricks), "rows": rows}

    result = {
        "format": "dryrun-policy-curriculum-compare-v1", "run_id": str(run_id),
        "source_commit": _commit(), "base_checkpoint": str(base_path),
        "batch_size": len(nemotron_batch), "epochs": int(epochs),
        "datasets": datasets, "training": training,
        "heldout_seeds": heldout_seeds,
        "expert": aggregate(expert_rows),
        "nemotron": aggregate(policy_rows["nemotron"]),
        "random": aggregate(policy_rows["random"]),
    }
    result["delta_nemotron_minus_random"] = result["nemotron"]["score"] - result["random"]["score"]
    result["helps"] = result["delta_nemotron_minus_random"] > 0
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    nemotron: str,
    random_batch: str,
    run_id: str = "curriculum-compare",
    heldout_seeds: str = "0,1,2",
    epochs: int = 2,
    out: str = "/bulk3/runs/dryrun/m4/scene-batch-compare.json",
):
    def load(path: str) -> list[dict[str, Any]]:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return rows
    n_batch, r_batch = load(nemotron), load(random_batch)
    if not n_batch or len(n_batch) != len(r_batch):
        raise ValueError("batches must be non-empty and equal-sized JSONL files")
    seeds = [int(v) for v in heldout_seeds.split(",") if v.strip()]
    result = compare.remote(n_batch, r_batch, run_id, seeds, int(epochs))
    target = Path(out); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


__all__ = ["app", "compare"]
