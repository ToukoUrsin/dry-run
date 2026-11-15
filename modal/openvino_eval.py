"""Map a final exported ACT OpenVINO controller through the public rollout API."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("dryrun-openvino-policy-eval")
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
    .add_local_file("/bulk3/runs/dryrun/m4/hi224-policy-final/openvino/policy.xml", remote_path="/root/repo/model/policy.xml")
    .add_local_file("/bulk3/runs/dryrun/m4/hi224-policy-final/openvino/policy.bin", remote_path="/root/repo/model/policy.bin")
)


def _counts(events):
    ids = {k: set() for k in ("reached", "grasped", "lifted", "presented", "dropped")}
    names = {"reach": "reached", "grasp": "grasped", "lift": "lifted", "present": "presented", "drop": "dropped"}
    for event in events or []:
        stage = names.get(str(event.get("event"))) if isinstance(event, dict) else None
        if stage is None or event.get("brick") is None:
            continue
        try:
            ids[stage].add(int(event["brick"]))
        except (TypeError, ValueError):
            pass
    return {k: len(v) for k, v in ids.items()}


class _OpenVINOController:
    """Local copy of the inference-only boundary; avoids importing LeRobot."""
    def __init__(self, model_path: str, max_delta: float = 0.06):
        import openvino as ov
        self.compiled = ov.Core().compile_model(model_path, "CPU")
        self.request = self.compiled.create_infer_request()
        self.max_delta = float(max_delta)
        self.image_size = int(self.compiled.input("top").shape[-1])

    def act(self, obs):
        import numpy as np
        import torch
        def tensor(frame):
            x = torch.from_numpy(np.asarray(frame).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            if x.shape[-2:] != (self.image_size, self.image_size):
                x = torch.nn.functional.interpolate(x, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
            return x.numpy()
        state = np.r_[obs["joints"], obs.get("verdict", [0])].astype(np.float32)[None]
        target = np.asarray(self.request.infer({"top": tensor(obs["top"]), "inspection": tensor(obs["inspection"]), "state": state})[self.compiled.output(0)]).reshape(6)
        current = np.asarray(obs["joints"], dtype=np.float32).reshape(6)
        return np.clip(target, current - self.max_delta, current + self.max_delta)


@app.function(image=image, timeout=1800, cpu=4, retries=2)
def evaluate_seed(seed: int, tier: str, max_steps: int, render_size: int, run_id: str, source_commit: str):
    import sys
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from dryrun.inspector import Inspector
    from dryrun.rollout import run

    controller = _OpenVINOController("/root/repo/model/policy.xml", max_delta=0.06)
    result = run(int(seed), controller, inspector=Inspector("/root/repo/model/dryrun_inspector.xml", threads=1), tier=tier, max_steps=int(max_steps), render_size=int(render_size))
    stages = _counts(result.get("events", []))
    return {
        "format": "dryrun-openvino-policy-eval-row-v1", "run_id": run_id, "source_commit": source_commit,
        "seed": int(seed), "tier": tier, "bricks": int(result.get("bricks", 0)),
        "successes": int(result.get("successes", 0)), "score": float(result.get("score", 0.0)),
        "stages": {**stages, "right_bowl": int(result.get("successes", 0))},
        "max_steps": int(max_steps), "render_size": int(render_size), "max_delta": 0.06,
    }


@app.local_entrypoint()
def main(seeds: str = "0,1,2", tier: str = "loose", max_steps: int = 2600, render_size: int = 320, run_id: str = "", out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/openvino-policy"):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = list(evaluate_seed.map(values, kwargs={"tier": tier, "max_steps": int(max_steps), "render_size": int(render_size), "run_id": run_id, "source_commit": source}))
    rows.sort(key=lambda x: x["seed"])
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "dryrun-modal-openvino-policy-map-v1", "run_id": run_id, "tier": tier, "source_commit": source, "checkpoint": "/bulk3/runs/dryrun/m4/hi224-policy-final/openvino/policy.xml", "max_steps": int(max_steps), "render_size": int(render_size), "max_delta": 0.06, "rows": rows}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
