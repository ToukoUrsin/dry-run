"""Modal ``.map`` evaluator for policy seeds."""
from __future__ import annotations
import json, sys
import modal

app = modal.App("dryrun-policy-eval")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
         .pip_install("mujoco", "numpy", "torch", "torchvision", "imageio", "lerobot[dataset]")
         .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
         .add_local_dir("assets", remote_path="/root/repo/assets")
         .add_local_file("data/reach_map_polar.json", remote_path="/root/repo/data/reach_map_polar.json"))


@app.function(image=image, timeout=1800, cpu=4, volumes={"/runs": volume})
def evaluate_seed(seed: int, tier: str = "loose", checkpoint: str = "/runs/act-final-modal.pt", max_steps: int = 1800):
    sys.path.insert(0, "/root/repo")
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from dryrun.policy_benchmark import ACTController
    from dryrun.rollout import run
    controller = ACTController(checkpoint, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    result = run(int(seed), controller, tier=tier, max_steps=max_steps)
    return {"seed": int(seed), "tier": tier, "checkpoint": checkpoint, "max_steps": max_steps, "bricks": result["bricks"], "successes": result["successes"], "score": result["score"], "source": "modal-map"}


@app.local_entrypoint()
def main(seeds: str = "0,1,2", tier: str = "loose", checkpoint: str = "/runs/act-final-modal.pt", max_steps: int = 60):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    rows = list(evaluate_seed.map(values, kwargs={"tier": tier, "checkpoint": checkpoint, "max_steps": max_steps})); print(json.dumps({"mapped": len(rows), "tier": tier, "rows": rows}, indent=2))
