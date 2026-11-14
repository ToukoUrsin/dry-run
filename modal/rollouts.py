"""Fan out real Dry Run rollouts on Modal with ``.map``.

This is the scalable path for once the jaw-pad fix lands: each seed runs the
same public MuJoCo rollout and writes only successful-cycle records to the
``dryrun-demos`` volume. The local entrypoint is intentionally small so a
thousand-seed run is a one-line change to ``seeds``.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import modal

app = modal.App("dryrun-rollouts")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
         .pip_install("mujoco", "numpy", "imageio", "imageio-ffmpeg", "lerobot[dataset]")
         .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
         .add_local_dir("assets", remote_path="/root/repo/assets")
         .add_local_file("data/reach_map_polar.json", remote_path="/root/repo/data/reach_map_polar.json"))


class RawRecorder:
    def __init__(self, seed: int):
        self.seed = seed; self.frames = []; self.success = False; self.open = False
    def cycle_start(self, brick=None):
        self.open = True; self.frames = []
    def cycle_end(self, success: bool):
        self.success = bool(success); self.open = False
    def record(self, row):
        if row.get("type") == "cycle_start": self.cycle_start(row.get("brick"))
        elif row.get("type") == "cycle_end": self.cycle_end(bool(row.get("success")))
        elif self.open and "observation" in row:
            obs = row["observation"]
            self.frames.append({"t": row.get("t", 0), "state": obs["joints"].tolist(), "action": row["action"].tolist(), "verdict": int(row.get("verdict", 0))})


@app.function(image=image, timeout=900, retries=2, volumes={"/runs": volume}, cpu=4)
def rollout_seed(seed: int, tier: str = "loose", max_steps: int = 1800, max_cycles: int | None = None):
    sys.path.insert(0, "/root/repo")
    import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
    from dryrun.rollout import run
    recorder = RawRecorder(int(seed)); result = run(int(seed), "expert", tier=tier, record=recorder, max_steps=int(max_steps), max_cycles=max_cycles)
    path = Path(f"/runs/raw-{tier}-{int(seed):04d}.json")
    payload = {"seed": int(seed), "tier": tier, "max_steps": int(max_steps), "max_cycles": max_cycles, "result": result, "successful_cycle_frames": recorder.frames if recorder.success else [], "source": "modal-map"}
    path.write_text(json.dumps(payload)); volume.commit(); return {"seed": int(seed), "tier": tier, "max_steps": int(max_steps), "max_cycles": max_cycles, "bricks": result["bricks"], "successes": result["successes"], "score": result["score"], "raw": str(path)}


@app.local_entrypoint()
def main(seeds: str = "0,1,2,3,4,5", tier: str = "loose", max_steps: int = 1800, max_cycles: int | None = None):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    rows = list(rollout_seed.map(values, kwargs={"tier": tier, "max_steps": int(max_steps), "max_cycles": max_cycles})); print(json.dumps({"mapped": len(rows), "tier": tier, "max_steps": int(max_steps), "max_cycles": max_cycles, "rows": rows}, indent=2))
