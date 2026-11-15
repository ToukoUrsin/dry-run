"""Build the fixed-answer one-brick LeRobot dataset from Modal Dict shards."""
from __future__ import annotations
import base64, io, json, subprocess, sys
from pathlib import Path
import modal

app = modal.App("dryrun-onebrick-merge")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
shards = modal.Dict.from_name("dryrun-demo-shards", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("lerobot[dataset]==0.7.0", "numpy", "pillow", "opencv-python-headless")
         .add_local_dir("dryrun", remote_path="/root/repo/dryrun"))

@app.function(image=image, timeout=7200, cpu=4, volumes={"/runs": volume})
def merge(sources: list[tuple[str, int, int]], dataset_path: str, limit: int = 600):
    sys.path.insert(0, "/root/repo")
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from dryrun.demos import FEATURES
    root = Path(dataset_path)
    if root.exists():
        raise FileExistsError(f"dataset path already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create("dryrun-so101", fps=30, features=FEATURES, root=root,
                               robot_type="so101", use_videos=False,
                               image_writer_processes=0, image_writer_threads=0)
    episodes = frames = 0
    selected = []
    def decode(frame, name):
        encoded = frame.get(name)
        if not encoded:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
    for prefix, start, end in sources:
        for seed in range(int(start), int(end)):
            if episodes >= int(limit):
                break
            blob = shards.get(f"{prefix}/loose/{seed:04d}")
            if blob is None:
                continue
            if isinstance(blob, bytes): blob = blob.decode("utf-8")
            payload = json.loads(blob)
            for cycle in payload.get("cycles", []):
                if episodes >= int(limit): break
                if not cycle.get("success") or not cycle.get("frames"): continue
                for frame in cycle["frames"]:
                    ds.add_frame({
                        "observation.images.top": decode(frame, "top_jpeg"),
                        "observation.images.inspection": decode(frame, "inspection_jpeg"),
                        "observation.state": np.asarray(frame.get("state", [0]*6), dtype=np.float32),
                        "action": np.asarray(frame.get("action", [0]*6), dtype=np.float32),
                        "inspection.verdict": np.asarray([int(frame.get("verdict", 0))], dtype=np.int8),
                        "success": np.asarray([True], dtype=np.bool_),
                        "task": str(cycle.get("task", "sort brick")),
                    })
                    frames += 1
                ds.save_episode(); episodes += 1; selected.append(seed)
        if episodes >= int(limit): break
    ds.finalize(); volume.commit()
    return {"dataset": dataset_path, "episodes": episodes, "frames": frames,
            "limit": int(limit), "sources": sources, "selected_seeds": selected}

def _commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"

@app.local_entrypoint()
def main(sources: str = "onebrick-1400-20261115-1:12000:13400,onebrick-600-20261115-1:11000:11600",
         dataset_path: str = "/runs/onebrick-fixed-600-dataset-20261115-1",
         limit: int = 600, out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/onebrick-fixed-600"):
    parsed = []
    for item in sources.split(","):
        p, s, e = item.split(":"); parsed.append((p, int(s), int(e)))
    result = merge.remote(parsed, dataset_path, int(limit))
    target = Path(out_dir) / "onebrick-fixed-600-20261115-1"; target.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "dryrun-modal-lerobot-merge-v1", "source_commit": _commit(), **result}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
