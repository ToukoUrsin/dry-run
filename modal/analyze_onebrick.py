"""Inspect one-brick shard wrist-roll choices before building a fixed-answer set."""
from __future__ import annotations
import json
import modal

app = modal.App("dryrun-onebrick-analysis")
shards = modal.Dict.from_name("dryrun-demo-shards", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")

@app.function(image=image, timeout=600)
def analyze(keys: list[str]):
    import numpy as np
    counts = {"positive": 0, "negative": 0, "zero": 0}
    values = []
    episodes = 0
    frames = 0
    for key in keys:
        blob = shards.get(key)
        if blob is None:
            continue
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8")
        payload = json.loads(blob)
        for cycle in payload.get("cycles", []):
            if not cycle.get("success"):
                continue
            episodes += 1
            acts = []
            for frame in cycle.get("frames", []):
                a = frame.get("action")
                if isinstance(a, list) and len(a) >= 5:
                    acts.append(float(a[4]))
            if not acts:
                continue
            frames += len(acts)
            # The fixed jaw answer is the terminal wrist-roll target; use the
            # median of the last 10% of actions to suppress approach motion.
            tail = acts[max(0, len(acts) - max(1, len(acts)//10)):]
            v = float(np.median(tail))
            values.append(v)
            if v > 1e-3: counts["positive"] += 1
            elif v < -1e-3: counts["negative"] += 1
            else: counts["zero"] += 1
    return {"episodes": episodes, "frames": frames, "counts": counts, "values": values}

@app.local_entrypoint()
def main(prefixes: str = "onebrick-600-20261115-1,onebrick-1400-20261115-1", starts: str = "11000,12000", ends: str = "11600,13400"):
    ps = prefixes.split(","); ss = [int(x) for x in starts.split(",")]; ee = [int(x) for x in ends.split(",")]
    keys = [f"{p}/loose/{seed:04d}" for p, start, end in zip(ps, ss, ee) for seed in range(start, end)]
    print(json.dumps(analyze.remote(keys), indent=2))
