"""Live custom-defect inspection and one-brick rollout.

The request renders the user's mark through the real inspection camera, then
runs the one-brick SO-101 MuJoCo rollout with the same replay exporter. The
compressed replay is returned inline for a scale-to-zero worker.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .defects import render_drawing_frame
from .inspector import Inspector
from .replay_export import export


def _image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def run_custom_defect(payload: dict, out_root: str | Path | None = None) -> dict:
    """Run inspection plus a real one-brick rollout for a drawing request."""
    drawing = str(payload.get("drawing", ""))
    if not drawing.startswith("data:image") or "," not in drawing:
        raise ValueError("drawing must be a PNG data URI")
    try:
        raw = base64.b64decode(drawing.split(",", 1)[1], validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"invalid drawing image: {exc}") from exc
    if image.width < 2 or image.height < 2:
        raise ValueError("drawing image is empty")
    seed = int(payload.get("seed", 0))
    frame = render_drawing_frame(np.asarray(image), seed=seed)
    decision = Inspector().inspect(frame)
    heatmap = Image.fromarray(decision["heatmap"], mode="L").convert("RGB")

    root = Path(out_root or os.environ.get("DRYRUN_LIVE_DIR", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    run_id = "custom-" + uuid.uuid4().hex[:12]
    out = root / run_id
    # One brick bounds on-demand work while exercising the same contact-only
    # expert and compressed replay contract as held-out exports.
    manifest = export(seed, out, checkpoint="live-custom", count=1)
    replay_path = Path(manifest["replay"])
    with gzip.open(replay_path, "rt", encoding="utf-8") as fh:
        replay = json.load(fh)
    replay["custom_defect"] = {
        "verdict": decision["verdict"],
        "score": float(decision["score"]),
        "threshold": float(decision["threshold"]),
        "drawing_sha256": hashlib.sha256(raw).hexdigest(),
    }
    replay["events"].append({
        "t": float(replay["frames"][-1]["t"] if replay.get("frames") else 0),
        "name": "custom_inspector",
        "frame": max(0, len(replay.get("frames", [])) - 1),
        "verdict": decision["verdict"],
        "score": float(decision["score"]),
    })
    with gzip.open(replay_path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(replay, fh, separators=(",", ":"))
    manifest["replay_bytes"] = replay_path.stat().st_size
    manifest["run_id"] = run_id
    manifest["inspector"] = {
        "verdict": decision["verdict"],
        "score": float(decision["score"]),
        "threshold": float(decision["threshold"]),
        "heatmap": _image_data_uri(heatmap),
    }
    manifest["replay_gzip_base64"] = base64.b64encode(replay_path.read_bytes()).decode("ascii")
    (out / "live-response.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "replay_gzip_base64"}, indent=2))
    return manifest


__all__ = ["run_custom_defect"]
