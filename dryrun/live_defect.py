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
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .defects import render_drawing_frame
from .inspector import Inspector
from .replay_export import export


def _image_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_drawing(drawing: Any) -> tuple[bytes, Image.Image]:
    """Decode the page's PNG data URI (and reject ambiguous input early)."""
    if not isinstance(drawing, str) or not drawing.startswith("data:image") or "," not in drawing:
        raise ValueError("drawing must be a PNG data URI")
    try:
        raw = base64.b64decode(drawing.split(",", 1)[1], validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
        image.load()
    except Exception as exc:
        raise ValueError(f"invalid drawing image: {exc}") from exc
    if image.width < 2 or image.height < 2:
        raise ValueError("drawing image is empty")
    return raw, image


@lru_cache(maxsize=4)
def _cached_inspector(model_path: str | None, threshold: float | None, threads: int) -> Inspector:
    """Keep the OpenVINO graph warm across scale-to-zero requests.

    Loading the graph dominates the first request.  A small process-local cache
    means subsequent drawings only pay for the MuJoCo render and one inference,
    while still allowing a deployment to select a different model or threshold.
    """
    return Inspector(model_path=model_path, threshold=threshold, threads=threads)


def inspect_custom_defect(
    payload: Mapping[str, Any],
    *,
    model_path: str | Path | None = None,
    threshold: float | None = None,
    threads: int = 1,
) -> dict[str, Any]:
    """Render and inspect a drawn defect without starting a rollout.

    This is the fast Modal payload boundary.  ``rendered_rgb`` and ``heatmap``
    are RGB/gray ``uint8`` NumPy arrays for local callers; use
    :func:`inspection_json` when returning the result over HTTP.  The model is
    cached after the first request and the function reports its own wall-clock
    latency so the page can show the measured value.
    """
    started = time.perf_counter()
    raw, image = _decode_drawing(payload.get("drawing"))
    seed = int(payload.get("seed", 0))
    size = str(payload.get("size", "2x2"))
    color = str(payload.get("color", "red"))
    frame = render_drawing_frame(np.asarray(image), seed=seed, size=size, color=color)
    decision = _cached_inspector(
        None if model_path is None else str(model_path), threshold, int(threads)
    ).inspect(frame)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "rendered_rgb": np.asarray(frame, dtype=np.uint8),
        "score": float(decision["score"]),
        "threshold": float(decision["threshold"]),
        "heatmap": np.asarray(decision["heatmap"], dtype=np.uint8),
        "verdict": str(decision["verdict"]),
        "latency_ms": float(elapsed_ms),
        "seed": seed,
        "drawing_sha256": hashlib.sha256(raw).hexdigest(),
    }


# Names used by the page/Modal integration in earlier session notes.  Keep
# these aliases so callers do not need to know whether they are local or remote.
draw_defect = inspect_custom_defect
inspect_drawing = inspect_custom_defect


def inspection_json(result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert :func:`inspect_custom_defect` output into HTTP-safe JSON."""
    rendered = Image.fromarray(np.asarray(result["rendered_rgb"], dtype=np.uint8), mode="RGB")
    heat = Image.fromarray(np.asarray(result["heatmap"], dtype=np.uint8), mode="L").convert("RGB")
    return {
        "rendered_rgb": _image_data_uri(rendered),
        "heatmap": _image_data_uri(heat),
        "score": float(result["score"]),
        "threshold": float(result["threshold"]),
        "verdict": str(result["verdict"]),
        "latency_ms": float(result.get("latency_ms", 0.0)),
        "seed": int(result.get("seed", 0)),
        "drawing_sha256": str(result.get("drawing_sha256", "")),
        "frame_shape": list(np.asarray(result["rendered_rgb"]).shape),
        "heatmap_shape": list(np.asarray(result["heatmap"]).shape),
    }


def run_custom_defect(payload: dict, out_root: str | Path | None = None) -> dict:
    """Run inspection plus a real one-brick rollout for a drawing request.

    The quick inspection fields are repeated at the top level for the page;
    the nested ``inspector`` object is retained for existing replay consumers.
    """
    raw, _image = _decode_drawing(payload.get("drawing"))
    inspected = inspect_custom_defect(payload)
    seed = int(inspected["seed"])
    decision = inspected
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
    # HTTP-safe fast-path fields.  Arrays remain available from
    # ``inspect_custom_defect`` for local callers.
    manifest.update(inspection_json(inspected))
    (out / "live-response.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "replay_gzip_base64"}, indent=2))
    return manifest


__all__ = [
    "draw_defect",
    "inspect_custom_defect",
    "inspect_drawing",
    "inspection_json",
    "run_custom_defect",
]
