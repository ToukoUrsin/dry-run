"""Modal scale-to-zero boundary for the Dry Run inspection camera.

The function accepts the same PNG data URI that the page sends to
``/api/defect``.  The model is mounted from the ``dryrun-demos`` volume so
the image stays small and model replacement does not require rebuilding the
container.  The response is JSON-safe: rendered frame and heat map are PNG
data URIs, with the score, threshold, verdict and measured latency alongside
them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

import modal

# Modal workers have no X11 display; MuJoCo's software renderer is required
# before the defect renderer imports the simulator.
os.environ.setdefault("MUJOCO_GL", "osmesa")


app = modal.App("dryrun-inspector")
volume = modal.Volume.from_name("dryrun-demos", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
    .pip_install("mujoco", "numpy", "opencv-python-headless", "openvino", "pillow", "imageio", "imageio-ffmpeg")
    .add_local_dir("dryrun", remote_path="/root/repo/dryrun")
    .add_local_dir("assets", remote_path="/root/repo/assets")
)


@app.function(image=image, timeout=30, retries=1, cpu=2, volumes={"/runs": volume})
def draw_inspector(
    payload: Mapping[str, Any],
    model_path: str = "/runs/m2_rebuild/default_model/openvino/weights/openvino/dryrun_inspector.xml",
) -> dict[str, Any]:
    """Inspect a page drawing and return render, heat map and verdict.

    ``payload`` must contain a PNG data URI under ``drawing`` and may include
    ``seed``, ``size`` and ``color``.  ``rollout`` is deliberately ignored by
    this fast function; the page can request a real rollout separately after
    showing this result.
    """
    import sys

    # The model may have been uploaded after this app object was first
    # created; refresh the mounted volume before resolving the default path.
    volume.reload()
    sys.path.insert(0, "/root/repo")
    from dryrun.live_defect import inspect_custom_defect, inspection_json

    chosen = str(payload.get("model_path") or model_path or os.environ.get("DRYRUN_INSPECTOR_MODEL", ""))
    result = inspect_custom_defect(payload, model_path=chosen or None, threads=1)
    output = inspection_json(result)
    output["schema"] = "dryrun.inspector.draw.v1"
    output["model"] = chosen
    return output


@app.local_entrypoint()
def main(
    drawing: str,
    model_path: str = "/runs/m2_rebuild/default_model/openvino/weights/openvino/dryrun_inspector.xml",
) -> None:
    response = draw_inspector.remote({"drawing": drawing}, model_path=model_path)
    print(json.dumps(response, indent=2))


__all__ = ["app", "draw_inspector"]
