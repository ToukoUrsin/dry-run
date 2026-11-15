"""Scale-to-zero HTTP endpoint for the page's custom-defect action.

Deploy with the Junction Modal credentials:

  hsec exec --only JUNCTION_MODAL_TOKEN_ID --only JUNCTION_MODAL_TOKEN_SECRET -- bash -c \
    'MODAL_TOKEN_ID=$JUNCTION_MODAL_TOKEN_ID MODAL_TOKEN_SECRET=$JUNCTION_MODAL_TOKEN_SECRET modal deploy modal_live_defect.py'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal


APP_ROOT = Path(__file__).resolve().parent
# The real-present-frame model is the shipped M2 artifact.  Keep the local
# source on /bulk3 because /bulk2 is full; Modal bundles these three files
# into the image at deploy time, so the endpoint does not depend on a mounted
# host path after deployment.
MODEL_ROOT = Path(os.environ.get(
    "DRYRUN_MODAL_MODEL_ROOT",
    "/bulk3/runs/dryrun/m2_rebuild/default_model/openvino/weights/openvino",
))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libegl1", "ffmpeg")
    .pip_install(
        "mujoco>=3.2",
        "numpy>=1.26",
        "pillow>=10",
        "imageio>=2.31",
        "imageio-ffmpeg>=0.4",
        "opencv-python-headless>=4.10",
        "openvino>=2026.0",
        "fastapi[standard]>=0.115",
    )
    .add_local_dir(APP_ROOT / "dryrun", remote_path="/root/dryrun")
    .add_local_dir(APP_ROOT / "assets", remote_path="/root/assets")
    .add_local_file(MODEL_ROOT / "dryrun_inspector.xml", remote_path="/root/model/dryrun_inspector.xml")
    .add_local_file(MODEL_ROOT / "dryrun_inspector.bin", remote_path="/root/model/dryrun_inspector.bin")
    .add_local_file(MODEL_ROOT / "threshold.json", remote_path="/root/model/threshold.json")
)

app = modal.App("dryrun-live-defect")


@app.function(image=image, timeout=900, scaledown_window=60)
@modal.fastapi_endpoint(method="POST")
def live_defect(payload: dict):
    """Inspect a user drawing; optionally append a one-brick MuJoCo replay.

    The default request is the fast draw/inspect path (rendered RGB, score,
    heat map and verdict).  Set ``{"rollout": true}`` when the page needs the
    recorded replay as well.  Keeping the rollout opt-in makes the inspector
    response scale-to-zero friendly and keeps local inspection below two
    seconds after the model has warmed.
    """
    sys.path.insert(0, "/root")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("DRYRUN_INSPECTOR_MODEL", "/root/model/dryrun_inspector.xml")
    from dryrun.live_defect import inspection_json, inspect_custom_defect, run_custom_defect

    if bool(payload.get("rollout", False)):
        return run_custom_defect(payload, out_root="/tmp/dryrun-live")
    return inspection_json(inspect_custom_defect(payload))


if __name__ == "__main__":
    print("Use `modal deploy modal_live_defect.py` to publish the endpoint.")
