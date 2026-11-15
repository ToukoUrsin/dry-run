"""Map real Dry Run demo attempts on Modal.

Each mapped seed writes an independent, durable episode shard to a Modal Dict.
The shard contains the public rollout observations and rendered camera frames
for successful cycles; a later single-writer merge can turn those shards into
one LeRobot dataset. Keeping one key per mapped task avoids corrupting
LeRobot metadata when thousands of containers finish at once.

The local entrypoint writes a JSONL manifest under ``/bulk3/runs/dryrun``.
The ``--download`` flag materializes those Dict values there after the map
finishes.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal


app = modal.App("dryrun-rollouts")
shards = modal.Dict.from_name("dryrun-demo-shards", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libegl1", "libgles2", "libosmesa6")
    .pip_install("mujoco", "numpy", "imageio", "imageio-ffmpeg", "pillow", "opencv-python-headless", "openvino")
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


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    return value


def _jpeg(value: Any) -> str | None:
    """Encode one public camera frame compactly for a shard."""

    if value is None:
        return None
    try:
        import numpy as np
        from PIL import Image

        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3:
            return None
        buf = io.BytesIO()
        Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
            buf, format="JPEG", quality=78, optimize=True
        )
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        # Camera pixels are useful evidence, but a missing optional encoder
        # must not turn a physics rollout into a failed task.
        return None


class EpisodeRecorder:
    """Collect cycle-scoped public observations without simulator state access."""

    def __init__(self, seed: int, image_size: int = 64, crop_top: bool = False):
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.crop_top = bool(crop_top)
        self.cycles: list[dict[str, Any]] = []
        self._open: dict[str, Any] | None = None
        self._seen_events: set[tuple[Any, ...]] = set()
        self._scene_success = False
        self._finalized = False

    def cycle_start(self, brick: str | None = None, task: str = "sort brick") -> None:
        # The current rollout API emits one scene-level cycle_start.  Keep
        # that marker as an empty holding state; actual episodes begin at the
        # first physics ``reach`` event.  A future API may emit one start per
        # brick, in which case the previous brick is flushed here.
        self._finish_open()
        self._seen_events.clear()
        self._scene_success = False
        self._finalized = False
        self._open = None if brick is None else {
            "brick": int(brick), "task": str(task), "frames": [], "success": False,
        }

    def cycle_end(self, success: bool) -> None:
        self._scene_success = bool(success)
        self._finish_open()

    def _finish_open(self) -> None:
        """Move the current brick attempt to ``cycles`` if it has frames."""
        if self._open is None:
            return
        if self._open.get("brick") is not None and self._open.get("frames"):
            self.cycles.append(self._open)
        self._open = None

    @staticmethod
    def _event_key(event: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(sorted((str(k), repr(v)) for k, v in event.items()))

    def _start_brick(self, brick: int, task: str = "sort brick") -> None:
        brick = int(brick)
        if self._open is not None and self._open.get("brick") == brick:
            return
        self._finish_open()
        self._open = {
            "brick": brick,
            "task": f"{task} ({brick})",
            "frames": [],
            "success": False,
        }

    def _consume_events(self, events: Any) -> None:
        for event in events or []:
            if not isinstance(event, dict):
                continue
            key = self._event_key(event)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            if event.get("event") == "reach" and event.get("brick") is not None:
                self._start_brick(int(event["brick"]), task="sort brick")

    @staticmethod
    def _has_drop(cycle: dict[str, Any]) -> bool:
        brick = cycle.get("brick")
        for frame in cycle.get("frames", []):
            for event in frame.get("events", []) or []:
                if isinstance(event, dict) and event.get("event") == "drop":
                    if event.get("brick") is None or int(event["brick"]) == int(brick):
                        return True
        return False

    def finalize_result(self, result: dict[str, Any]) -> None:
        """Assign per-brick success after the scene result is available.

        ``rollout.run`` emits a scene-level cycle_end before it computes the
        final result.  The only authoritative per-brick labels are therefore
        ``result['correct']``; a matching drop event also prevents a failed
        retry from becoming a training episode when the same brick is later
        sorted successfully.
        """
        self._finish_open()
        correct = {int(x) for x in result.get("correct", [])}
        for cycle in self.cycles:
            brick = cycle.get("brick")
            cycle["success"] = (
                brick is not None
                and int(brick) in correct
                and self._has_drop(cycle)
            )
        self._finalized = True

    def record(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "cycle_start":
            self.cycle_start(payload.get("brick"), str(payload.get("task", "sort brick")))
            return
        if kind == "cycle_end":
            self.cycle_end(bool(payload.get("success", False)))
            return
        if "observation" not in payload:
            return
        self._consume_events(payload.get("events", []))
        # Do not put the scene's initial neutral/hover frames in an episode:
        # they precede the first brick-level reach event.
        if self._open is None or self._open.get("brick") is None:
            return
        obs = payload["observation"]
        top = obs.get("top")
        inspection = obs.get("inspection")
        if top is not None or inspection is not None:
            from dryrun.policy_view import crop_work_area, resize_rgb
            if top is not None:
                top = crop_work_area(top, self.image_size) if self.crop_top else resize_rgb(top, self.image_size)
            if inspection is not None:
                inspection = resize_rgb(inspection, self.image_size)
        self._open["frames"].append(
            {
                "t": float(payload.get("t", 0.0)),
                "step": int(payload.get("step", 0)),
                "state": _jsonable(obs.get("joints", [])),
                "action": _jsonable(payload.get("action", [])),
                "verdict": int(payload.get("verdict", 0)),
                "top_jpeg": _jpeg(top),
                "inspection_jpeg": _jpeg(inspection),
                "events": _jsonable(payload.get("events", [])),
            }
        )


@app.function(image=image, timeout=1800, retries=2, cpu=4)
def rollout_seed(
    seed: int,
    tier: str = "loose",
    count: int | None = None,
    max_steps: int = 3600,
    max_cycles: int | None = 1,
    run_id: str = "latest",
    inspector_model: str = "/root/repo/model/dryrun_inspector.xml",
    render_size: int = 64,
    image_size: int | None = None,
    crop_top: bool = False,
):
    """Run one expert seed and persist a shard before returning its summary."""

    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from dryrun.rollout import run
    from dryrun.inspector import Inspector

    target_image_size = int(image_size if image_size is not None else render_size)
    recorder = EpisodeRecorder(int(seed), image_size=target_image_size, crop_top=bool(crop_top))
    inspector = Inspector(inspector_model, threads=1)
    result = run(
        int(seed),
        "expert",
        tier=tier,
        count=None if count is None else int(count),
        inspector=inspector,
        record=recorder,
        max_steps=int(max_steps),
        max_cycles=max_cycles,
        render_size=int(render_size),
    )
    if recorder._open is not None:
        recorder.cycle_end(bool(result.get("successes", 0)))
    recorder.finalize_result(result)
    store_key = f"{run_id}/{tier}/{int(seed):04d}"
    payload = {
        "format": "dryrun-demo-shard-v1",
        "source": "modal-map",
        "run_id": str(run_id),
        "seed": int(seed),
        "tier": str(tier),
        "max_steps": int(max_steps),
        "max_cycles": max_cycles,
        "render_size": int(render_size),
        "image_size": target_image_size,
        "crop_top": bool(crop_top),
        "result": _jsonable(result),
        "cycles": [c for c in recorder.cycles if c.get("success")],
        "cycles_seen": len(recorder.cycles),
    }
    # A shared Volume snapshot is not a safe fan-in primitive: concurrent
    # commits can race and lose files. Modal Dict provides atomic per-seed
    # puts, so the mapped workers never touch one another's artifact.
    shards.put(store_key, json.dumps(payload, separators=(",", ":")))
    return {
        "seed": int(seed),
        "tier": str(tier),
        "bricks": int(result.get("bricks", 0)),
        "successes": int(result.get("successes", 0)),
        "score": float(result.get("score", 0.0)),
        "cycles_seen": len(recorder.cycles),
        "successful_cycles": sum(bool(c.get("success")) for c in recorder.cycles),
        "store_key": store_key,
        "source": "modal-map",
    }


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(
    seeds: str = "0,1,2",
    tier: str = "loose",
    count: int = 0,
    max_steps: int = 3600,
    max_cycles: int = 1,
    run_id: str = "",
    out_dir: str = "/bulk3/runs/dryrun/m4/modal-map/demo-map",
    download: bool = False,
    inspector_model: str = "/root/repo/model/dryrun_inspector.xml",
    render_size: int = 64,
    image_size: int | None = None,
    crop_top: bool = False,
):
    values = [int(x) for x in seeds.split(",") if x.strip()]
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    dict_key_prefix = f"{run_id}/{tier}"
    rows = list(
        rollout_seed.map(
            values,
            kwargs={
                "tier": tier,
                "count": None if int(count) <= 0 else int(count),
                "max_steps": int(max_steps),
                "max_cycles": int(max_cycles) if max_cycles > 0 else None,
                "run_id": run_id,
                "inspector_model": inspector_model,
                "render_size": int(render_size),
                "image_size": None if image_size is None else int(image_size),
                "crop_top": bool(crop_top),
            },
        )
    )
    rows.sort(key=lambda row: row["seed"])
    target = Path(out_dir) / run_id
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "dryrun-modal-demo-map-v1",
        "source_commit": _commit(),
        "run_id": run_id,
        "tier": tier,
        "count": None if int(count) <= 0 else int(count),
        "max_steps": int(max_steps),
        "max_cycles": int(max_cycles) if max_cycles > 0 else None,
        "render_size": int(render_size),
        "image_size": None if image_size is None else int(image_size),
        "crop_top": bool(crop_top),
        "dict_name": "dryrun-demo-shards",
        "dict_key_prefix": dict_key_prefix,
        "rows": rows,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (target / "rows.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    if download:
        shard_dir = target / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            key = f"{run_id}/{tier}/{int(row['seed']):04d}"
            blob = shards.get(key)
            if blob is None:
                raise RuntimeError(f"mapped shard missing from Modal Dict: {key}")
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8")
            (shard_dir / f"seed-{int(row['seed']):04d}.json").write_text(str(blob) + "\n")
    print(json.dumps(manifest, indent=2))
