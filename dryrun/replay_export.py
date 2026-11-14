"""Export a compact replay from the public StationEnv rollout boundary.

The page receives recorded body poses and camera frames. It never rebuilds
the arm kinematics, and this exporter never drives a second legacy expert
path: every action goes through ``StationEnv.expert_action`` and
``StationEnv.step``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from .env import StationEnv, TIERS


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _quat(q: np.ndarray | list[float]) -> list[int]:
    return [int(max(-32767, min(32767, round(float(v) * 32767)))) for v in q]


def _scene(model: mujoco.MjModel) -> dict[str, Any]:
    bodies = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
        geoms = []
        start = int(model.body_geomadr[bid])
        end = start + int(model.body_geomnum[bid])
        for gid in range(start, end):
            typ = int(model.geom_type[gid])
            matid = int(model.geom_matid[gid])
            geom = {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}",
                "type": typ,
                "pos": [round(float(x), 6) for x in model.geom_pos[gid]],
                "quat": _quat(model.geom_quat[gid]),
                "size": [round(float(x), 6) for x in model.geom_size[gid]],
                "rgba": [round(float(x), 4) for x in model.mat_rgba[matid]] if matid >= 0 else [0.5, 0.5, 0.5, 1.0],
            }
            if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
                geom["mesh"] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[gid]))
            geoms.append(geom)
        bodies.append({"id": bid, "name": name, "parent": int(model.body_parentid[bid]), "geoms": geoms})

    cameras = []
    for cid in range(model.ncam):
        cameras.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid),
            "id": cid,
            "pos": [round(float(x), 6) for x in model.cam_pos[cid]],
            "quat": _quat(model.cam_quat[cid]),
            "fovy": round(float(model.cam_fovy[cid]), 4),
        })
    meshes = []
    for mid in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        meshes.append({"name": name, "file": name + ".stl" if name else None})
    return {
        "bodies": bodies,
        "cameras": cameras,
        "meshes": meshes,
        "mesh_root": "assets/SO101/Simulation/SO101/assets",
        "units": "metres",
        "pose_quantization": {"position_m": 1e-4, "quaternion": 1 / 32767},
    }


class Recorder:
    """Capture public observations plus the live MuJoCo pose."""

    def __init__(self, env: StationEnv, out: Path, width: int = 240, height: int = 320):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.frames: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._seen_events: set[tuple[Any, ...]] = set()
        self._last_tick = -1
        self._render_count = 0
        self._renderers = {
            "top_view": mujoco.Renderer(self.model, width, height),
            "inspection_view": mujoco.Renderer(self.model, width, height),
        }
        self._writers = {
            "top_view": imageio.get_writer(out / "top-view.mp4", fps=10, codec="libx264", quality=6, macro_block_size=1),
            "inspection_view": imageio.get_writer(out / "inspection-camera.mp4", fps=10, codec="libx264", quality=6, macro_block_size=1),
        }

    def sample(self, obs: dict[str, Any], *, force: bool = False) -> None:
        tick = int(round(float(self.data.time) * 30.0))
        if not force and tick == self._last_tick:
            return
        self._last_tick = tick
        poses = []
        for bid in range(self.model.nbody):
            poses.append([int(round(float(v) * 10000)) for v in self.data.xpos[bid]] + _quat(self.data.xquat[bid]))
        joints = np.asarray(obs.get("joints", []), dtype=float).reshape(-1).tolist()
        joints = (joints + [0.0] * 6)[:6]
        self.frames.append({
            "t": round(float(self.data.time), 4),
            "poses": poses,
            "qpos": [round(float(v), 5) for v in joints[:5]],
            "gripper": round(float(joints[5]), 5),
        })
        self._render_count += 1
        if force or self._render_count % 3 == 0:
            for camera, renderer in self._renderers.items():
                renderer.update_scene(self.data, camera)
                self._writers[camera].append_data(renderer.render())

    def events_from(self, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
        added = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # The public rollout recorder uses ``event`` for physics rows;
            # mapped demo/evaluation writers use ``type`` for cycle markers.
            # Keep both forms in the replay so no real event disappears at
            # the export boundary.
            event_name = row.get("event") or row.get("type")
            if not event_name:
                continue
            key = tuple(sorted((str(k), repr(v)) for k, v in row.items()))
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            event = {
                "t": round(float(row.get("t", self.data.time)), 4),
                "name": str(event_name),
                "frame": max(0, len(self.frames) - 1),
            }
            event.update({k: v for k, v in row.items() if k not in {"event", "type", "t"}})
            self.events.append(event)
            added.append(event)
        return added

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        for renderer in self._renderers.values():
            renderer.close()


def export(
    seed: int,
    out: Path,
    checkpoint: str = "expert",
    tier: str = "loose",
    max_steps: int = 3600,
    inspector: Callable[[np.ndarray], Any] | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    """Run and export one real rollout through ``StationEnv.step``."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    out.mkdir(parents=True, exist_ok=True)
    env = StationEnv(int(seed), tier=tier, count=count, render_size=64)
    recorder = Recorder(env, out)
    commit = _commit()
    inspector_status = "not configured"
    try:
        obs = env.reset()
        recorder.sample(obs, force=True)
        done = False
        steps = 0
        while not done and steps < int(max_steps):
            action = np.asarray(env.expert_action(), dtype=np.float32)
            obs, _, done, info = env.step(action)
            recorder.sample(obs)
            new_events = recorder.events_from(info.get("events", []))
            if inspector is not None:
                for event in new_events:
                    if event["name"] != "present":
                        continue
                    decision = inspector(obs["inspection"])
                    if isinstance(decision, dict):
                        verdict = decision.get("verdict")
                        event_data = {k: decision[k] for k in ("score", "threshold") if k in decision}
                    else:
                        verdict, event_data = decision, {}
                    recorder.events.append({
                        "t": round(float(env.data.time), 4),
                        "name": "verdict",
                        "frame": max(0, len(recorder.frames) - 1),
                        "brick": event.get("brick"),
                        "verdict": str(verdict),
                        **event_data,
                    })
                    inspector_status = "written"
            steps += 1

        recorder.sample(obs, force=True)
        result = env.result()
        bricks = len(env.brick_ids)
        successes = len(result.get("good_bowl", []))
        failures = [e for e in recorder.events if e.get("name") in {"unstable", "miss", "slip", "ik_fail"}]
        outcome = {
            "status": "success" if bricks and successes == bricks else "failed",
            "bricks": bricks,
            "correct": successes,
            "successes": successes,
            "wrong_bowl": len(result.get("wrong_bowl", [])),
            "drops": len(result.get("dropped", [])),
            "score": successes / bricks if bricks else 0.0,
            "failure_reason": failures[0].get("name") if failures else None,
            "cycle_time_s": round(float(env.data.time) / max(1, bricks), 3),
            "seed": int(seed),
            "tier": tier,
            "steps": steps,
            "commit": commit,
        }
        payload = {
            "format": "dryrun-replay-v2",
            "seed": int(seed),
            "checkpoint": checkpoint,
            "tier": tier,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "commit": commit,
            "model": "SO-101 official MJCF",
            "scene": _scene(env.model),
            "frames": recorder.frames,
            "events": recorder.events,
            "outcome": outcome,
            "videos": {"top": "top-view.mp4", "inspection": "inspection-camera.mp4", "fps": 10, "source_pose_hz": 30, "frame_stride": 3},
        }
        replay = out / "replay.json.gz"
        with gzip.open(replay, "wt", encoding="utf-8", compresslevel=9) as fh:
            json.dump(payload, fh, separators=(",", ":"))
        if getattr(env, "xml_path", None) and Path(env.xml_path).exists():
            (out / f"station-{int(seed)}.xml").write_text(Path(env.xml_path).read_text())

        telemetry_status = "not configured"
        if os.environ.get("DRYRUN_DB_URL"):
            try:
                from .telemetry import record_rollout, refresh
                base = datetime.now(timezone.utc).timestamp()
                samples = [{"ts": base + float(frame["t"]), "qpos": frame["qpos"] + [frame["gripper"]], "gripper": frame["gripper"]} for frame in recorder.frames]
                record_rollout(samples, recorder.events, outcome, rollout_id=f"replay-seed-{seed}-{commit[:12]}", seed=int(seed), checkpoint=checkpoint)
                refresh()
                telemetry_status = "written"
            except Exception as exc:
                telemetry_status = f"error: {exc}"
        manifest = {
            "replay": str(replay),
            "replay_bytes": replay.stat().st_size,
            "frames": len(recorder.frames),
            "events": len(recorder.events),
            "videos": payload["videos"],
            "outcome": outcome,
            "telemetry": telemetry_status,
            "inspector": inspector_status,
            "commit": commit,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
    finally:
        recorder.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tier", choices=sorted(TIERS), default="loose")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default="expert")
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    print(json.dumps(export(args.seed, args.out, args.checkpoint, args.tier, args.max_steps, count=args.count), indent=2))


if __name__ == "__main__":
    main()
