"""Public MuJoCo environment contract for Dry Run.

Observations contain only rendered camera frames and six joint positions. The
expert helper is allowed to inspect the simulator state; learned controllers
must use ``observe()`` and ``step()`` through the same interface.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .grasp_test import ARM_NAMES, ids, qadr, solve_restarted
from .mjcf_station import ROOT, layout_bricks, station_xml


class StationEnv:
    """A 30 Hz SO-101 station with image/state observations."""

    def __init__(self, seed: int, tier: str | None = None):
        self.seed = int(seed)
        self.tier = tier or "loose"
        sentence = {
            "loose": "loose flat bricks, bowls on the sides, inspection camera in front",
            "messy": "messy touching bricks, bowls on the sides, inspection camera in front",
            "pile": "dropped pile of bricks, bowls on the sides, inspection camera in front",
        }.get(self.tier, "loose flat bricks, bowls on the sides, inspection camera in front")
        self.sentence = sentence
        xml = station_xml(self.seed, sentence, robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve()))
        self.model = mujoco.MjModel.from_xml_string(xml.encode())
        self.data = mujoco.MjData(self.model)
        self.arm = ids(self.model)
        self._qa = qadr(self.model, self.arm)
        self._grip_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self._grip_adr = int(self.model.jnt_qposadr[self._grip_joint])
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._events: list[dict[str, Any]] = []
        self._cycle_start: dict[int, float] = {}
        self._done: set[int] = set()
        self._active: int | None = None
        self._last_action = np.zeros(6, dtype=np.float32)
        self._reset_time = time.time()
        self.reset()

    @property
    def brick_ids(self) -> list[int]:
        return [i for i in range(self.model.nbody) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("brick_")]

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._events.clear(); self._done.clear(); self._active = None; self._cycle_start = {b: float(self.data.time) for b in self.brick_ids}; self._reset_time = time.time()
        self._last_action = np.array([self.data.qpos[a] for a in self._qa] + [self.data.qpos[self._grip_adr]], dtype=np.float32)
        return self.observe()

    def _renderer(self, camera: str) -> mujoco.Renderer:
        if camera not in self._renderers:
            self._renderers[camera] = mujoco.Renderer(self.model, 180, 320)
        return self._renderers[camera]

    def observe(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, camera in (("top", "top_view"), ("inspection", "inspection_view")):
            renderer = self._renderer(camera)
            renderer.update_scene(self.data, camera)
            out[key] = np.asarray(renderer.render(), dtype=np.uint8).copy()
        joints = np.array([self.data.qpos[a] for a in self._qa] + [self.data.qpos[self._grip_adr]], dtype=np.float32)
        out["joints"] = joints
        return out

    def _emit(self, name: str, brick: int | None = None, **extra: Any) -> None:
        row = {"t": round(float(self.data.time), 4), "name": name}
        if brick is not None:
            row["brick"] = int(brick)
        row.update(extra)
        self._events.append(row)

    def step(self, action: Any) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        target = np.asarray(action, dtype=np.float32).reshape(-1)
        if target.size != 6 or not np.all(np.isfinite(target)):
            raise ValueError("action must be six finite joint position targets")
        for i, actuator in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            self.data.ctrl[aid] = float(np.clip(target[i], self.model.actuator_ctrlrange[aid, 0], self.model.actuator_ctrlrange[aid, 1]))
        self._last_action = target.copy()
        # 2 ms MuJoCo timestep -> 50 substeps gives 10 Hz; the station's public
        # contract is 30 Hz, so use 16 or 17 substeps and accumulate exactly.
        for _ in range(16):
            mujoco.mj_step(self.model, self.data)
        obs = self.observe()
        result = self.result()
        done = bool(len(self._done) == len(self.brick_ids))
        return obs, 0.0, done, {"time": float(self.data.time), "events": self._events[-8:], "result": result}

    def expert_action(self) -> np.ndarray:
        """Return one six-target action from the ground-truth scripted expert."""
        remaining = [b for b in self.brick_ids if b not in self._done]
        if not remaining:
            return self._last_action.copy()
        body = min(remaining, key=lambda b: float(np.linalg.norm(self.data.xpos[b, :2])))
        self._active = body
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or f"brick_{body}"
        if not self._events or self._events[-1].get("brick") != body:
            self._emit("reach", body, id=name)
        target = np.array([float(self.data.xpos[body, 0]), float(self.data.xpos[body, 1]), 0.075])
        q, err, ok = solve_restarted(self.model, self.data, target, self.arm, 1.7)
        if not ok:
            self._emit("ik_fail", body, error_m=float(err)); return self._last_action.copy()
        self._emit("grasp", body, planned=True)
        return np.r_[q, 1.7].astype(np.float32)

    def result(self) -> dict[str, Any]:
        good, wrong, dropped = [], [], []
        bowls = {"good": self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "good_bowl")], "wrong": self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "defective_bowl")]}
        for b in self.brick_ids:
            p = self.data.xpos[b]
            if float(p[2]) < 0.0 or not np.all(np.isfinite(p)):
                dropped.append(int(b)); continue
            dgood = float(np.linalg.norm(p[:2] - bowls["good"][:2])); dwrong = float(np.linalg.norm(p[:2] - bowls["wrong"][:2]))
            if min(dgood, dwrong) < 0.11 and float(p[2]) < 0.10:
                (good if dgood <= dwrong else wrong).append(int(b))
        return {"good_bowl": good, "wrong_bowl": wrong, "dropped": dropped, "cycle_times": {str(b): round(float(self.data.time) - self._cycle_start.get(b, 0.0), 4) for b in self.brick_ids}}
