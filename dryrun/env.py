"""Public SO-101 MuJoCo environment for expert and learned rollouts."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import mujoco
from .grasp_test import ARM_NAMES, ids, qadr
from .closed_form_ik import IKError, ik_pad_pose
from .expert import MeasuredExpert
from .mjcf_station import ROOT, layout_bricks, station_xml

TIERS = {
    "loose": "loose flat bricks, bowls on the sides, inspection camera in front",
    "messy": "messy touching bricks, bowls on the sides, inspection camera in front",
    "pile": "dropped pile of bricks, bowls on the sides, inspection camera in front",
}

class StationEnv:
    """Vision/state observations at a deterministic public 30 Hz boundary."""
    def __init__(self, seed: int, tier: str | None = None, render_size: int = 64,
                 *, sentence: str | None = None, station_path: Path | None = None,
                 count: int | None = None):
        self.seed = int(seed); self.tier = tier or "loose"
        if self.tier not in TIERS: raise ValueError(f"tier must be one of {sorted(TIERS)}")
        self.sentence = sentence or TIERS[self.tier]
        xml = (Path(station_path).read_text() if station_path is not None else
               station_xml(self.seed, self.sentence, count=count,
                           robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve())))
        self.xml_path = Path("/tmp") / f"dryrun_env_{self.seed}_{self.tier}.xml"; self.xml_path.write_text(xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path)); self.data = mujoco.MjData(self.model)
        self.specs = layout_bricks(self.seed, self.sentence, count=count)
        self.arm = ids(self.model); self._qa = qadr(self.model, self.arm)
        self._grip = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")])
        self.renderer = mujoco.Renderer(self.model, render_size, render_size); self._inspection_renderer = mujoco.Renderer(self.model, render_size, render_size)
        self._planner = MeasuredExpert(self)
        self._ticks = 0; self._done = False; self._events: list[dict[str, Any]] = []; self._cycle_times: dict[str, float] = {}; self._substeps = 0.0; self._last_action = np.zeros(6, np.float32); self._active = None
        self.reset()

    @property
    def brick_ids(self):
        return [i for i in range(self.model.nbody) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("brick_")]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data); mujoco.mj_forward(self.model, self.data)
        self._ticks = 0; self._done = False; self._events.clear(); self._cycle_times.clear(); self._substeps = 0.0; self._active = None
        self._last_action = np.asarray([self.data.qpos[a] for a in self._qa] + [self.data.qpos[self._grip]], np.float32)
        self._planner = MeasuredExpert(self)
        return self.observe()

    def observe(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, "top_view"); self._inspection_renderer.update_scene(self.data, "inspection_view")
        return {"top": self.renderer.render().copy().astype(np.uint8), "inspection": self._inspection_renderer.render().copy().astype(np.uint8), "joints": np.asarray([self.data.qpos[a] for a in self._qa] + [self.data.qpos[self._grip]], np.float32)}

    def _event(self, event: str, brick: int | None = None, **extra):
        row = {"event": event, "t": float(self.data.time)}
        if brick is not None: row["brick"] = int(brick)
        row.update(extra); self._events.append(row)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (6,) or not np.all(np.isfinite(action)): raise ValueError("action must be six finite joint targets")
        self.data.ctrl[:6] = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]); self._last_action = action.astype(np.float32)
        self._substeps += (1.0 / 30.0) / float(self.model.opt.timestep); n = max(1, int(round(self._substeps))); self._substeps -= n
        for _ in range(n): mujoco.mj_step(self.model, self.data)
        self._ticks += 1
        if self._active is not None:
            pads = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fixed_grip_pad", "moving_grip_pad")}
            touched: set[int] = set()
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.geom1 in pads or c.geom2 in pads:
                    touched.add(c.geom1 if c.geom1 in pads else c.geom2)
            if len(touched) == 2 and not any(e.get("event") == "grasp" and e.get("brick") == self._active for e in self._events):
                self._event("grasp", self._active)
            p = self.data.xpos[self._active]
            if not np.all(np.isfinite(p)) or float(p[2]) < -0.01 or float(np.linalg.norm(p[:2])) > 1.0:
                self._event("unstable", self._active); self._done = True
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)) or float(np.max(np.abs(self.data.qvel))) > 50:
            self._event("unstable", self._active); self._done = True
        if self._ticks >= 2150: self._done = True
        return self.observe(), 0.0, self._done, {"events": list(self._events[-8:]), "time": float(self.data.time)}

    def expert_action(self):
        action = self._planner.action()
        for row in self._planner.events():
            extra = {k: v for k, v in row.items() if k not in {"event", "brick", "t"}}
            self._event(str(row.get("event", "event")), row.get("brick"), **extra)
        self._active = self._planner.current.body if self._planner.current is not None else None
        return np.asarray(action, dtype=np.float32)

    def result(self):
        good_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "good_bowl"); bad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "defective_bowl")
        good_xy, bad_xy = self.data.xpos[good_id, :2], self.data.xpos[bad_id, :2]; good, wrong, dropped = [], [], []
        for b in self.brick_ids:
            p = self.data.xpos[b]
            if not np.all(np.isfinite(p)) or float(p[2]) < 0: dropped.append(int(b)); continue
            dg, db = np.linalg.norm(p[:2] - good_xy), np.linalg.norm(p[:2] - bad_xy)
            if min(dg, db) < .11 and p[2] < .10: (good if dg <= db else wrong).append(int(b))
        return {"good_bowl": good, "wrong_bowl": wrong, "dropped": dropped, "cycle_times": dict(self._cycle_times), "events": list(self._events)}

    def close(self): self.renderer.close(); self._inspection_renderer.close()

__all__ = ["StationEnv", "TIERS"]
