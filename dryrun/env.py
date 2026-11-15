"""Public SO-101 MuJoCo environment for expert and learned rollouts."""
from __future__ import annotations
from pathlib import Path
import os
from typing import Any
import numpy as np
import mujoco
from .grasp_test import ARM_NAMES, ids, qadr
from .closed_form_ik import IKError, ik_pad_pose
from .defects import INSPECTION_PRESENT_POSE, render_inspection_frame
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
                 count: int | None = None, look: str = "lab"):
        self.seed = int(seed); self.tier = tier or "loose"; self.look = str(look)
        if self.tier not in TIERS: raise ValueError(f"tier must be one of {sorted(TIERS)}")
        self.sentence = sentence or TIERS[self.tier]
        xml = (Path(station_path).read_text() if station_path is not None else
               station_xml(self.seed, self.sentence, count=count,
                           robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve()),
                           look=look))
        self.xml_path = Path("/tmp") / f"dryrun_env_{self.seed}_{self.tier}.xml"; self.xml_path.write_text(xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path)); self.data = mujoco.MjData(self.model)
        self.specs = layout_bricks(self.seed, self.sentence, count=count)
        self.arm = ids(self.model); self._qa = qadr(self.model, self.arm)
        self._grip = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")])
        self._render_size = int(render_size)
        self.renderer = mujoco.Renderer(self.model, render_size, render_size); self._inspection_renderer = mujoco.Renderer(self.model, render_size, render_size)
        self._planner = MeasuredExpert(self)
        self._last_obs: dict[str, Any] | None = None
        self._ticks = 0; self._done = False; self._events: list[dict[str, Any]] = []; self._cycle_times: dict[str, float] = {}; self._substeps = 0.0; self._last_action = np.zeros(6, np.float32); self._active = None
        self._physical_events: set[tuple[str, int, str]] = set(); self._active_z_prev: float | None = None
        self._brick_z_prev: dict[int, float] = {}; self._inspection_focus: int | None = None
        self._force_inspection = False
        self._max_ticks = (120 + 330 * len(self.brick_ids)) * (1 if self.tier == "loose" else 2)
        self.reset()

    @property
    def brick_ids(self):
        return [i for i in range(self.model.nbody) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("brick_")]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data); mujoco.mj_forward(self.model, self.data)
        self._ticks = 0; self._done = False; self._events.clear(); self._cycle_times.clear(); self._substeps = 0.0; self._active = None
        self._physical_events.clear(); self._active_z_prev = None
        self._brick_z_prev = {int(body): float(self.data.xpos[body, 2]) for body in self.brick_ids}
        self._inspection_focus = None
        self._last_action = np.asarray([self.data.qpos[a] for a in self._qa] + [self.data.qpos[self._grip]], np.float32)
        self._planner = MeasuredExpert(self)
        self._last_obs = None
        return self.observe()

    def observe(self):
        mujoco.mj_forward(self.model, self.data)
        joints = np.asarray([self.data.qpos[a] for a in self._qa] + [self.data.qpos[self._grip]], np.float32)
        # Rendering two EGL views dominates CPU rollout time.  Keep the
        # public two-camera observation contract, but refresh at 1 Hz; joints
        # remain current at the 30 Hz physics boundary.  A fresh frame is
        # available on the presentation holds (their phase boundaries are
        # aligned to the 30-tick cadence).
        if os.environ.get("DRYRUN_TELEMETRY_ONLY") == "1":
            # Tiger backfills need the real physics/joint tick stream, not
            # camera pixels.  Avoiding EGL renders makes the six bounded
            # COPY batches finish without changing any simulation state.
            if self._last_obs is None:
                blank = np.zeros((self._render_size, self._render_size, 3), dtype=np.uint8)
                self._last_obs = {"top": blank, "inspection": blank.copy()}
            self._force_inspection = False
        elif self._last_obs is None or self._ticks % 30 == 0 or self._force_inspection:
            self.renderer.update_scene(self.data, "top_view")
            top = self.renderer.render().copy().astype(np.uint8)
            focus = self._active if self._active is not None else self._inspection_focus
            inspection = render_inspection_frame(self.model, self.data, self._inspection_renderer, focus_body=focus)
            self._last_obs = {"top": top, "inspection": inspection.astype(np.uint8, copy=False)}
            self._force_inspection = False
        return {"top": self._last_obs["top"], "inspection": self._last_obs["inspection"], "joints": joints}

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
        pads = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fixed_grip_pad", "moving_grip_pad")}
        pad_forces: dict[int, dict[int, float]] = {}
        brick_set = {int(body) for body in self.brick_ids}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pad = c.geom1 if c.geom1 in pads else (c.geom2 if c.geom2 in pads else -1)
            if pad < 0:
                continue
            other = c.geom2 if pad == c.geom1 else c.geom1
            body = int(self.model.geom_bodyid[other])
            if body not in brick_set:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            pad_forces.setdefault(body, {}).setdefault(int(pad), 0.0)
            pad_forces[body][int(pad)] += abs(float(force[0]))

        # These events are derived from MuJoCo contacts and poses, so learned
        # controllers get the same physical stage accounting as the expert.
        # The controller may still attach an opaque token to its own event;
        # this stream uses the simulator body id only for internal scoring.
        tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        tcp = self.data.site_xpos[tcp_site] if tcp_site >= 0 else None
        if tcp is not None:
            for body in brick_set:
                if float(np.linalg.norm(self.data.xpos[body] - tcp)) <= .06:
                    reach_key = ("reach", body, "")
                    if reach_key not in self._physical_events:
                        self._physical_events.add(reach_key); self._event("reach", body)

        for body, forces in pad_forces.items():
            grasp_key = ("grasp", body, "")
            if len(pads) == 2 and all(forces.get(pad, 0.0) >= 0.2 for pad in pads):
                grasped.add(body)
                if grasp_key not in self._physical_events:
                    self._physical_events.add(grasp_key); self._event("grasp", body)
                self._inspection_focus = body

        good_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "good_bowl")
        bad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "defective_bowl")
        bowl_xy = [self.data.xpos[idx, :2] for idx in (good_id, bad_id) if idx >= 0]
        for body in brick_set:
            p = self.data.xpos[body]
            z = float(p[2])
            prev_z = self._brick_z_prev.get(body, z)
            if ("grasp", body, "") in self._physical_events and prev_z <= .04 < z:
                lift_key = ("lift", body, "")
                if lift_key not in self._physical_events:
                    self._physical_events.add(lift_key); self._event("lift", body)
            self._brick_z_prev[body] = z
            grasp_key = ("grasp", body, "")
            if grasp_key in self._physical_events:
                present_key = ("present", body, "")
                if float(np.linalg.norm(p - INSPECTION_PRESENT_POSE)) <= .05:
                    self._inspection_focus = body
                    if present_key not in self._physical_events and not any(
                        row.get("event") == "present" and int(row.get("brick", -1)) == body
                        for row in self._events
                    ):
                        self._physical_events.add(present_key); self._event("present", body); self._force_inspection = True
                if not pad_forces.get(body) and any(float(np.linalg.norm(p[:2] - xy)) < .11 and z < .10 for xy in bowl_xy):
                    drop_key = ("drop", body, "")
                    if drop_key not in self._physical_events and not any(
                        row.get("event") == "drop" and int(row.get("brick", -1)) == body
                        for row in self._events
                    ):
                        self._physical_events.add(drop_key); self._event("drop", body)
            if not np.all(np.isfinite(p)) or z < -0.01 or float(np.linalg.norm(p[:2])) > 1.0:
                self._event("unstable", body); self._done = True
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)) or float(np.max(np.abs(self.data.qvel))) > 50:
            self._event("unstable", self._active); self._done = True
        if self._ticks >= self._max_ticks: self._done = True
        return self.observe(), 0.0, self._done, {"events": list(self._events[-8:]), "time": float(self.data.time)}

    def expert_action(self):
        action = self._planner.action()
        for row in self._planner.events():
            extra = {k: v for k, v in row.items() if k not in {"event", "brick", "t"}}
            self._event(str(row.get("event", "event")), row.get("brick"), **extra)
            if row.get("event") == "present":
                self._force_inspection = True
        next_active = self._planner.current.body if self._planner.current is not None else None
        if next_active != self._active:
            self._active_z_prev = None
        self._active = next_active
        return np.asarray(action, dtype=np.float32)

    def result(self):
        good_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "good_bowl"); bad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "defective_bowl")
        good_xy, bad_xy = self.data.xpos[good_id, :2], self.data.xpos[bad_id, :2]
        good, wrong, dropped, correct, false_accepts, false_rejects = [], [], [], [], [], []
        by_body = {int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"brick_{s.index}")): str(s.defect) for s in self.specs}
        for b in self.brick_ids:
            p = self.data.xpos[b]
            if not np.all(np.isfinite(p)) or float(p[2]) < 0: dropped.append(int(b)); continue
            dg, db = np.linalg.norm(p[:2] - good_xy), np.linalg.norm(p[:2] - bad_xy)
            if min(dg, db) < .11 and p[2] < .10:
                if dg <= db:
                    good.append(int(b))
                    (false_accepts if by_body.get(int(b), "none") != "none" else correct).append(int(b))
                else:
                    wrong.append(int(b))
                    (correct if by_body.get(int(b), "none") != "none" else false_rejects).append(int(b))
        return {"good_bowl": good, "wrong_bowl": wrong, "correct": correct,
                "false_accepts": false_accepts, "false_rejects": false_rejects,
                "dropped": dropped, "cycle_times": dict(self._cycle_times), "events": list(self._events)}

    def close(self): self.renderer.close(); self._inspection_renderer.close()

__all__ = ["StationEnv", "TIERS"]
