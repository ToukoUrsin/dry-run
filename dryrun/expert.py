"""Measured loose-tier expert from prompts/019-grasp-recipe-measured.md.

The expert is intentionally a small state machine.  It owns only target
generation and timing; MuJoCo remains the source of contact, lift, slip and
drop events through :class:`StationEnv`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_POS = np.asarray((0.0225, 0.0, -0.086), dtype=float)
PAN_AXIS = np.asarray((0.0388, 0.0), dtype=float)
OPEN_Q, CLOSED_Q = 1.0, 0.5
TICK_HZ = 30.0


@dataclass
class PlanItem:
    body: int
    index: int
    size: str
    jaw_angle: float
    target_xy: np.ndarray
    defect: str


def _unit_xy(angle: float) -> np.ndarray:
    return np.asarray((math.cos(angle), math.sin(angle), 0.0), dtype=float)


class MeasuredExpert:
    """Prompt-019 cycle planner and DLS TCP solver."""

    def __init__(self, env: Any):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.arm_joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES]
        self.qa = [int(self.model.jnt_qposadr[j]) for j in self.arm_joints]
        self.grip_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self.grip_adr = int(self.model.jnt_qposadr[self.grip_joint])
        self.tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "dryrun_tcp")
        if self.tcp < 0:
            # The vendor SO-101 include exposes the calibrated TCP as
            # ``gripperframe``; keep ``dryrun_tcp`` as an optional alias for
            # stations that add the explicit recipe site.
            self.tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.gripper_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.pad_ids = tuple(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fixed_grip_pad", "moving_grip_pad"))
        self.tick = 0
        self.phase = "neutral"
        self.phase_tick = 0
        self.item_no = 0
        self.plan: list[PlanItem] = []
        self.current: PlanItem | None = None
        self.current_q: np.ndarray | None = None
        self._events: list[dict[str, Any]] = []
        self._planned = False
        self._present_b = False
        self._last_result = None
        self._pose_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def _body_specs(self):
        return self.env.specs

    def _body_ids(self):
        return {int(b): int(i) for i, b in enumerate(self.env.brick_ids)}

    def _clear(self, item: PlanItem, axis: np.ndarray) -> bool:
        """Prompt-019 footprint test against other brick rectangles/bowls."""
        p = item.target_xy
        for other in self._body_specs():
            if other.index == item.index:
                continue
            minimum = .078 if item.size == other.size == "2x2" else .095
            if math.hypot(float(p[0] - other.x), float(p[1] - other.y)) < minimum:
                return False
        for bowl in ((-.0093, -.2657), (-.0093, .2657)):
            if math.hypot(float(p[0] - bowl[0]), float(p[1] - bowl[1])) < .095 + .015:
                return False
        return True

    def _solve(self, target: np.ndarray, pitch: float, jaw_angle: float, *, initial: np.ndarray | None = None) -> np.ndarray | None:
        """Measured damped least-squares TCP pose solve from prompt 019."""
        target = np.asarray(target, dtype=float)
        radial = target[:2] - PAN_AXIS
        norm = float(np.linalg.norm(radial))
        if norm < 1e-9:
            return None
        rhat = radial / norm
        f_des = math.sin(pitch) * np.asarray((rhat[0], rhat[1], 0.0)) - math.cos(pitch) * np.asarray((0.0, 0.0, 1.0))
        jaw_des = _unit_xy(jaw_angle)
        seeds = []
        if initial is not None:
            seeds.append(np.asarray(initial, dtype=float).copy())
        seeds.extend([
            np.asarray((-math.atan2(radial[1], radial[0]), -.3, .6, 1.2, 0.0)),
            np.asarray((-math.atan2(radial[1], radial[0]), .2, .2, 1.2, 0.0)),
            np.asarray((-math.atan2(radial[1], radial[0]), -.8, 1.2, 1.1, 0.0)),
        ])
        for roll in (0.0, 1.4, -1.4, 2.5, -2.5):
            for base in seeds[:3]:
                q = base.copy(); q[4] = roll
                for _ in range(120):
                    for k, adr in enumerate(self.qa):
                        self.data.qpos[adr] = q[k]
                    mujoco.mj_forward(self.model, self.data)
                    p = self.data.site_xpos[self.tcp].copy()
                    R = self.data.xmat[self.gripper_body].reshape(3, 3)
                    f = -R[:, 2]; jaw = R[:, 0]
                    e = np.r_[target - p, .05 * (np.cross(f, f_des) + np.cross(jaw, jaw_des))]
                    if np.linalg.norm(target - p) < .001 and np.linalg.norm(np.cross(f, f_des)) < math.sin(math.radians(2)) and np.linalg.norm(np.cross(jaw, jaw_des)) < math.sin(math.radians(3)):
                        return q.copy()
                    Jp = np.zeros((3, self.model.nv)); Jr = np.zeros((3, self.model.nv))
                    mujoco.mj_jacSite(self.model, self.data, Jp, Jr, self.tcp)
                    cols = [self.model.jnt_dofadr[j] for j in self.arm_joints]
                    J = np.vstack((Jp[:, cols], .05 * Jr[:, cols]))
                    dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
                    dq = np.clip(dq, -.3, .3)
                    q = np.clip(q + dq, self.model.jnt_range[self.arm_joints, 0] + .02, self.model.jnt_range[self.arm_joints, 1] - .02)
        return None

    def _make_plan(self) -> None:
        if self._planned:
            return
        body_by_index = {int(i): int(b) for b, i in self._body_ids().items()}
        candidates: list[PlanItem] = []
        for spec in self._body_specs():
            b = body_by_index.get(spec.index)
            if b is None:
                continue
            short = float(spec.yaw + (math.pi / 2.0 if spec.size == "2x4" else 0.0))
            options = (short, short + math.pi) if spec.size == "2x4" else (short, short + math.pi / 2, short + math.pi, short + 3 * math.pi / 2)
            for angle in options:
                axis = _unit_xy(angle)
                item = PlanItem(b, spec.index, spec.size, angle, np.asarray((spec.x, spec.y), dtype=float), spec.defect)
                if self._clear(item, axis):
                    # The measured work-area gate is checked at the three
                    # recipe heights before an item enters the queue.
                    ok = all(self._solve(np.asarray((spec.x - .0045 * axis[0], spec.y - .0045 * axis[1], z)), 0.0, angle) is not None for z in (.013, .055, .07))
                    if ok:
                        candidates.append(item)
                        break
        # Nearest-first is the measured recipe order and remains deterministic.
        self.plan = sorted(candidates, key=lambda x: float(np.linalg.norm(x.target_xy - PAN_AXIS)))
        self._planned = True

    def _set_phase(self, phase: str) -> None:
        self.phase = phase; self.phase_tick = 0
        if self.current is not None:
            # Keep the public stream at the physical task vocabulary.  The
            # detailed phase remains internal to the planner state machine.
            public = {"close": "grasp", "lift04": "lift", "present_a": "present",
                      "present_b": "present", "drop": "drop"}.get(phase)
            if public is not None:
                self._events.append({"event": public, "brick": self.current.index, "t": float(self.data.time)})

    def _pose(self, z: float, pitch: float, jaw_angle: float, *, bowl: np.ndarray | None = None,
              target_xy: np.ndarray | None = None) -> np.ndarray:
        if bowl is not None:
            target = np.asarray((bowl[0], bowl[1], z), dtype=float)
        elif target_xy is not None:
            target = np.asarray((target_xy[0], target_xy[1], z), dtype=float)
        else:
            assert self.current is not None
            axis = _unit_xy(self.current.jaw_angle)
            target = np.asarray((self.current.target_xy[0] - .0045 * axis[0], self.current.target_xy[1] - .0045 * axis[1], z), dtype=float)
        key = (self.phase, tuple(np.round(target, 6)), round(float(pitch), 6), round(float(jaw_angle), 6))
        solved = self._pose_cache.get(key)
        if solved is None:
            solved = self._solve(target, pitch, jaw_angle, initial=self.current_q)
            if solved is not None:
                self._pose_cache[key] = solved.copy()
        if solved is not None:
            self.current_q = np.asarray(solved, dtype=float).copy()
        if self.current_q is None:
            self._events.append({"event": "ik_fail", "brick": self.current.index if self.current else None, "t": float(self.data.time)})
            self.current_q = np.asarray([self.data.qpos[a] for a in self.qa], dtype=float)
        return np.r_[self.current_q, OPEN_Q if self.phase in {"neutral", "hover", "down"} else CLOSED_Q].astype(np.float32)

    def action(self) -> np.ndarray:
        self._make_plan()
        if self.current is None and self.item_no < len(self.plan):
            self.current = self.plan[self.item_no]; self.current_q = None; self._set_phase("hover")
        if self.current is None:
            return np.r_[np.asarray([self.data.qpos[a] for a in self.qa]), OPEN_Q].astype(np.float32)
        # Fixed recipe durations at 30 Hz.  Pose interpolation is represented
        # by the servo target changing each tick; MuJoCo supplies the 17 steps.
        self.phase_tick += 1; self.tick += 1
        if self.phase == "hover" and self.phase_tick > 35: self._set_phase("down035")
        elif self.phase == "down035" and self.phase_tick > 11: self._set_phase("down013")
        elif self.phase == "down013" and self.phase_tick > 17: self._set_phase("close")
        elif self.phase == "close" and self.phase_tick > 27: self._set_phase("lift04")
        elif self.phase == "lift04" and self.phase_tick > 12: self._set_phase("lift07")
        elif self.phase == "lift07" and self.phase_tick > 12: self._set_phase("present_a")
        elif self.phase == "present_a" and self.phase_tick > 48: self._set_phase("present_b")
        elif self.phase == "present_b" and self.phase_tick > 42: self._set_phase("drop")
        elif self.phase == "drop" and self.phase_tick > 42: self._set_phase("release")
        elif self.phase == "release" and self.phase_tick > 27: self._set_phase("neutral")
        elif self.phase == "neutral" and self.phase_tick > 30:
            self.item_no += 1; self.current = None; self.current_q = None
        if self.phase == "hover": return self._pose(.055, .0, self.current.jaw_angle)
        if self.phase == "down035": return self._pose(.035, .0, self.current.jaw_angle)
        if self.phase in {"down013", "close"}: return self._pose(.013, .0, self.current.jaw_angle)
        if self.phase == "lift04": return self._pose(.04, .6, self.current.jaw_angle)
        if self.phase == "lift07": return self._pose(.07, .6, self.current.jaw_angle)
        if self.phase == "present_a": return self._pose(.16, .9, -2.3, target_xy=np.asarray((.20, -.10)))
        if self.phase == "present_b": return self._pose(.16, .9, .84, target_xy=np.asarray((.20, -.10)))
        if self.phase in {"drop", "release"}:
            bad = self.current.defect != "none"
            bowl = np.asarray((-.0093, .2657 if bad else -.2657))
            return self._pose(.115, .6, self.current.jaw_angle, bowl=bowl)
        return self._pose(.14, .6, self.current.jaw_angle)

    def events(self) -> list[dict[str, Any]]:
        rows, self._events = self._events, []
        return rows


__all__ = ["MeasuredExpert", "PlanItem"]
