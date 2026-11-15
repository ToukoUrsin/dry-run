"""Camera-only scripted sorter.

The controller consumes only top/inspection pixels and the six measured arm
joint readings.  It keeps an estimated brick list from perception; it never
consults MuJoCo brick poses, contacts, labels, ``env.specs`` or the active
body.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import cv2
import mujoco
import numpy as np

from .expert import MeasuredExpert
from .perception import BrickDetection, TopCameraCalibration, arm_mask_from_kinematics, bowl_mask, detect_bricks

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_LOCAL = np.array([0.0225, 0.0, -0.086], dtype=float)
PRESENT_A = np.array([0.5527, -1.2042, 0.8575, 1.0175, -2.3], dtype=float)
PRESENT_B = np.array([0.7468, -1.5394, 1.1230, 1.0625, 0.9507], dtype=float)
Q_OPEN, Q_CLOSE = 1.0, 0.2
HZ = 30
GOOD_BOWL = np.array((-.0093, -.2657), dtype=float)
BAD_BOWL = np.array((-.0093, .2657), dtype=float)
DETECTOR_FRAME_SIZE = 720
CAMERA_PARK_TCP = np.array((0.0, -0.30, 0.18), dtype=float)


class CameraExpert(MeasuredExpert):
    """A measured cycle controller driven by top-camera brick estimates."""

    def __init__(self, env, *, calibration: TopCameraCalibration | None = None,
                 detector=detect_bricks, max_look_misses: int = 2):
        # Do not call MeasuredExpert.__init__: it reads env.specs and builds a
        # truth brick map.  We retain the class boundary and initialise only
        # arm kinematics needed for camera-estimated targets.
        self.env = env
        self.m = env.model
        self.d = env.data
        nid = lambda typ, name: mujoco.mj_name2id(self.m, typ, name)
        jids = [nid(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]
        self.qa = [int(self.m.jnt_qposadr[j]) for j in jids]
        self.da = [int(self.m.jnt_dofadr[j]) for j in jids]
        self.lo = self.m.jnt_range[jids, 0] + .02
        self.hi = self.m.jnt_range[jids, 1] - .02
        self.gq = int(self.m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "gripper")])
        self.G = nid(mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.k = mujoco.MjData(self.m)
        # Only six joint positions are read from the live environment.
        self._cmd = np.asarray([self.d.qpos[a] for a in self.qa] + [self.d.qpos[self.gq]], dtype=float)
        self.calibration = calibration
        self.detector = detector
        self.max_look_misses = int(max_look_misses)
        self.phase = "look"
        self.current = None
        self._motion = None
        self._latest_obs: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._token = 0
        self._look_misses = 0
        self._detections: list[BrickDetection] = []
        # Perception failures must remain visible in rollout telemetry.  A
        # failed refresh is recoverable (the next neutral-pose look can still
        # find bricks), but it must never masquerade as an empty table.
        self._refresh_errors: list[dict[str, Any]] = []
        # A failed camera target must not be selected again on the next look;
        # the opaque rollout token changes every time a frame is refreshed.
        self._failed_tokens: dict[int, tuple[Any, ...]] = {}
        self._camera_parked = False
        self._parking = False
        self._done = False

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Scratch kinematics starts from zeros; no simulator brick qpos is read.
        self.k.qpos[:] = 0.0
        self.k.qpos[self.qa] = np.asarray(q, dtype=float)
        self.k.qpos[self.gq] = float(self._cmd[5])
        mujoco.mj_kinematics(self.m, self.k)
        mujoco.mj_comPos(self.m, self.k)
        R = self.k.xmat[self.G].reshape(3, 3)
        return self.k.xpos[self.G] + R @ TCP_LOCAL, R

    def _ik(self, pos, *, jaw=None, pitch=0.0, q0=None):
        pos = np.asarray(pos, dtype=float)
        az = math.atan2(pos[1], pos[0] - .0388)
        rhat = np.array((math.cos(az), math.sin(az), 0.0))
        f_des = math.sin(pitch) * rhat + np.array((0.0, 0.0, -math.cos(pitch)))
        jd = None
        if jaw is not None:
            jd = np.asarray(jaw, dtype=float)
            jd -= jd.dot(f_des) * f_des
            jd /= max(np.linalg.norm(jd), 1e-9)
        seeds = [] if q0 is None else [np.asarray(q0, dtype=float)]
        seeds += [np.array((-az, a, b, c, roll), dtype=float)
                  for roll in (0.0, -1.4, 1.4, -2.5, 2.5)
                  for a, b, c in ((-.3, .6, 1.2), (.2, .2, 1.2), (-.8, 1.2, 1.1))]
        best = None
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        for seed in seeds:
            q = np.clip(seed, self.lo, self.hi)
            for _ in range(90):
                p, R = self._fk(q)
                er = np.cross(-R[:, 2], f_des)
                if jd is not None:
                    er += np.cross(R[:, 0], jd)
                e = np.r_[pos - p, .05 * er]
                if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(er) < 2e-3:
                    break
                mujoco.mj_jac(self.m, self.k, jp, jr, p, self.G)
                J = np.vstack((jp[:, self.da], .05 * jr[:, self.da]))
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
                mx = float(np.max(np.abs(dq)))
                if mx > .3:
                    dq *= .3 / mx
                q = np.clip(q + dq, self.lo, self.hi)
            p, R = self._fk(q)
            pe = float(np.linalg.norm(pos - p))
            fe = math.degrees(math.acos(float(np.clip((-R[:, 2]).dot(f_des), -1, 1))))
            je = 0.0 if jd is None else math.degrees(math.acos(float(np.clip(R[:, 0].dot(jd), -1, 1))))
            score = pe * 1000 + .2 * fe + .2 * je
            if best is None or score < best[0]:
                best = (score, q.copy(), pe, fe, je)
        if best is None or best[2] >= .002 or best[3] >= 4.0 or best[4] >= 6.0:
            return None
        return best[1]

    def _move(self, q, grip, seconds, phase):
        self.phase = phase
        q = np.asarray(q, dtype=float)
        q0 = self._cmd[:5].copy(); g0 = float(self._cmd[5])
        n = max(2, round(seconds * HZ))
        for i in range(1, n + 1):
            a = i / n; a = 10 * a**3 - 15 * a**4 + 6 * a**5
            self._cmd = np.r_[q0 + a * (q - q0), g0 + a * (grip - g0)]
            yield self._cmd.copy()

    def _hold(self, seconds, phase=None):
        if phase:
            self.phase = phase
        for _ in range(max(1, round(seconds * HZ))):
            yield self._cmd.copy()

    def _emit(self, name: str, **extra) -> None:
        row = {"event": name, "t": float(getattr(self.d, "time", 0.0)), **extra}
        self._events.append(row)
        # The public rollout wrapper consumes env events. A camera token is a
        # log identifier only; env._active stays None and no body is touched.
        try:
            self.env._event(name, extra.get("brick"), **{k: v for k, v in extra.items() if k != "brick"})
        except Exception:
            pass

    def events(self):
        out, self._events = self._events, []
        return out

    def _refresh(self, obs):
        image = obs.get("top") if isinstance(obs, dict) else None
        if image is None:
            return []
        # The perception thresholds and residual checks were calibrated on the
        # 720 px static-evaluation contract.  Modal rollouts render 960 px;
        # normalize that frame before doing *any* geometry or masking so the
        # detector receives the same pixel scale while the back-projection
        # still uses the physical camera calibration.
        detector_image = np.asarray(image)
        if detector_image.shape[:2] != (DETECTOR_FRAME_SIZE, DETECTOR_FRAME_SIZE):
            detector_image = cv2.resize(
                detector_image,
                (DETECTOR_FRAME_SIZE, DETECTOR_FRAME_SIZE),
                interpolation=cv2.INTER_AREA,
            )
        if self.calibration is None:
            cal = TopCameraCalibration(
                width_px=DETECTOR_FRAME_SIZE,
                height_px=DETECTOR_FRAME_SIZE,
            )
        else:
            cal = replace(
                self.calibration,
                width_px=DETECTOR_FRAME_SIZE,
                height_px=DETECTOR_FRAME_SIZE,
            )
        try:
            bm = bowl_mask(detector_image.shape[:2], cal)
            joints = obs.get("joints")
            arm = arm_mask_from_kinematics(detector_image.shape[:2], cal, self.m, joints) if joints is not None else None
            return list(self.detector(detector_image, cal, arm_mask=arm, bowls_mask=bm))
        except Exception as exc:
            # Keep the loop alive while recording the exact failure.  The old
            # bare except made calibration, joint-shape, and detector errors
            # indistinguishable from a genuinely empty camera frame.
            row = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "image_shape": tuple(int(x) for x in np.asarray(image).shape),
                "joint_shape": tuple(int(x) for x in np.asarray(obs.get("joints")).shape)
                if isinstance(obs, dict) and obs.get("joints") is not None else None,
                "count": len(self._refresh_errors) + 1,
            }
            self._refresh_errors.append(row)
            self._emit("perception_error", **row)
            return []

    def _label(self, token: int) -> str:
        events = list(getattr(self.env, "_events", []))
        verdicts = [str(e.get("verdict", "")).lower() for e in events
                    if e.get("event") == "verdict" and e.get("brick") == token]
        # The rollout inspector attaches its verdict to the physical body id,
        # while the camera controller uses an opaque token.  If the ids differ,
        # consume the most recent present-frame verdict instead of silently
        # routing every brick to the defective bowl.
        if not verdicts:
            verdicts = [str(e.get("verdict", "")).lower() for e in reversed(events)
                        if e.get("event") == "verdict"][:1]
        if any(v in {"defective", "bad", "-1"} for v in verdicts):
            return "defective"
        if any(v in {"good", "+1", "1"} for v in verdicts):
            return "good"
        return "defective"  # no verdict routes conservatively to the bad bowl

    @staticmethod
    def _target_key(det: BrickDetection) -> tuple[Any, ...]:
        # Retain the measured geometry so failed-target quarantine can absorb
        # the larger residual jitter seen in the unreachable-row frames.
        return (
            str(det.colour),
            round(float(det.centre_m[0]), 4),
            round(float(det.centre_m[1]), 4),
        )

    def _failed_target(self, det: BrickDetection) -> bool:
        """Return whether a detection lies in a quarantined failed region."""
        colour = str(det.colour)
        x, y = float(det.centre_m[0]), float(det.centre_m[1])
        return any(
            colour == failed_colour
            and math.hypot(x - failed_x, y - failed_y) <= .030
            for failed_colour, failed_x, failed_y in self._failed_tokens.values()
        )

    def _plan_cycle(self, det: BrickDetection):
        """Preflight all camera-target IK before starting a motion generator."""
        x, y, _ = det.centre_m
        # Centre the brick in the 49.9 mm open gap, leaving room for camera error.
        jaw_ang = det.yaw_rad + math.pi / 2.0
        jaw = np.array((math.cos(jaw_ang), math.sin(jaw_ang), 0.0))
        qn = self._ik((.22, 0.0, .14), pitch=.6)
        qh = self._ik((x, y, .075), jaw=jaw, q0=qn) if qn is not None else None
        # The camera-grasp debug set found the previous target about 4.4 mm
        # above the measured brick-centre height; XY and yaw were within a few
        # millimetres/degrees.  Keep the approach and lift poses unchanged.
        qg = self._ik((x, y, .0130), jaw=jaw, q0=qh) if qh is not None else None
        ql = self._ik((x, y, .07), jaw=jaw, q0=qg) if qg is not None else None
        qpark = self._ik(CAMERA_PARK_TCP, pitch=.6)
        if any(q is None for q in (qn, qh, qg, ql, qpark)):
            return None
        return qn, qh, qg, ql, qpark

    def _cycle(self, det: BrickDetection, token: int, plan):
        x, y, _ = det.centre_m
        qn, qh, qg, ql, qpark = plan
        self.current = SimpleNamespace(body=token, token=token, detection=det)
        self._emit("reach", brick=token)
        yield from self._move(qn, Q_OPEN, 1.0, "neutral")
        yield from self._move(qh, Q_OPEN, 1.0, "hover")
        yield from self._move(qg, Q_OPEN, .7, "down")
        yield from self._hold(.2, "touch")
        yield from self._move(qg, Q_CLOSE, .6, "close")
        # The station refreshes observations on a 30-tick boundary.  Closing
        # (18 ticks) plus the old 0.3 s hold (9 ticks) read the jaw before that
        # boundary, so a stale open value could be reported as a grip miss.
        # Hold for 12 ticks: close + hold is now exactly one fresh boundary.
        yield from self._hold(.4, "grip_check")
        joints = np.asarray(self._latest_obs.get("joints", [0] * 6)).reshape(-1)
        self._emit(
            "grip_check",
            brick=token,
            gripper=float(joints[5]) if len(joints) >= 6 else None,
            joint_count=int(len(joints)),
            close_hold_ticks=30,
        )
        if len(joints) >= 6 and float(joints[5]) > .88:
            self._emit("miss", brick=token, reason="camera_gripper_reading", gripper=float(joints[5]))
            return
        yield from self._move(ql, Q_CLOSE, .8, "lift")
        for pose, q in (("a", PRESENT_A), ("b", PRESENT_B)):
            yield from self._move(q, Q_CLOSE, 1.2 if pose == "a" else 1.0, "present_" + pose)
            yield from self._hold(.2, "present_" + pose)
            try:
                self.env._force_inspection = True
            except Exception:
                pass
            self._emit("present", brick=token, pose=pose)
            yield from self._hold(.2)
        label = self._label(token)
        bowl = GOOD_BOWL if label == "good" else BAD_BOWL
        qd = self._ik((bowl[0], bowl[1], .115), pitch=.6, q0=PRESENT_B)
        if qd is None:
            self._emit("ik_fail", brick=token, reason="camera_bowl_unreachable")
            return
        yield from self._move(qd, Q_CLOSE, 1.2, "drop")
        yield from self._move(qd, Q_OPEN, .5, "release")
        yield from self._hold(.4)
        self._emit("drop", brick=token, bowl=label)
        # Park outside the top camera's table view before the next look.  The
        # arm mask then cannot hide the remaining bricks after a drop.
        yield from self._move(qpark, Q_OPEN, 1.0, "camera_park")
        self._camera_parked = True
        self.current = None

    def act(self, obs):
        self._latest_obs = obs if isinstance(obs, dict) else {}
        if self._done:
            return self._cmd.copy()
        if self._motion is not None:
            try:
                return np.asarray(next(self._motion), dtype=np.float32)
            except StopIteration:
                self._motion = None; self.current = None; self.phase = "look"
                if self._parking:
                    self._parking = False
                    self._camera_parked = True
        if self.phase == "look":
            found = [d for d in self._refresh(self._latest_obs) if d.status == "ok" and d.confidence >= .2]
            if found:
                self._look_misses = 0
                ordered = sorted(found, key=lambda d: d.centre_m[0] ** 2 + d.centre_m[1] ** 2)
                for det in ordered:
                    if self._failed_target(det):
                        continue
                    key = self._target_key(det)
                    self._token += 1
                    token = self._token
                    self._detections.append(det)
                    plan = self._plan_cycle(det)
                    if plan is None:
                        self._failed_tokens[token] = key
                        self._emit("ik_fail", brick=token, reason="camera_estimate_unreachable", target_key=key)
                        # Do not spin at 30 Hz on an unreachable target.  The
                        # next neutral-pose look will skip this failed key.
                        self._motion = self._hold(.2, "ik_fail")
                        return np.asarray(next(self._motion), dtype=np.float32)
                    self._camera_parked = False
                    self._motion = self._cycle(det, token, plan)
                    return np.asarray(next(self._motion), dtype=np.float32)
            self._look_misses += 1
            # Never spend the miss budget while the arm is still over the
            # table.  Move to the camera-safe park pose first, then look again.
            if not self._camera_parked:
                qpark = self._ik(CAMERA_PARK_TCP, pitch=.6)
                if qpark is not None:
                    # Misses collected before parking were taken with the arm
                    # over the table and must not count against the parked
                    # camera's look budget.
                    self._look_misses = 0
                    self._parking = True
                    self._motion = self._move(qpark, Q_OPEN, 1.0, "camera_park")
                    return np.asarray(next(self._motion), dtype=np.float32)
                self._look_misses = 0
            if self._look_misses >= self.max_look_misses:
                self._done = True; self.phase = "done"
                return self._cmd.copy()
            self._motion = self._hold(1.0, "look")
            return np.asarray(next(self._motion), dtype=np.float32)
        return self._cmd.copy()

    action = act


__all__ = ["CameraExpert"]
