"""Camera-only sorter boundary tests."""
from __future__ import annotations

import numpy as np
import mujoco

from dryrun.camera_expert import CameraExpert
from dryrun.env import StationEnv


class _GuardQpos:
    """Permit only the six measured arm coordinates from live qpos."""

    def __init__(self, source, allowed):
        self._source = source
        self._allowed = {int(i) for i in allowed}

    def __getitem__(self, key):
        if not isinstance(key, (int, np.integer)) or int(key) not in self._allowed:
            raise AssertionError(f"camera sorter read forbidden brick qpos index: data.qpos[{key!r}]")
        return self._source[int(key)]


class _GuardData:
    """Expose joint readings and time; fail on brick pose arrays."""

    def __init__(self, source, allowed_qpos):
        self._source = source
        self.qpos = _GuardQpos(source.qpos, allowed_qpos)
        self.time = source.time

    def __getattr__(self, name):
        if name in {"xpos", "xmat", "xquat", "contact", "ncon"}:
            raise AssertionError(f"camera sorter read forbidden simulator state: data.{name}")
        return getattr(self._source, name)


class _GuardEnv:
    def __init__(self, source, allowed_qpos):
        self.model = source.model
        self.data = _GuardData(source.data, allowed_qpos)
        self._events = []
        self._force_inspection = False

    def _event(self, event, brick=None, **extra):
        row = {"event": event, **extra}
        if brick is not None:
            row["brick"] = brick
        self._events.append(row)


def test_camera_expert_does_not_read_brick_pose_arrays():
    source = StationEnv(1000, tier="loose", render_size=32)
    try:
        arm_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        allowed = [source.model.jnt_qposadr[mujoco.mj_name2id(source.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in arm_names]
        env = _GuardEnv(source, allowed)
        expert = CameraExpert(env)
        obs = {
            "top": np.zeros((32, 32, 3), dtype=np.uint8),
            "inspection": np.zeros((32, 32, 3), dtype=np.uint8),
            "joints": np.asarray([source.data.qpos[a] for a in expert.qa] + [source.data.qpos[expert.gq]], dtype=np.float32),
        }
        action = expert.act(obs)
        assert action.shape == (6,)
        assert np.isfinite(action).all()
    finally:
        source.close()
