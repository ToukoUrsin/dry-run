"""One rollout path shared by expert, learned policy and demo recording."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any

import numpy as np

from .env import StationEnv

EVENT_NAMES = ("reach", "grasp", "lift", "present", "verdict", "drop", "miss", "slip", "ik_fail", "unstable")


def _action(controller: Any, env: StationEnv, obs: dict[str, np.ndarray]) -> np.ndarray:
    if controller is None or controller == "expert":
        return env.expert_action()
    if hasattr(controller, "act"):
        return np.asarray(controller.act(obs), dtype=np.float32)
    if callable(controller):
        try:
            return np.asarray(controller(obs), dtype=np.float32)
        except TypeError:
            return np.asarray(controller(env, obs), dtype=np.float32)
    raise TypeError("controller must be 'expert', callable, or expose act(obs)")


def run(seed: int, controller: Any = "expert", inspector: Callable | None = None, record: Callable | None = None, *, tier: str | None = None, max_steps: int = 1800) -> dict[str, Any]:
    env = StationEnv(seed, tier=tier)
    obs = env.reset()
    if record:
        record({"t": 0.0, "obs": obs, "action": None, "events": []})
    for step_index in range(max_steps):
        action = _action(controller, env, obs)
        obs, reward, done, info = env.step(action)
        events = list(info.get("events", []))
        if inspector and any(e.get("name") in {"present", "verdict"} for e in events):
            decision = inspector(obs["inspection"])
            env._emit("verdict", env._active, decision=decision)
            events = list(env._events[-8:])
        if record:
            record({"t": float(env.data.time), "step": step_index + 1, "obs": obs, "action": np.asarray(action, dtype=np.float32), "events": events})
        if done:
            break
    result = env.result()
    result.update({"seed": int(seed), "tier": tier or "loose", "steps": step_index + 1, "events": list(env._events), "observations": {k: tuple(v.shape) for k, v in obs.items()}})
    return result
