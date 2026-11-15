"""Single rollout entry point for the expert, learned policy and demos."""
from __future__ import annotations
from typing import Any
import numpy as np
from .env import StationEnv, TIERS

EVENT_NAMES = ("reach", "grasp", "lift", "present", "verdict", "drop", "push", "miss", "slip", "ik_fail", "unstable")

def _call(controller, obs, env):
    if controller is None or controller == "expert": return env.expert_action()
    if hasattr(controller, "act"): return controller.act(obs)
    if callable(controller):
        try: return controller(obs)
        except TypeError: return controller(env, obs)
    raise TypeError("controller must be 'expert', callable, or expose act(observation)")

def _record(record, payload):
    if record is None: return
    if hasattr(record, "record"): record.record(payload)
    elif callable(record): record(payload)
    else: raise TypeError("record must be callable or expose record(payload)")

def run(seed: int, controller: Any = "expert", inspector=None, record=None, *, tier: str | None = None, max_steps: int = 3600, max_cycles: int | None = None, count: int | None = None, render_size: int = 64, sentence: str | None = None, station_path: str | None = None, look: str = "lab"):
    """Run one scene through the public :class:`StationEnv` boundary.

    Expert, learned-policy, and demonstration runs deliberately share the
    same reset/observe/step loop.  This keeps observations, physics events,
    and recorder payloads identical across controllers.
    """
    tier = tier or "loose"
    if tier not in TIERS: raise ValueError(f"unknown tier {tier}")
    # StationEnv accepts the optional checked sentence/path on newer builds;
    # retain a small compatibility fallback for callers using an older local
    # environment while the rollout API migrates.
    kwargs = {"tier": tier, "render_size": render_size, "look": look}
    if sentence is not None: kwargs["sentence"] = sentence
    if station_path is not None: kwargs["station_path"] = station_path
    if count is not None: kwargs["count"] = count
    try:
        env = StationEnv(int(seed), **kwargs)
    except TypeError:
        env = StationEnv(int(seed), tier=tier, render_size=render_size, look=look)
    obs = env.reset()
    _record(record, {"type": "cycle_start", "brick": None, "task": f"sort {tier}"})
    verdict = 0; steps = 0; seen: set[tuple[Any, ...]] = set(); cycles = 0
    for steps in range(1, max_steps + 1):
        action = np.asarray(_call(controller, obs, env), dtype=np.float32).reshape(-1)
        if action.shape != (6,): raise ValueError("controller must return six joint targets")
        obs, _, done, info = env.step(action)
        # ``info.events`` is a short cumulative tail.  Process each physical
        # event once, and let the inspector attach its verdict to the same
        # environment event stream at presentation time.
        for event in info.get("events", []):
            key = tuple(sorted((str(k), repr(v)) for k, v in event.items())) if isinstance(event, dict) else (repr(event),)
            if key in seen: continue
            seen.add(key)
            if isinstance(event, dict) and event.get("event") == "present" and inspector is not None:
                decision = inspector(obs["inspection"])
                label = decision.get("verdict", decision) if isinstance(decision, dict) else decision
                text = str(label).lower()
                verdict = 1 if text in {"good", "+1", "1"} else (-1 if text in {"defective", "-1"} else 0)
                env._event("verdict", event.get("brick", getattr(env, "_active", None)), verdict=label)
            if isinstance(event, dict) and event.get("event") in {"drop", "miss", "slip", "unstable"}:
                cycles += 1
        _record(record, {"t": float(env.data.time), "step": steps, "observation": obs, "action": action, "verdict": verdict, "events": info.get("events", [])})
        # ``drop`` is emitted when the planner enters its drop motion.  Wait
        # for the release/neutral phase to finish before honoring max_cycles,
        # otherwise a bounded smoke run would stop while the brick is still
        # in the gripper and report a false drop failure.
        planner = getattr(env, "_planner", None)
        cycle_finished = bool(
            max_cycles is not None
            and cycles >= int(max_cycles)
            and getattr(planner, "phase", None) == "neutral"
            and getattr(planner, "current", None) is None
        )
        if done or getattr(planner, "phase", None) == "done" or cycle_finished: break
    # A brick in the wrong bowl is a physical completion with an incorrect
    # task outcome; do not mark that cycle as a successful demonstration.
    result = env.result(); success = bool(result.get("correct"))
    _record(record, {"type": "cycle_end", "success": success})
    result.update({
        "seed": int(seed), "tier": tier, "steps": steps,
        "bricks": len(env.brick_ids),
        "successes": len(result.get("correct", [])),
        "score": len(result.get("correct", [])) / max(1, len(env.brick_ids)),
        "observations": {"top": (render_size, render_size, 3),
                         "inspection": (render_size, render_size, 3),
                         "joints": (6,)},
    })
    env.close()
    return result

__all__ = ["run", "EVENT_NAMES"]
