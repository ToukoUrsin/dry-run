"""Single rollout entry point for the expert, learned policy and demos."""
from __future__ import annotations
from typing import Any
import numpy as np
from .env import StationEnv, TIERS
from pathlib import Path
import mujoco
from .mjcf_station import ROOT, layout_bricks, station_xml
from .render import rollout as expert_rollout

EVENT_NAMES = ("reach", "grasp", "lift", "present", "verdict", "drop", "miss", "slip", "ik_fail", "unstable")

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

def run(seed: int, controller: Any = "expert", inspector=None, record=None, *, tier: str | None = None, max_steps: int = 1800, max_cycles: int | None = None, count: int | None = None, render_size: int = 64):
    """Run one scene; expert and learned paths expose the same observations."""
    tier = tier or "loose"
    if tier not in TIERS: raise ValueError(f"unknown tier {tier}")
    if controller is None or controller == "expert":
        sentence = TIERS[tier]
        xml = Path("/tmp") / f"dryrun_rollout_{int(seed)}_{tier}.xml"
        xml.write_text(station_xml(int(seed), sentence, count=count, robot_include=str((ROOT / "assets/SO101/Simulation/SO101/so101_dryrun.xml").resolve())))
        model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        result = expert_rollout(model, data, int(seed), specs=layout_bricks(int(seed), sentence, count=count), inspector=inspector, record=record, max_cycles=max_cycles)
        result.update({"seed": int(seed), "tier": tier, "steps": int(round(float(data.time) * 30.0)), "observations": {"top": (64, 64, 3), "inspection": (64, 64, 3), "joints": (6,)}})
        return result
    env = StationEnv(seed, tier=tier, render_size=render_size); obs = env.reset()
    _record(record, {"type": "cycle_start", "brick": None, "task": f"sort {tier}"})
    verdict = 0; steps = 0
    for steps in range(1, max_steps + 1):
        action = np.asarray(_call(controller, obs, env), dtype=np.float32).reshape(-1)
        if action.shape != (6,): raise ValueError("controller must return six joint targets")
        obs, _, done, info = env.step(action)
        if steps == 300:
            env._event("present", env._active)
            if inspector is not None:
                decision = inspector(obs["inspection"])
                label = decision.get("verdict", decision) if isinstance(decision, dict) else decision
                verdict = 1 if str(label).lower() in {"good", "+1", "1"} else (-1 if str(label).lower() in {"defective", "-1"} else 0)
                env._event("verdict", env._active, verdict=label)
        _record(record, {"t": float(env.data.time), "step": steps, "observation": obs, "action": action, "verdict": verdict, "events": info.get("events", [])})
        if done: break
    result = env.result(); success = bool(result.get("good_bowl") or result.get("wrong_bowl"))
    _record(record, {"type": "cycle_end", "success": success})
    result.update({"seed": int(seed), "tier": tier, "steps": steps, "bricks": len(env.brick_ids), "successes": len(result.get("good_bowl", [])), "score": len(result.get("good_bowl", [])) / max(1, len(env.brick_ids))})
    env.close()
    return result

__all__ = ["run", "EVENT_NAMES"]
