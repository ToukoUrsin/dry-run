"""Replay recorded one-brick targets through the public rollout loop.

This deliberately performs no model inference or network calls.  It compares
the target read from each recorded frame with the target actually passed to
StationEnv.step after the policy loop's per-tick clamp.
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from dryrun.rollout import run


class ReplayController:
    def __init__(self, actions: np.ndarray, *, prefix_ticks: int = 36, max_delta: float = 0.06):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.prefix_ticks = int(prefix_ticks)
        self.max_delta = float(max_delta)
        self.i = 0
        self.sent: list[list[float]] = []
        self.targets: list[list[float]] = []
        self.clipped = 0
        self.recorded_clipped = 0
        self.max_abs_mismatch = 0.0
        self.abs_mismatch = 0.0
        self.recorded_abs_mismatch = 0.0
        self.recorded_max_abs_mismatch = 0.0

    def __call__(self, env, obs):
        # The recorder's first frame is after the expert's first 36 approach
        # ticks (frame step 37).  Recreate that approach before replaying the
        # 301 stored targets so the brick and gripper are at the same phase.
        if self.prefix_ticks and env._ticks < self.prefix_ticks:
            target = np.asarray(env.expert_action(), dtype=np.float32)
        else:
            target = self.actions[min(self.i, len(self.actions) - 1)]
            self.i += 1
        current = np.asarray(obs["joints"], dtype=np.float32)
        sent = np.clip(target, current - self.max_delta, current + self.max_delta)
        diff = np.abs(target - sent)
        self.targets.append(target.tolist()); self.sent.append(sent.tolist())
        self.abs_mismatch += float(diff.sum()); self.max_abs_mismatch = max(self.max_abs_mismatch, float(diff.max()))
        if env._ticks >= self.prefix_ticks:
            self.recorded_abs_mismatch += float(diff.sum())
            self.recorded_max_abs_mismatch = max(self.recorded_max_abs_mismatch, float(diff.max()))
            if np.any(diff > 1e-7): self.recorded_clipped += 1
        if np.any(diff > 1e-7): self.clipped += 1
        return sent


def load_actions(root: Path) -> list[np.ndarray]:
    table = pq.read_table(next(root.glob("data/chunk-*/file-*.parquet")), columns=["episode_index", "action", "observation.state"])
    eps = np.asarray(table["episode_index"]); actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    return [(actions[eps == i], states[eps == i]) for i in range(int(eps.max()) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--seeds", default="10020,10022,10023,10025,10026,10027,10030,10031,10033,10035,10036,10040,10041,10042,10044,10046,10049,10052,10053,10058")
    ap.add_argument("--max-steps", type=int, default=330)
    ap.add_argument("--prefix-ticks", type=int, default=36)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    actions = load_actions(args.dataset)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for episode, seed in enumerate(seeds):
        episode_actions, episode_states = actions[episode]
        controller = ReplayController(episode_actions, prefix_ticks=args.prefix_ticks)
        result = run(seed, controller, inspector=None, tier="loose", count=1,
                      max_steps=args.max_steps, max_cycles=1, render_size=16)
        rows.append({
            "episode": episode, "seed": seed, "frames": int(len(episode_actions)),
            "steps": int(result["steps"]), "successes": int(result["successes"]),
            "correct": result.get("correct", []), "events": result.get("events", []),
            "clipped_ticks": controller.clipped,
            "recorded_clipped_ticks": controller.recorded_clipped,
            "target_sent_mae": controller.recorded_abs_mismatch / max(1, len(episode_actions) * 6),
            "target_sent_max_abs": controller.recorded_max_abs_mismatch,
            "recorded_action_vs_previous_state_mae": float(np.abs(episode_actions[1:] - episode_states[:-1]).mean()),
            "recorded_action_vs_previous_state_max_abs": float(np.abs(episode_actions[1:] - episode_states[:-1]).max()),
            "sent_last": controller.sent[-1] if controller.sent else None,
        })
    summary = {
        "format": "dryrun-recorded-action-replay-v1", "dataset": str(args.dataset),
        "seeds": seeds, "episodes": len(rows), "max_steps": args.max_steps,
        "prefix_ticks": args.prefix_ticks, "max_delta": 0.06,
        "successes": sum(r["successes"] for r in rows),
        "clipped_ticks": sum(r["clipped_ticks"] for r in rows),
        "recorded_clipped_ticks": sum(r["recorded_clipped_ticks"] for r in rows),
        "target_sent_mae": float(np.mean([r["target_sent_mae"] for r in rows])),
        "target_sent_max_abs": max(r["target_sent_max_abs"] for r in rows),
        "recorded_action_vs_previous_state_mae": float(np.mean([r["recorded_action_vs_previous_state_mae"] for r in rows])),
        "recorded_action_vs_previous_state_max_abs": max(r["recorded_action_vs_previous_state_max_abs"] for r in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "successes", "clipped_ticks", "target_sent_mae", "target_sent_max_abs")}, indent=2))


if __name__ == "__main__": main()
