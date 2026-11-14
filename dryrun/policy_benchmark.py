"""Closed-loop policy evaluation through :mod:`dryrun.rollout`."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from .train_policy import load_policy
from .rollout import run


class ACTController:
    def __init__(self, checkpoint, device: str = "cpu"):
        self.policy = load_policy(checkpoint, device=device)
        self.device = device
        self._ticks = 0
        self._last = np.zeros(6, dtype=np.float32)

    def act(self, obs):
        if self._ticks and self._ticks % 30:
            self._ticks += 1
            return self._last
        top = torch.from_numpy(np.asarray(obs["top"]).transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        inspection = torch.from_numpy(np.asarray(obs["inspection"]).transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        if top.shape[-2:] != (64, 64):
            top = torch.nn.functional.interpolate(top, (64, 64), mode="bilinear", align_corners=False)
            inspection = torch.nn.functional.interpolate(inspection, (64, 64), mode="bilinear", align_corners=False)
        joints = torch.from_numpy(np.asarray(obs["joints"], dtype=np.float32)).reshape(1, 6).to(self.device)
        verdict = torch.as_tensor(obs.get("verdict", [0]), dtype=torch.float32).reshape(1, 1).to(self.device)
        with torch.no_grad():
            action = self.policy.select_action({"observation.images.top": top, "observation.images.inspection": inspection, "observation.state": torch.cat([joints, verdict], dim=1)})
        self._last = action.detach().cpu().numpy().reshape(6)
        self._ticks += 1
        return self._last


class OpenVINOController:
    def __init__(self, model_path):
        import openvino as ov
        self.compiled = ov.Core().compile_model(str(model_path), "CPU")
        self.request = self.compiled.create_infer_request()
    def act(self, obs):
        top = np.asarray(obs["top"], np.float32).transpose(2,0,1)[None] / 255.0
        inspection = np.asarray(obs["inspection"], np.float32).transpose(2,0,1)[None] / 255.0
        state = np.r_[obs["joints"], obs.get("verdict", [0])].astype(np.float32)[None]
        return np.asarray(self.request.infer({"top": top, "inspection": inspection, "state": state})[self.compiled.output(0)]).reshape(6)


def one(seed, tier, controller, checkpoint):
    policy = ACTController(checkpoint) if controller == "policy" else "expert"
    return run(seed, policy, tier=tier)


def evaluate(checkpoint: str | Path, seeds: int = 200, workers: int = 8, controller: str = "policy", max_steps: int = 1800, export_replays: Path | None = None):
    if seeds == 200:
        groups = {"loose": list(range(0, 80)), "messy": list(range(80, 150)), "pile": list(range(150, 200))}
    else:
        groups = {"loose": list(range(seeds))}
    # ACT is loaded once per worker invocation in the serial path. Threads
    # keep the expensive MuJoCo rendering parallel without duplicating files.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if export_replays is not None:
        # Import lazily to keep the policy module usable on training-only
        # installs and to share the exact benchmark replay contract.
        from dataclasses import asdict
        from .benchmark import _ReplayCapture, _commit, _write_replay
        from .env import TIERS
        from .mjcf_station import layout_bricks
    all_rows = {}
    for tier, values in groups.items():
        rows = []
        if controller == "policy":
            shared_controller = ACTController(checkpoint) if max_steps <= 1 else None
            # n_action_steps=1 leaves no stateful action queue between calls.
            def task(seed):
                controller_obj = shared_controller or ACTController(checkpoint)
                # A short horizon ends before `present`; avoid compiling the
                # inspector for every smoke worker. Full-cycle runs still use
                # the real OpenVINO inspector.
                if max_steps >= 300:
                    try:
                        from .inspector import Inspector
                        inspector = Inspector()
                    except Exception:
                        inspector = None
                else:
                    inspector = None
                capture = _ReplayCapture() if export_replays is not None else None
                result = run(seed, controller_obj, inspector=inspector, record=capture, tier=tier, max_steps=max_steps, render_size=16 if max_steps <= 1 else 64)
                return result, capture
        else:
            def task(seed): return one(seed, tier, "expert", checkpoint), None
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(task, seed) for seed in values]
            for future in as_completed(futures):
                result, capture = future.result()
                if export_replays is not None and capture is not None:
                    specs = layout_bricks(int(result["seed"]), TIERS[tier])
                    row = {
                        "tier": tier,
                        "seed": int(result["seed"]),
                        "bricks": result["bricks"],
                        "successes": result["successes"],
                        "score": result["score"],
                        "picked_by_contact": sum(int(e.get("picked_by_contact", False)) for e in result.get("events", [])),
                        "verdicts": sum(1 for e in result.get("events", []) if e.get("event") == "verdict"),
                        "scene": [asdict(spec) for spec in specs],
                        "events": result.get("events", []),
                        "controller": controller,
                        "commit": _commit(),
                    }
                    _write_replay(export_replays, row, capture)
                    rows.append(row)
                else:
                    rows.append(result)
        bricks = sum(r["bricks"] for r in rows); successes = sum(r["successes"] for r in rows)
        all_rows[tier] = {"seeds": len(rows), "bricks": bricks, "successes": successes, "score": successes / bricks if bricks else 0.0, "rows": sorted(rows, key=lambda x: x["seed"])}
    total_bricks = sum(v["bricks"] for v in all_rows.values()); total_successes = sum(v["successes"] for v in all_rows.values())
    return {"controller": controller, "checkpoint": str(checkpoint), "held_out_seeds": seeds, "tiers": all_rows, "score": total_successes / total_bricks if total_bricks else 0.0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--controller", choices=("policy", "expert"), default="policy"); ap.add_argument("--checkpoint", type=Path, required=False, default=Path("/bulk2/runs/dryrun/m4/act-final.pt")); ap.add_argument("--seeds", type=int, default=200); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-steps", type=int, default=1800); ap.add_argument("--export-replays", type=Path); ap.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m4/policy-benchmark.json")); a = ap.parse_args()
    result = evaluate(a.checkpoint, a.seeds, a.workers, a.controller, max_steps=a.max_steps, export_replays=a.export_replays); result["max_steps"] = a.max_steps; a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2)); print(json.dumps({k: v for k, v in result.items() if k != "tiers"}, indent=2)); print(json.dumps({k: {x: v[x] for x in ("seeds", "bricks", "successes", "score")} for k, v in result["tiers"].items()}, indent=2))


if __name__ == "__main__": main()
