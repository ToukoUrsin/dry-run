"""Closed-loop policy evaluation through :mod:`dryrun.rollout`."""
from __future__ import annotations
import argparse, json
from collections import deque
from pathlib import Path
import numpy as np, torch
from .train_policy import load_policy
from .rollout import run


class ACTController:
    def __init__(self, checkpoint, device: str = "cpu", open_loop_ticks: int = 0, max_delta: float = 0.06, crop_top: bool = False, demo_jpeg: bool = False):
        self.policy = load_policy(checkpoint, device=device)
        self.device = device
        self.open_loop_ticks = max(0, int(open_loop_ticks))
        self.max_delta = None if max_delta is None or float(max_delta) <= 0 else float(max_delta)
        self.crop_top = bool(crop_top)
        self.demo_jpeg = bool(demo_jpeg)
        self._action_queue: deque[np.ndarray] = deque()
        shape = self.policy.config.input_features["observation.images.top"].shape
        self.image_size = int(shape[-1])

    def act(self, obs):
        top_frame = obs["top"]
        if self.crop_top:
            from .policy_view import crop_work_area
            top_frame = crop_work_area(top_frame, self.image_size)
        if self.demo_jpeg:
            import io
            from PIL import Image
            def jpeg78(frame):
                buf = io.BytesIO(); Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="JPEG", quality=78, optimize=True)
                return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)
            top_frame = jpeg78(top_frame)
            obs_inspection = jpeg78(obs["inspection"])
        else:
            obs_inspection = obs["inspection"]
        top = torch.from_numpy(np.asarray(top_frame).transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        inspection = torch.from_numpy(np.asarray(obs_inspection).transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        # Cropping can make the top view exactly model-sized while the
        # inspection render remains at the station resolution.  Resize each
        # camera independently so the crop flag never changes the inspection
        # tensor contract.
        if top.shape[-2:] != (self.image_size, self.image_size):
            top = torch.nn.functional.interpolate(top, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if inspection.shape[-2:] != (self.image_size, self.image_size):
            inspection = torch.nn.functional.interpolate(inspection, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        joints = torch.from_numpy(np.asarray(obs["joints"], dtype=np.float32)).reshape(1, 6).to(self.device)
        verdict = torch.as_tensor(obs.get("verdict", [0]), dtype=torch.float32).reshape(1, 1).to(self.device)
        batch = {
            "observation.images.top": top,
            "observation.images.inspection": inspection,
            "observation.state": torch.cat([joints, verdict], dim=1),
        }
        with torch.no_grad():
            if self._action_queue:
                action_np = self._action_queue.popleft()
                current_np = joints.detach().cpu().numpy().reshape(6)
                return np.clip(action_np, current_np - self.max_delta, current_np + self.max_delta) if self.max_delta else action_np
            if self.open_loop_ticks > 1:
                # Explicit fallback experiment: consume a bounded prefix of
                # ACT's predicted chunk before asking for another image.
                chunk = self.policy.predict_action_chunk(batch)[0]
                self._action_queue.extend(
                    row.detach().cpu().numpy().reshape(6)
                    for row in chunk[: self.open_loop_ticks]
                )
                action_np = self._action_queue.popleft()
                current_np = joints.detach().cpu().numpy().reshape(6)
                return np.clip(action_np, current_np - self.max_delta, current_np + self.max_delta) if self.max_delta else action_np
            action = self.policy.select_action(batch)
        # ACT is configured with n_action_steps=1.  A rollout tick is 1/30 s,
        # so evaluate it on every observation rather than holding an action for
        # 29 ticks (which silently turns a 30 Hz policy into a 1 Hz controller).
        action_np = action.detach().cpu().numpy().reshape(6)
        current_np = joints.detach().cpu().numpy().reshape(6)
        return np.clip(action_np, current_np - self.max_delta, current_np + self.max_delta) if self.max_delta else action_np


class OpenVINOController:
    def __init__(self, model_path, crop_top: bool = False, max_delta: float = 0.06, demo_jpeg: bool = False):
        import openvino as ov
        self.compiled = ov.Core().compile_model(str(model_path), "CPU")
        self.request = self.compiled.create_infer_request()
        self.crop_top = bool(crop_top)
        self.max_delta = None if max_delta is None or float(max_delta) <= 0 else float(max_delta)
        self.demo_jpeg = bool(demo_jpeg)
        self.image_size = int(self.compiled.input("top").shape[-1])
    def act(self, obs):
        top_frame = obs["top"]
        if self.crop_top:
            from .policy_view import crop_work_area
            top_frame = crop_work_area(top_frame, self.compiled.input("top").shape[-1])
        inspection_frame = obs["inspection"]
        if self.demo_jpeg:
            import io
            from PIL import Image
            def jpeg78(frame):
                buf = io.BytesIO(); Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="JPEG", quality=78, optimize=True)
                return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)
            top_frame, inspection_frame = jpeg78(top_frame), jpeg78(inspection_frame)
        top = torch.from_numpy(np.asarray(top_frame).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        inspection = torch.from_numpy(np.asarray(inspection_frame).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        if top.shape[-2:] != (self.image_size, self.image_size):
            top = torch.nn.functional.interpolate(top, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if inspection.shape[-2:] != (self.image_size, self.image_size):
            inspection = torch.nn.functional.interpolate(inspection, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        state = np.r_[obs["joints"], obs.get("verdict", [0])].astype(np.float32)[None]
        target = np.asarray(self.request.infer({"top": top.numpy(), "inspection": inspection.numpy(), "state": state})[self.compiled.output(0)]).reshape(6)
        current = np.asarray(obs["joints"], dtype=np.float32).reshape(6)
        return np.clip(target, current - self.max_delta, current + self.max_delta) if self.max_delta else target


def _controller_for(kind: str, checkpoint: str | Path | None, *, open_loop_ticks: int = 0, max_delta: float = 0.06, crop_top: bool = False, demo_jpeg: bool = False):
    if kind == "expert":
        return "expert"
    if checkpoint is None:
        raise ValueError(f"--checkpoint is required for {kind} evaluation")
    if kind == "policy":
        return ACTController(checkpoint, open_loop_ticks=open_loop_ticks, max_delta=max_delta, crop_top=crop_top, demo_jpeg=demo_jpeg)
    if kind == "openvino":
        return OpenVINOController(checkpoint, crop_top=crop_top, max_delta=max_delta, demo_jpeg=demo_jpeg)
    raise ValueError(f"unknown controller {kind!r}")


def evaluate(checkpoint: str | Path | None, seeds: int = 200, workers: int = 8, controller: str = "policy", max_steps: int = 3600, export_replays: Path | None = None, open_loop_ticks: int = 0, max_delta: float = 0.06, crop_top: bool = False, demo_jpeg: bool = False, tier: str | None = None):
    from .benchmark import TIER_COUNTS, TIER_OFFSETS
    if tier is not None and tier not in TIER_COUNTS:
        raise ValueError(f"unknown tier {tier!r}")
    if tier is not None:
        count = min(int(seeds), TIER_COUNTS[tier])
        start = TIER_OFFSETS[tier]
        groups = {tier: list(range(start, start + count))}
    elif seeds == 200:
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
        from .benchmark import _ReplayCapture, _commit, _contact_count, _write_replay
        from .env import TIERS
        from .mjcf_station import layout_bricks
    all_rows = {}
    for tier, values in groups.items():
        rows = []
        if controller in {"policy", "openvino"}:
            shared_controller = _controller_for(controller, checkpoint, open_loop_ticks=open_loop_ticks, max_delta=max_delta, crop_top=crop_top, demo_jpeg=demo_jpeg) if max_steps <= 1 else None
            def task(seed):
                controller_obj = shared_controller or _controller_for(controller, checkpoint, open_loop_ticks=open_loop_ticks, max_delta=max_delta, crop_top=crop_top, demo_jpeg=demo_jpeg)
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
            def task(seed): return run(seed, "expert", tier=tier, max_steps=max_steps), None
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
                        "picked_by_contact": _contact_count(result.get("events", [])),
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
    return {
        "controller": controller,
        "checkpoint": str(checkpoint),
        "held_out_seeds": seeds,
        "tier": tier,
        "seed_partition": {name: values for name, values in groups.items()},
        "tiers": all_rows,
        "score": total_successes / total_bricks if total_bricks else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--controller", choices=("policy", "openvino", "expert"), default="policy"); ap.add_argument("--checkpoint", type=Path, required=False); ap.add_argument("--seeds", type=int, default=200); ap.add_argument("--tier", choices=("loose", "messy", "pile")); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-steps", type=int, default=3600); ap.add_argument("--open-loop-ticks", type=int, default=0, help="consume this many ACT chunk actions open-loop"); ap.add_argument("--max-delta", type=float, default=0.06, help="per-tick joint target clamp in radians"); ap.add_argument("--crop-top", action="store_true", help="apply the shared work-area crop before policy resize"); ap.add_argument("--demo-jpeg", action="store_true", help="match recorder JPEG quality=78 preprocessing"); ap.add_argument("--export-replays", type=Path); ap.add_argument("--out", type=Path, default=Path("/bulk3/runs/dryrun/m4/policy-benchmark.json")); a = ap.parse_args()
    result = evaluate(a.checkpoint, a.seeds, a.workers, a.controller, max_steps=a.max_steps, export_replays=a.export_replays, open_loop_ticks=a.open_loop_ticks, max_delta=a.max_delta, crop_top=a.crop_top, demo_jpeg=a.demo_jpeg, tier=a.tier); result["max_steps"] = a.max_steps; result["open_loop_ticks"] = a.open_loop_ticks; result["max_delta"] = a.max_delta; result["crop_top"] = a.crop_top; result["demo_jpeg"] = a.demo_jpeg; a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2)); print(json.dumps({k: v for k, v in result.items() if k != "tiers"}, indent=2)); print(json.dumps({k: {x: v[x] for x in ("seeds", "bricks", "successes", "score")} for k, v in result["tiers"].items()}, indent=2))


if __name__ == "__main__": main()
