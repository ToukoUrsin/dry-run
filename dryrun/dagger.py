"""Simulator DAgger rounds with expert labels on every reached state."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .demos import FEATURES
from .env import StationEnv
from .policy_benchmark import ACTController, evaluate
from .train_policy import train

RUN_ROOT = Path("/bulk3/runs/dryrun")


def _as_image(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return array


def _episode_lengths(ds) -> list[int]:
    """Read real episode lengths instead of assuming a 320-frame boundary."""
    episodes = getattr(getattr(ds, "meta", None), "episodes", None)
    try:
        if hasattr(episodes, "columns") and "length" in episodes.columns:
            values = [int(v) for v in episodes["length"].tolist()]
            if values and sum(values) == len(ds):
                return values
        if isinstance(episodes, dict) and "length" in episodes:
            values = [int(v) for v in episodes["length"]]
            if values and sum(values) == len(ds):
                return values
    except Exception:
        pass
    return [len(ds)]


def _chw_to_hwc(value):
    if hasattr(value, "permute"):
        return value.permute(1, 2, 0).numpy()
    return value


def _copy_row(row: dict) -> dict:
    def vector(name, size, dtype=np.float32):
        value = row.get(name)
        if value is None:
            return np.zeros(size, dtype=dtype)
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=dtype)

    task = row.get("task", "sort brick")
    return {
        "observation.images.top": _as_image(_chw_to_hwc(row["observation.images.top"])),
        "observation.images.inspection": _as_image(_chw_to_hwc(row["observation.images.inspection"])),
        "observation.state": vector("observation.state", 6),
        "action": vector("action", 6),
        "inspection.verdict": vector("inspection.verdict", 1, np.int8).reshape(1),
        "success": vector("success", 1, np.bool_).reshape(1),
        "task": task if isinstance(task, str) else "sort brick",
    }


def _build_dataset(source: Path, target: Path, extra_rows: list[dict]) -> int:
    if target.exists():
        shutil.rmtree(target)
    ds = LeRobotDataset("dryrun-so101", root=source, download_videos=False)
    out = LeRobotDataset.create(
        "dryrun-so101", fps=30, features=FEATURES, root=target,
        robot_type="so101", use_videos=False, image_writer_processes=0,
        image_writer_threads=0,
    )
    boundaries = set()
    cursor = 0
    for length in _episode_lengths(ds):
        cursor += int(length)
        boundaries.add(cursor)
    for index in range(len(ds)):
        out.add_frame(_copy_row(ds[index]))
        if index + 1 in boundaries:
            out.save_episode()
    for row in extra_rows:
        out.add_frame(row)
    if extra_rows:
        out.save_episode()
    out.finalize()
    return len(ds)


def _expert_rows(checkpoint: Path, round_no: int, seeds: int, max_steps: int):
    rows: list[dict] = []
    for seed in range(int(seeds)):
        env = StationEnv(seed, "loose")
        policy = ACTController(checkpoint)  # reset state for every environment
        try:
            obs = env.reset()
            verdict = 0
            for _ in range(int(max_steps)):
                policy_action = policy.act({**obs, "verdict": np.asarray([verdict], np.float32)})
                expert_action = np.asarray(env.expert_action(), dtype=np.float32).reshape(6)
                rows.append({
                    "observation.images.top": obs["top"],
                    "observation.images.inspection": obs["inspection"],
                    "observation.state": np.asarray(obs["joints"], dtype=np.float32),
                    "action": expert_action,
                    "inspection.verdict": np.asarray([verdict], dtype=np.int8),
                    "success": np.asarray([False], dtype=np.bool_),
                    "task": f"dagger round {round_no}",
                })
                obs, _, done, info = env.step(policy_action)
                for event in info.get("events", []):
                    if event.get("event") == "verdict":
                        label = str(event.get("verdict", "")).lower()
                        verdict = 1 if label in {"good", "+1", "1"} else -1 if label in {"defective", "-1"} else 0
                if done:
                    break
        finally:
            env.close()
    return rows


def append_round(source: Path, target: Path, checkpoint: Path, round_no: int, seeds: int = 2, max_steps: int = 3600):
    """Copy source episodes and append expert labels from policy-reached states."""
    if not source.resolve().is_relative_to(RUN_ROOT.resolve()):
        raise ValueError(f"source must be under {RUN_ROOT}: {source}")
    if not target.resolve().is_relative_to(RUN_ROOT.resolve()):
        raise ValueError(f"target must be under {RUN_ROOT}: {target}")
    rows = _expert_rows(checkpoint, round_no, seeds, max_steps)
    source_frames = _build_dataset(source, target, rows)
    total = len(LeRobotDataset("dryrun-so101", root=target, download_videos=False))
    return {"source": str(source), "target": str(target), "round": round_no,
            "source_frames": source_frames, "expert_labeled_frames": len(rows),
            "frames": total}


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _score_rows(stage: str, checkpoint: str, result: dict) -> list[dict]:
    rows = []
    for tier, summary in result.get("tiers", {}).items():
        for row in summary.get("rows", []):
            rows.append({
                "controller": stage,
                "checkpoint": str(checkpoint),
                "seed": int(row["seed"]),
                "tier": tier,
                "bricks": int(row.get("bricks", 0)),
                "correct": int(row.get("successes", 0)),
                "score": float(row.get("score", 0.0)),
                "events": row.get("events", []),
                "commit": _commit(),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=RUN_ROOT / "m4" / "lerobot-real-current")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=RUN_ROOT / "m4" / "dagger-current")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--score-seeds", type=int, default=0, help="held-out smoke size; 0 skips scoring")
    ap.add_argument("--score-workers", type=int, default=1)
    ap.add_argument("--score-max-steps", type=int, default=3600)
    a = ap.parse_args()
    if a.seeds < 1 or a.max_steps < 1 or a.score_seeds < 0:
        ap.error("seeds and horizons must be positive")
    if not a.out.resolve().is_relative_to(RUN_ROOT.resolve()):
        ap.error(f"--out must be under {RUN_ROOT}")
    a.out.mkdir(parents=True, exist_ok=True)
    source, checkpoint = a.dataset, a.checkpoint
    records = []
    score_rows = []
    if a.score_seeds:
        scored = evaluate(checkpoint, a.score_seeds, a.score_workers, "policy", a.score_max_steps)
        records.append({"stage": "demos-only", "checkpoint": str(checkpoint), "result": scored})
        score_rows.extend(_score_rows("demos-only", str(checkpoint), scored))
    for round_no in (1, 2):
        target = a.out / f"round-{round_no}"
        row = append_round(source, target, checkpoint, round_no, a.seeds, a.max_steps)
        training = train(target, out=a.out / f"train-{round_no}", epochs=a.epochs, batch_size=a.batch_size, checkpoint_every=a.checkpoint_every)
        row["training"] = training
        row["checkpoint"] = training["checkpoint"]
        records.append(row)
        if a.score_seeds:
            scored = evaluate(training["checkpoint"], a.score_seeds, a.score_workers, "policy", a.score_max_steps)
            records.append({"stage": f"dagger-round-{round_no}", "checkpoint": training["checkpoint"], "result": scored})
            score_rows.extend(_score_rows(f"dagger-round-{round_no}", training["checkpoint"], scored))
        source, checkpoint = target, Path(training["checkpoint"])
    payload = {"format": "dryrun-dagger-v2", "source_commit": _commit(), "records": records}
    (a.out / "rounds.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    if score_rows:
        score_path = Path("data/policy_scores.jsonl")
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with score_path.open("a") as fh:
            for row in score_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
