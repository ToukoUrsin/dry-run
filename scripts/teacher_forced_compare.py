"""Compare live and recorded inputs/actions on one replayed episode."""
from __future__ import annotations
import io, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from dryrun.env import StationEnv
from dryrun.train_policy import load_policy


def decode(cell):
    return np.asarray(Image.open(io.BytesIO(cell["bytes"].as_py())).convert("RGB"), dtype=np.uint8)


def policy_action(policy, obs, verdict):
    top = torch.from_numpy(np.asarray(obs["top"]).transpose(2, 0, 1)).float()[None] / 255.0
    inspection = torch.from_numpy(np.asarray(obs["inspection"]).transpose(2, 0, 1)).float()[None] / 255.0
    state = torch.from_numpy(np.asarray(obs["joints"], dtype=np.float32))[None]
    v = torch.tensor([[float(verdict)]], dtype=torch.float32)
    with torch.no_grad():
        return policy.select_action({"observation.images.top": top, "observation.images.inspection": inspection,
                                     "observation.state": torch.cat([state, v], 1)}).numpy().reshape(6)


def main():
    dataset = Path("/bulk3/runs/dryrun/m4/onebrick-overfit-replay-dataset/onebrick-overfit-20-dataset-20261115-1")
    parquet = next(dataset.glob("data/chunk-*/file-*.parquet"))
    table = pq.read_table(parquet, columns=["episode_index", "observation.images.top", "observation.images.inspection", "observation.state", "action", "inspection.verdict"])
    ep = np.asarray(table["episode_index"]); idx = np.flatnonzero(ep == 0)
    top = [decode(table["observation.images.top"][i]) for i in idx]
    inspection = [decode(table["observation.images.inspection"][i]) for i in idx]
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[idx]
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[idx]
    verdicts = np.asarray(table["inspection.verdict"].to_pylist(), dtype=np.float32).reshape(-1)[idx]
    policy = load_policy("/bulk3/runs/dryrun/m4/onebrick-overfit-checkpoint/act-final.pt", device="cpu")
    env = StationEnv(10020, tier="loose", count=1, render_size=64)
    obs = env.reset()
    for _ in range(36):
        env.step(env.expert_action())
    live_before_top, live_before_inspection, live_before_state = [], [], []
    live_after_top, live_after_inspection, live_after_state = [], [], []
    live_policy, recorded_policy = [], []
    fields = {k: [] for k in ("top_before", "inspection_before", "state_before", "top_after", "inspection_after", "state_after")}
    out_dir = Path("/bulk3/runs/dryrun/m4/modal-map/onebrick-overfit/training/teacher-forced")
    out_dir.mkdir(parents=True, exist_ok=True)
    first_saved = False
    for i in range(len(actions)):
        # Keep the active brick/presentation phase identical while applying the
        # exact recorded target, without calling a network or a policy.
        env.expert_action()
        before = env.observe()
        pa_live = policy_action(policy, before, 0)
        rec_obs = {"top": top[i], "inspection": inspection[i], "joints": states[i]}
        pa_rec = policy_action(policy, rec_obs, verdicts[i])
        live_policy.append(pa_live); recorded_policy.append(pa_rec)
        live_before_top.append(before["top"]); live_before_inspection.append(before["inspection"]); live_before_state.append(before["joints"])
        after, _, _, _ = env.step(actions[i])
        live_after_top.append(after["top"]); live_after_inspection.append(after["inspection"]); live_after_state.append(after["joints"])
        diffs = {
            "top_before": np.abs(before["top"].astype(np.float32) - top[i].astype(np.float32)),
            "inspection_before": np.abs(before["inspection"].astype(np.float32) - inspection[i].astype(np.float32)),
            "state_before": np.abs(before["joints"] - states[i]),
            "top_after": np.abs(after["top"].astype(np.float32) - top[i].astype(np.float32)),
            "inspection_after": np.abs(after["inspection"].astype(np.float32) - inspection[i].astype(np.float32)),
            "state_after": np.abs(after["joints"] - states[i]),
        }
        for k, d in diffs.items(): fields[k].append(d)
        if not first_saved and any(float(np.max(d)) > 1e-3 for d in diffs.values()):
            first_saved = True
            for name, arr in (("live-before-top", before["top"]), ("recorded-top", top[i]),
                              ("live-before-inspection", before["inspection"]), ("recorded-inspection", inspection[i])):
                Image.fromarray(arr).save(out_dir / f"first-{i:03d}-{name}.png")
    env.close()
    lp, rp = np.asarray(live_policy), np.asarray(recorded_policy)
    summary = {
        "format": "dryrun-teacher-forced-input-compare-v1", "seed": 10020, "frames": len(actions),
        "prefix_ticks": 36, "checkpoint": "/bulk3/runs/dryrun/m4/onebrick-overfit-checkpoint/act-final.pt",
        "live_policy_vs_recorded_action_mae": float(np.abs(lp - actions).mean()),
        "recorded_input_policy_vs_recorded_action_mae": float(np.abs(rp - actions).mean()),
        "live_policy_vs_recorded_action_mae_per_joint": np.abs(lp - actions).mean(0).tolist(),
        "recorded_input_policy_vs_recorded_action_mae_per_joint": np.abs(rp - actions).mean(0).tolist(),
        "input_diffs": {k: {"mean": float(np.asarray(v).mean()), "max": float(np.asarray(v).max())} for k, v in fields.items()},
        "recorded_inspection_mean": float(np.asarray(inspection, dtype=np.float32).mean()),
        "live_before_inspection_mean": float(np.asarray(live_before_inspection, dtype=np.float32).mean()),
        "recorded_verdict_counts": {str(int(v)): int(np.sum(verdicts == v)) for v in sorted(set(verdicts.tolist()))},
        "live_before_verdict": 0,
        "verdict_mismatch_ticks": int(np.sum(verdicts != 0)),
        "first_difference_dir": str(out_dir),
    }
    out = Path("/bulk3/runs/dryrun/m4/modal-map/onebrick-overfit/training/teacher-forced-compare.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
