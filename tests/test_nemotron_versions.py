"""The Nemotron ablation versions must change the camera request itself."""
from pathlib import Path

from dryrun.nemotron import FailureInput, Frame, build_request, frames_for_version, score_diagnoses


ROOT = Path(__file__).resolve().parents[1]


def _input() -> FailureInput:
    base = ROOT / "data" / "nemotron_failures300.jsonl"
    import json

    row = json.loads(base.read_text().splitlines()[0])
    frames = tuple(
        Frame(str(path), "top" if i % 2 == 0 else "inspection", i / 30)
        for i, path in enumerate(row["frames"])
    )
    return FailureInput(row["id"], frames, row["scene_features"])


def test_ablation_frame_envelopes_are_distinct():
    inp = _input()
    assert len(frames_for_version(inp.frames, "v1-one-frame")) == 1
    assert len(frames_for_version(inp.frames, "v2-both-cameras")) == 6


def test_ablation_prompts_add_chain_and_fewshot_in_order():
    inp = _input()
    v2 = build_request(inp, version="v2-both-cameras")
    v3 = build_request(inp, version="v3-observation-chain")
    v4 = build_request(inp, version="v4-both-chain-fewshot")
    assert len(v2["messages"]) == 1
    assert len(v3["messages"]) == 1
    assert len(v4["messages"]) == 5
    assert "without an observation chain" in v2["messages"][0]["content"][0]["text"]
    assert "observation chain" in v3["messages"][0]["content"][0]["text"]


def test_scoring_aligns_out_of_order_provider_rows_by_id():
    labels = [
        {"id": "a", "failure_stage": "slip", "cause": "same"},
        {"id": "b", "failure_stage": "drop", "cause": "other"},
    ]
    predictions = [
        {"id": "b", "failure_stage": "drop", "cause": "other"},
        {"id": "a", "failure_stage": "slip", "cause": "same"},
    ]
    assert score_diagnoses(labels, predictions) == {
        "n": 2, "stage_accuracy": 1.0, "cause_accuracy": 1.0, "accuracy": 1.0
    }
