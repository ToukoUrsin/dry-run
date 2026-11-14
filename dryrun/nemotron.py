"""Nemotron failure analyst.

The network path sends camera evidence only.  Event-log labels are consumed by
the scoring helpers separately, so simulator ground truth cannot leak into a
production prompt.  Requests and every response are retained for review.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
ALLOWED_STAGES = {
    "approach", "ik_fail", "grasp", "miss", "slip", "knocked_neighbour",
    "inspection", "wrong_bowl", "inspector_false_accept", "drop", "unstable",
}
SCENE_FIELDS = ("tier", "brick_count", "spacing", "yaw_spread", "positions", "lighting")
DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["failure_stage", "cause", "frame", "scene_features", "next_scenes"],
    "properties": {
        "failure_stage": {"type": "string", "enum": sorted(ALLOWED_STAGES)},
        "cause": {"type": "string", "minLength": 8, "maxLength": 240},
        "frame": {"type": "string", "minLength": 1},
        "scene_features": {"type": "object"},
        "next_scenes": {"type": "array", "minItems": 1, "maxItems": 32,
            "items": {"type": "object", "additionalProperties": False,
                "required": list(SCENE_FIELDS), "properties": {
                    "tier": {"type": "string", "enum": ["loose", "messy", "pile"]},
                    "brick_count": {"type": "integer", "minimum": 3, "maximum": 12},
                    "spacing": {"type": "number", "minimum": 0, "maximum": 1},
                    "yaw_spread": {"type": "number", "minimum": 0, "maximum": 3.142},
                    "positions": {"type": "array", "minItems": 1},
                    "lighting": {"type": "string", "minLength": 1},
                }}}},
}
# Backwards-compatible name used by early M5 scripts.
SCHEMA = DIAGNOSIS_SCHEMA


@dataclass(frozen=True)
class Frame:
    path: str
    camera: str
    t: float


@dataclass(frozen=True)
class FailureInput:
    rollout_id: str
    frames: tuple[Frame, ...]
    scene_features: Mapping[str, Any]


def _one_sentence(value: Any) -> bool:
    if not isinstance(value, str) or not (8 <= len(value.strip()) <= 240):
        return False
    text = value.strip()
    if "\n" in text or text.startswith(("-", "*", "1.")):
        return False
    return len(re.findall(r"[.!?](?:\s|$)", text)) <= 1


def validate_diagnosis(value: Any, allowed_frames: Sequence[str] | None = None) -> dict[str, Any]:
    """Validate one strict response, raising with all schema errors."""
    if not isinstance(value, Mapping):
        raise ValueError("diagnosis must be an object")
    errors: list[str] = []
    required = set(DIAGNOSIS_SCHEMA["required"])
    if set(value) - required: errors.append("unknown fields: " + ",".join(sorted(set(value) - required)))
    if required - set(value): errors.append("missing fields: " + ",".join(sorted(required - set(value))))
    if value.get("failure_stage") not in ALLOWED_STAGES: errors.append("invalid failure_stage")
    if not _one_sentence(value.get("cause")): errors.append("cause must be one plain sentence")
    if not isinstance(value.get("frame"), str) or not value.get("frame", "").strip(): errors.append("frame must be text")
    if allowed_frames is not None and value.get("frame") not in set(allowed_frames) and not re.fullmatch(r"<image [1-8]>", str(value.get("frame", ""))):
        errors.append("frame is not one of the supplied camera frames or image index")
    if not isinstance(value.get("scene_features"), Mapping): errors.append("scene_features must be an object")
    scenes = value.get("next_scenes")
    if not isinstance(scenes, list) or not (1 <= len(scenes) <= 32):
        errors.append("next_scenes must contain 1 to 32 specs")
    else:
        for i, scene in enumerate(scenes):
            if not isinstance(scene, Mapping): errors.append(f"next_scenes[{i}] must be an object"); continue
            miss, extra = set(SCENE_FIELDS) - set(scene), set(scene) - set(SCENE_FIELDS)
            if miss: errors.append(f"next_scenes[{i}] missing {sorted(miss)}")
            if extra: errors.append(f"next_scenes[{i}] unknown {sorted(extra)}")
            if scene.get("tier") not in {"loose", "messy", "pile"}: errors.append(f"next_scenes[{i}].tier invalid")
            n = scene.get("brick_count")
            if not isinstance(n, int) or isinstance(n, bool) or not 3 <= n <= 12: errors.append(f"next_scenes[{i}].brick_count invalid")
            for field, lo, hi in (("spacing", 0, 1), ("yaw_spread", 0, 3.142)):
                x = scene.get(field)
                if not isinstance(x, (int, float)) or isinstance(x, bool) or not lo <= float(x) <= hi: errors.append(f"next_scenes[{i}].{field} invalid")
            if not isinstance(scene.get("positions"), list) or not scene.get("positions"): errors.append(f"next_scenes[{i}].positions invalid")
            if not isinstance(scene.get("lighting"), str) or not scene.get("lighting", "").strip(): errors.append(f"next_scenes[{i}].lighting invalid")
    if errors: raise ValueError("; ".join(errors))
    return json.loads(json.dumps(dict(value)))


def select_frames(frames: Sequence[Frame | Mapping[str, Any]], failure_index: int | None = None, count: int = 8) -> list[Frame]:
    """Select 6–8 local frames and guarantee top plus inspection cameras."""
    converted = [f if isinstance(f, Frame) else Frame(str(f["path"]), str(f.get("camera", "top")), float(f.get("t", i))) for i, f in enumerate(frames)]
    count = max(6, min(8, int(count)))
    if len(converted) < 6: raise ValueError("at least six camera frames are required")
    i = len(converted) // 2 if failure_index is None else int(failure_index)
    window = converted[max(0, i - count // 2):max(0, i - count // 2) + count]
    tops = [f for f in window if f.camera.lower().startswith("top")]
    inspections = [f for f in window if f.camera.lower().startswith("inspection")]
    if not tops or not inspections:
        tops = [f for f in converted if f.camera.lower().startswith("top")]
        inspections = [f for f in converted if f.camera.lower().startswith("inspection")]
        if not tops or not inspections: raise ValueError("frames must include top and inspection cameras")
        window = [(tops if j % 2 == 0 else inspections)[min(len(tops if j % 2 == 0 else inspections) - 1, j // 2)] for j in range(count)]
    return window[:count]


def _image_payload(frame: Frame) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(Path(frame.path).read_bytes()).decode("ascii")}}


def build_request(inp: FailureInput, *, version: str = "v4-both-chain-fewshot") -> dict[str, Any]:
    """Build an OpenAI-compatible camera-only request."""
    frames = select_frames(inp.frames)
    text = ("You are a failure analyst for a vision-only brick sorting robot. Inspect the supplied top and inspection frames. "
            "Write a short observation chain internally, then return ONLY one JSON object. "
            "Use exactly these fields: failure_stage (one of the allowed stages), cause (one plain sentence), "
            "frame (one supplied frame path), scene_features (object), and next_scenes (one or more objects). "
            "Each next_scenes object must contain tier (loose, messy or pile), brick_count (integer 3-12), "
            "spacing (number 0-1), yaw_spread (number 0-3.142), positions (non-empty array), and lighting (string). "
            "failure_stage must be one of: approach, ik_fail, grasp, miss, slip, knocked_neighbour, inspection, "
            "wrong_bowl, inspector_false_accept, drop, unstable. "
            f"Analyst version: {version}. Do not use simulator state, event labels, or hidden object metadata.")
    messages: list[dict[str, Any]] = []
    if "fewshot" in version:
        # Two tiny format examples improve consistency without smuggling any
        # simulator labels into the actual camera request.
        examples = [
            {"failure_stage": "slip", "cause": "The gripper lost contact with the brick.", "frame": "<image 2>", "scene_features": {"visible": "brick below open jaws"}, "next_scenes": [{"tier": "messy", "brick_count": 6, "spacing": 0.01, "yaw_spread": 1.57, "positions": [[0.22, 0.0]], "lighting": "overhead"}]},
            {"failure_stage": "wrong_bowl", "cause": "The brick was released into the wrong bowl.", "frame": "<image 5>", "scene_features": {"visible": "brick above red bowl"}, "next_scenes": [{"tier": "pile", "brick_count": 8, "spacing": 0.005, "yaw_spread": 2.4, "positions": [[0.25, 0.02]], "lighting": "side light"}]},
        ]
        for example in examples:
            messages.extend([{"role": "user", "content": [{"type": "text", "text": "Example camera-only failure; return the strict schema."}]}, {"role": "assistant", "content": json.dumps(example, separators=(",", ":"))}])
    messages.append({"role": "user", "content": [{"type": "text", "text": text}] + [_image_payload(f) for f in frames]})
    return {"model": MODEL, "temperature": 0, "max_tokens": 6000,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "metadata": {"analyst_version": version, "rollout_id": inp.rollout_id,
                         "frames": [asdict(f) for f in frames], "schema": DIAGNOSIS_SCHEMA}}


def _content_text(raw: Mapping[str, Any]) -> str:
    message = raw.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list): content = "".join(p.get("text", "") for p in content if isinstance(p, Mapping))
    text = str(content or message.get("reasoning_content", "")).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()


class NemotronClient:
    """Rate-limited NVIDIA client with invalid-response retries."""
    def __init__(self, key: str | None = None, out_dir: str | Path = "/bulk2/runs/dryrun/m5/diagnoses", max_retries: int = 4, min_interval: float = 1.6):
        self.key = key or os.environ.get("NVIDIA_API_KEY")
        if not self.key: raise RuntimeError("NVIDIA_API_KEY is not injected; run through hsec")
        self.out_dir, self.max_retries, self.min_interval, self._last_call = Path(out_dir), max_retries, min_interval, 0.0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def analyse(self, inp: FailureInput, *, version: str = "v4-both-chain-fewshot") -> dict[str, Any]:
        body, rid = build_request(inp, version=version), inp.rollout_id or uuid.uuid4().hex
        folder = self.out_dir / rid; folder.mkdir(parents=True, exist_ok=True)
        (folder / "request.json").write_text(json.dumps(body, indent=2))
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            delay = self.min_interval - (time.monotonic() - self._last_call)
            if delay > 0: time.sleep(delay)
            try:
                req = Request(ENDPOINT, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
                with urlopen(req, timeout=120) as response: raw = json.loads(response.read())
                self._last_call = time.monotonic(); (folder / f"raw_response_{attempt}.json").write_text(json.dumps(raw, indent=2))
                selected = select_frames(inp.frames)
                diagnosis = validate_diagnosis(json.loads(_content_text(raw)), [f.path for f in selected])
                # Models sometimes refer to multimodal parts as ``<image 4>``;
                # persist the concrete path so the page can render the frame.
                match = re.fullmatch(r"<image ([1-8])>", diagnosis["frame"])
                if match:
                    index = int(match.group(1)) - 1
                    if index >= len(selected): raise ValueError("image index outside supplied frames")
                    diagnosis["frame"] = selected[index].path
                result = {"model": MODEL, "rollout_id": rid, "version": version, "attempt": attempt + 1, "diagnosis": diagnosis}
                (folder / "response.json").write_text(json.dumps(result, indent=2)); return result
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError) as exc:
                last = exc
                if attempt >= self.max_retries: break
                body = dict(body); body["messages"] = list(body["messages"]) + [{"role": "user", "content": [{"type": "text", "text": "Previous output was invalid. Return only one valid JSON object with every required field; no markdown."}]}]
                time.sleep(min(30.0, 2 ** attempt))
        (folder / "error.json").write_text(json.dumps({"error": repr(last)}, indent=2))
        raise RuntimeError(f"Nemotron failed after {self.max_retries + 1} attempts: {last}")


def call(frames: Path | Sequence[Path], seed: int, out: Path) -> dict[str, Any]:
    """Compatibility wrapper for the original M5 command.

    The old one-frame stub is intentionally rejected: M5 requires six to eight
    frames from both cameras.  Pass paths in alternating top/inspection order.
    """
    paths = [frames] if isinstance(frames, (str, Path)) else list(frames)
    if len(paths) < 6: raise ValueError("M5 requires 6-8 frames from top and inspection cameras")
    fs = tuple(Frame(str(path), "top" if i % 2 == 0 else "inspection", i / 30) for i, path in enumerate(paths[:8]))
    inp = FailureInput(f"seed-{seed}", fs, {"tier": "pile", "brick_count": 8, "spacing": .01, "yaw_spread": 3.14, "lighting": "unknown"})
    return NemotronClient(out_dir=out).analyse(inp)


def _canonical_cause(event: Mapping[str, Any]) -> str:
    reason = str(event.get("reason") or event.get("failed_stage") or event.get("failure_reason") or "unknown").lower()
    if "ik" in reason: return "inverse kinematics did not reach the target"
    if "slip" in reason or "contact" in reason or "grasp" in reason: return "the gripper lost contact with the brick"
    if "neighbour" in reason or "neighbor" in reason or "knock" in reason: return "the arm contacted a neighbouring brick"
    if "bowl" in reason or "wrong" in reason: return "the brick was released into the wrong bowl"
    if "accept" in reason or "inspect" in reason: return "the inspector accepted a defective brick"
    if "miss" in reason or "drop" in reason: return "the gripper missed or dropped the brick"
    return reason.replace("_", " ")


def event_label(event: Mapping[str, Any]) -> dict[str, str]:
    """Map an event-log failure to the canonical scored label."""
    reason = str(event.get("reason") or event.get("failed_stage") or event.get("failure_reason") or "miss").lower()
    if "ik" in reason: stage = "ik_fail"
    elif "slip" in reason or "contact" in reason or "grasp" in reason: stage = "slip"
    elif "neighbour" in reason or "neighbor" in reason or "knock" in reason: stage = "knocked_neighbour"
    elif "bowl" in reason or "wrong" in reason: stage = "wrong_bowl"
    elif "accept" in reason or "inspect" in reason: stage = "inspector_false_accept"
    elif "drop" in reason: stage = "drop"
    else: stage = "miss"
    return {"failure_stage": stage, "cause": _canonical_cause(event)}


def labelled_set(event_log: Sequence[Mapping[str, Any]], minimum: int = 300) -> list[dict[str, Any]]:
    """Build scored labels from real failure events, refusing a small sample."""
    failures = [e for e in event_log if e.get("reason") or e.get("failed_stage") or e.get("failure_reason")]
    if len(failures) < minimum:
        raise ValueError(f"event log contains {len(failures)} failures; at least {minimum} are required")
    return [{"id": str(e.get("id", i)), **event_label(e)} for i, e in enumerate(failures)]


def score_labelled_file(labels_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    """Score JSON/JSONL labels and predictions for reproducible status files."""
    def read(path: str | Path) -> list[Mapping[str, Any]]:
        p = Path(path); raw = json.loads(p.read_text()) if p.suffix != ".jsonl" else [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        return list(raw if isinstance(raw, list) else raw.get("rows", raw.get("labels", [])))
    return score_diagnoses(read(labels_path), read(predictions_path))


def score_diagnoses(labels: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score exact stage and cause accuracy against event-log labels."""
    n = min(len(labels), len(predictions))
    if n == 0: return {"n": 0, "stage_accuracy": 0.0, "cause_accuracy": 0.0, "accuracy": 0.0}
    stage = cause = 0
    for label, pred in zip(labels[:n], predictions[:n]):
        p = pred.get("diagnosis", pred); stage += int(p.get("failure_stage") == label.get("failure_stage")); cause += int(p.get("cause") == label.get("cause"))
    return {"n": n, "stage_accuracy": round(stage / n, 6), "cause_accuracy": round(cause / n, 6), "accuracy": round((stage + cause) / (2 * n), 6)}


def make_version_table(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    changes = {"v1-one-frame": "one top-camera frame", "v2-both-cameras": "six frames from top and inspection cameras", "v3-observation-chain": "both cameras plus an observation chain", "v4-both-chain-fewshot": "both cameras, observation chain, and few-shot examples"}
    return [{"version": v, "change": changes.get(v, "configuration change"), "accuracy": s.get("accuracy", 0.0), "stage_accuracy": s.get("stage_accuracy", 0.0), "cause_accuracy": s.get("cause_accuracy", 0.0), "n": s.get("n", 0)} for v, s in results.items()]


def compare_curriculum(suggested: Sequence[Mapping[str, Any]], random_batch: Sequence[Mapping[str, Any]], score_fn: Callable[[Sequence[Mapping[str, Any]], str], Mapping[str, Any]], checkpoint: str, output: str | Path | None = None) -> dict[str, Any]:
    """Compare equal-size batches from the same checkpoint."""
    if not suggested or len(suggested) != len(random_batch): raise ValueError("batches must be non-empty and equal-sized")
    result = {"checkpoint": checkpoint, "batch_size": len(suggested), "suggested": dict(score_fn(suggested, checkpoint)), "random": dict(score_fn(random_batch, checkpoint))}
    result["delta"] = round(float(result["suggested"].get("score", 0)) - float(result["random"].get("score", 0)), 6); result["helps"] = result["delta"] > 0
    if output:
        p = Path(output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(result, indent=2))
    return result


def diagnosis_record(result: Mapping[str, Any], inp: FailureInput) -> dict[str, Any]:
    """Shape a row for telemetry.record_diagnosis and the page API."""
    d = result.get("diagnosis", result)
    return {"rollout_id": inp.rollout_id, "frame_paths": [f.path for f in select_frames(inp.frames)], "failure_stage": d["failure_stage"], "cause": d["cause"], "frame": d["frame"], "scene_features": dict(inp.scene_features), "next_scenes": d["next_scenes"], "model": result.get("model", MODEL), "version": result.get("version", "unknown")}


def store_diagnosis(result: Mapping[str, Any], inp: FailureInput, writer: Callable[[Mapping[str, Any]], Any] | None = None) -> Any:
    """Persist one diagnosis through the page session's telemetry boundary."""
    row = diagnosis_record(result, inp)
    if writer is None:
        from . import telemetry
        writer = getattr(telemetry, "record_diagnosis", None)
    if writer is None:
        raise RuntimeError("telemetry.record_diagnosis is not wired")
    return writer(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--frame", type=Path, action="append", required=True); p.add_argument("--seed", type=int, default=7); p.add_argument("--rollout-id", default=None); p.add_argument("--out", type=Path, default=Path("/bulk2/runs/dryrun/m5/diagnoses")); p.add_argument("--scene-features", default='{"tier":"pile","brick_count":8,"spacing":0.01,"yaw_spread":3.14,"lighting":"key light"}')
    a = p.parse_args(); frames = tuple(Frame(str(x), "top" if i % 2 == 0 else "inspection", i / 30) for i, x in enumerate(a.frame)); inp = FailureInput(a.rollout_id or f"seed-{a.seed}", frames, json.loads(a.scene_features)); print(json.dumps(NemotronClient(out_dir=a.out).analyse(inp), indent=2))


if __name__ == "__main__": main()
