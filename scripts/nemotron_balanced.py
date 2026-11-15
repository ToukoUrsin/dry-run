"""Strict, cached four-stage Nemotron evaluation for prompt 055."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
STAGES = ("never_touched_brick", "touched_but_did_not_lift", "lifted_then_lost", "wrong_bowl")
PANELS = ("before_grasp", "end_closing", "presentation", "end_cycle")


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _valid_sentence(x: Any) -> bool:
    return isinstance(x, str) and 8 <= len(x.strip()) <= 400 and "\n" not in x


def validate(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("response is not an object")
    required = {"failure_stage", "visible_support", "scene_change", "evidence_panel"}
    missing, extra = required - set(value), set(value) - required
    errors = []
    if missing:
        errors.append("missing fields: " + ",".join(sorted(missing)))
    if extra:
        errors.append("unknown fields: " + ",".join(sorted(extra)))
    if value.get("failure_stage") not in STAGES:
        errors.append("invalid failure_stage")
    if not _valid_sentence(value.get("visible_support")):
        errors.append("visible_support must be one line of 8-400 chars")
    if not _valid_sentence(value.get("scene_change")):
        errors.append("scene_change must be one line of 8-400 chars")
    if value.get("evidence_panel") not in PANELS:
        errors.append("invalid evidence_panel")
    if errors:
        raise ValueError("; ".join(errors))
    return {k: str(value[k]).strip() for k in required}


def _image(path: str) -> dict[str, Any]:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def build_request(row: Mapping[str, Any]) -> dict[str, Any]:
    instruction = (
        "Judge one brick-sorting failure from the supplied 2x2 sheet of four "
        "captioned 448 px panels. Use only visible evidence. Return exactly one "
        "JSON object with four fields: failure_stage, visible_support, scene_change, "
        "and evidence_panel. failure_stage must be exactly one of: "
        "never_touched_brick, touched_but_did_not_lift, lifted_then_lost, wrong_bowl. "
        "visible_support must state concrete visual evidence; scene_change must give "
        "one concrete training change; evidence_panel must be one of before_grasp, "
        "end_closing, presentation, end_cycle. Do not mention hidden labels, source "
        "metadata, simulator events, or reasoning. No markdown and no extra keys."
    )
    return {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 800,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": [{"type": "text", "text": instruction}, _image(str(row["sheet"]))]}],
        "response_format": {"type": "json_object"},
        "metadata": {"analyst_version": "prompt055-balanced-448-sheet", "rollout_id": str(row["id"])},
    }


def _content(raw: Mapping[str, Any]) -> str:
    msg = raw.get("choices", [{}])[0].get("message", {})
    c = msg.get("content", "")
    if isinstance(c, list):
        c = "".join(str(part.get("text", "")) for part in c if isinstance(part, Mapping))
    c = str(c or msg.get("reasoning_content", "")).strip()
    c = re.sub(r"^```(?:json)?\s*|\s*```$", "", c, flags=re.I).strip()
    try:
        value = json.loads(c)
        if isinstance(value, Mapping) and "failure_stage" in value:
            return json.dumps(value, separators=(",", ":"))
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", c):
        try:
            value, _ = decoder.raw_decode(c[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "failure_stage" in value:
            return json.dumps(value, separators=(",", ":"))
    return c


class Client:
    def __init__(self, out: Path, interval: float = 1.5, retries: int = 2):
        self.key = os.environ.get("NVIDIA_API_KEY")
        if not self.key:
            raise RuntimeError("NVIDIA_API_KEY is not injected; use hsec exec")
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.interval = float(interval)
        self.retries = int(retries)
        self.last = 0.0

    def analyse(self, row: Mapping[str, Any]) -> dict[str, Any]:
        rid = str(row["id"])
        folder = self.out / rid
        folder.mkdir(parents=True, exist_ok=True)
        body = build_request(row)
        (folder / "request.json").write_text(json.dumps(body, indent=2))
        last_error: str | None = None
        for attempt in range(self.retries + 1):
            delay = self.interval - (time.monotonic() - self.last)
            if delay > 0:
                time.sleep(delay)
            try:
                request = Request(ENDPOINT, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
                with urlopen(request, timeout=120) as response:
                    payload = response.read()
                self.last = time.monotonic()
                raw = json.loads(payload)
                (folder / f"raw_response_{attempt}.json").write_text(json.dumps(raw, indent=2))
                diagnosis = validate(json.loads(_content(raw)))
                result = {"id": rid, "model": MODEL, "attempt": attempt + 1, "diagnosis": diagnosis}
                (folder / "response.json").write_text(json.dumps(result, indent=2))
                return result
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError) as exc:
                self.last = time.monotonic()
                last_error = repr(exc)
                if attempt < self.retries:
                    # NVIDIA's shared endpoint can return a rolling-window
                    # 429 even when this client is paced.  Give that window
                    # time to clear before spending the next cached attempt.
                    wait = 20.0 if isinstance(exc, HTTPError) and exc.code == 429 else 2**attempt
                    time.sleep(min(45.0, wait))
        result = {"id": rid, "model": MODEL, "error": last_error}
        (folder / "error.json").write_text(json.dumps(result, indent=2))
        return result


def score(rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = {str(r["id"]): str(r["label"]) for r in rows}
    pred = {str(r["id"]): r.get("diagnosis", {}).get("failure_stage") for r in result_rows}
    confusion = {a: {b: 0 for b in STAGES} for a in STAGES}
    per = {a: {"n": 0, "correct": 0, "accuracy": 0.0, "predicted": {b: 0 for b in STAGES}} for a in STAGES}
    valid = correct = 0
    for rid, actual in truth.items():
        guess = pred.get(rid)
        if actual not in STAGES:
            continue
        per[actual]["n"] += 1
        if guess in STAGES:
            valid += 1
            confusion[actual][guess] += 1
            per[actual]["predicted"][guess] += 1
            if guess == actual:
                correct += 1
                per[actual]["correct"] += 1
    for stage in STAGES:
        per[stage]["accuracy"] = round(per[stage]["correct"] / per[stage]["n"], 6) if per[stage]["n"] else None
    counts = {s: sum(1 for r in rows if r["label"] == s) for s in STAGES}
    largest = max(counts, key=lambda s: (counts[s], -STAGES.index(s)))
    acc = correct / len(rows) if rows else 0.0
    baseline = counts[largest] / len(rows) if rows else 0.0
    sentence = (
        f"Nemotron got {correct}/{len(rows)} ({acc:.1%}) on {len(STAGES)} balanced simulator-labeled failure stages; "
        f"the largest-class baseline is {counts[largest]}/{len(rows)} ({baseline:.1%}) and chance is 1/{len(STAGES)} (25.0%)."
    )
    return {
        "schema": "dryrun.nemotron_balanced.score.v1", "commit": _commit(), "model": MODEL,
        "n": len(rows), "valid_n": valid, "correct": correct, "accuracy": round(acc, 6),
        "counts": counts, "baseline": {"kind": "largest_class", "label": largest, "correct": counts[largest], "accuracy": round(baseline, 6)},
        "chance_baseline": {"kind": "uniform", "correct_expected": len(rows) / len(STAGES), "accuracy": round(1 / len(STAGES), 6)},
        "sentence": sentence, "per_stage": per, "confusion": confusion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    client = Client(args.out, interval=args.interval, retries=args.retries)
    results_path = args.out / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if "diagnosis" in value:
                    done[str(value["id"])] = value
    with results_path.open("a") as fh:
        for row in rows:
            rid = str(row["id"])
            if rid in done:
                continue
            value = client.analyse(row)
            fh.write(json.dumps(value, separators=(",", ":")) + "\n")
            fh.flush()
            print(json.dumps({"id": rid, "ok": "diagnosis" in value}), flush=True)
    result_rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    score_value = score(rows, result_rows)
    (args.out / "score.json").write_text(json.dumps(score_value, indent=2) + "\n")
    print(json.dumps(score_value, indent=2))


if __name__ == "__main__":
    main()
