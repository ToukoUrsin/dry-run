#!/usr/bin/env python3
"""Refresh the evidence payload for the Codex night picture.

The command reads only repository history and prompt/status files.  It keeps
the hand-written Codex section intact and replaces its generated ``night``
object, so the page can be refreshed before a deploy without hand-counting
commits or prompts.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "page" / "codex.json"
START = datetime(2026, 11, 14, 18, 0, tzinfo=timezone.utc)
LINES_COMMAND = "git ls-files '*.py' '*.js' '*.html' '*.css' | xargs wc -l"
BOUNDARY = "Everything outside `prompts/` was written by Codex. The supervisor wrote the prompts, and two reference sorters under `prompts/021-reference` and `prompts/027-reference`, which Codex ported into `dryrun/expert.py`."
AUTHORSHIP = "The supervisor (Claude, driving the Codex CLI on Touko's behalf) wrote the prompts and two reference sorters, `prompts/021-reference` and `prompts/027-reference`. Codex ported them into `dryrun/expert.py` and wrote everything else."
REVIEW = {
    "attempts": 5,
    "historical_blocked_attempts": 4,
    "findings": 28,
    "fixed": 10,
    "fixed_commits": [
        "2ccc9a6", "31e2ce0", "5d7c557", "724cb34", "86bcfc7",
        "4b6252d", "205bca6", "eb81e98", "c846434", "6471dbe",
    ],
    "status": "completed_with_findings_and_fixes",
    "source_file": "status/codex-review-0415.md",
    "runs": [
        {"name": "fresh review", "session_id": "01a1dd6a-74f3-7be2-b5de-008b0a2789ec", "findings": 0},
        {"name": "benchmark/seeds", "session_id": "01a1dd71-53f5-7721-943f-b47972b1ebaf", "findings": 4},
        {"name": "camera-only sorter", "session_id": "01a1dd75-2738-77b1-9ace-7dd62881a43d", "findings": 10},
        {"name": "learned-policy loop", "session_id": "01a1dd75-1f15-7d70-a814-44da7b471dbf", "findings": 6},
        {"name": "Tiger/web", "session_id": "01a1dd74-09d4-7913-8260-ad233e0760bf", "findings": 8},
    ],
    "statement": "codex review found why the learned policy scored zero: every action chunk was one action repeated, paired with the next frame. After the fix it reaches 5 of 20 one-brick scenes and still sorts 0.",
}
SESSION_IDS = {
    "main": "01a1db37-cd82-7771-9d68-113fc0bb7cfc",
    "page": "01a1db6e-0b22-7971-ae26-5dd52eb3ed3d",
    "inspector": "01a1db6e-0b7e-7103-9c3c-2fc35517f427",
    "analyst": "01a1db85-8680-71e1-8c2c-cea5752bafa1",
    "policy": "01a1db85-87b9-7760-b934-03b82b6ce199",
}
SUBAGENTS = {
    "count": 3,
    "parent": "main",
    "paths": ["/root/astra_geometry", "/root/curriculum_astra", "/root/docs_status"],
    "note": "These are three subagents of the main session; all commits carry one git author and lanes are assigned by touched files, not by session.",
}

SESSION_NAMES = ("main", "inspector", "page", "policy", "analyst")
SESSION_DATA = [
    {"name": "main", "model": "gpt-6-astra", "effort": "ultra", "owns": "simulator, expert, benchmark, replay"},
    {"name": "inspector", "model": "gpt-6.1-sol", "effort": "ultra", "owns": "defects and anomaly inspection"},
    {"name": "page", "model": "gpt-6.1-sol", "effort": "ultra", "owns": "viewer, telemetry, deployment"},
    {"name": "policy", "model": "gpt-6.1-sol", "effort": "ultra", "owns": "demos, ACT, policy evaluation"},
    {"name": "analyst", "model": "gpt-6.1-sol", "effort": "ultra", "owns": "Nemotron, stations, evidence docs", "external_model": "NVIDIA Nemotron"},
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def commit_files(sha: str) -> list[str]:
    value = subprocess.check_output(
        ["git", "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", sha],
        cwd=ROOT,
        text=True,
    )
    return [line for line in value.splitlines() if line]


def commit_session(subject: str, files: Iterable[str]) -> str:
    """Attribute one commit using the Prompt 038 path rules."""
    if "film plan:" in subject.lower() or subject.lower().startswith('revert "film plan'):
        return "other"
    scores = Counter()
    for raw in files:
        path = raw.lower()
        if path.startswith("web/"):
            scores["page"] += 1
        elif path == "data/page/inspector.json" or "inspector" in path:
            scores["inspector"] += 1
        elif path.startswith("modal/") or any(token in path for token in ("policy", "demos", "train", "dagger")) or path == "status/m4.md":
            scores["policy"] += 1
        elif (
            path.startswith("docs/")
            or any(token in path for token in ("analyst", "stations", "nemotron", "curriculum", "worker"))
            or path in {"codex.md", "readme.md", "submission.md", "film_plan.md", "morning.md"}
            or path.startswith("status/m5")
            or path.startswith("scripts/night_data.py")
        ):
            scores["analyst"] += 1
        else:
            scores["main"] += 1
    if not scores:
        return "main"
    order = {name: index for index, name in enumerate(("page", "inspector", "policy", "analyst", "main"))}
    return max(scores, key=lambda name: (scores[name], -order[name]))


def commit_rows() -> list[dict[str, str]]:
    rows = []
    for sha in git("log", "--reverse", "--since", START.isoformat(), "--format=%H").splitlines():
        stamp, subject = git("show", "-s", "--format=%cI%x09%s", sha).split("\t", 1)
        timestamp = datetime.fromisoformat(stamp).astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        rows.append({"t": timestamp, "sha": sha[:7], "subject": subject, "session": commit_session(subject, commit_files(sha))})
    return rows


def session_commit_stats(commits: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    """Return the lane counts and complete commit lists from git attribution."""
    grouped: dict[str, list[str]] = {name: [] for name in SESSION_NAMES}
    for row in commits:
        lane = row["session"]
        if lane in grouped:
            grouped[lane].append(row["sha"])
    return {
        name: {
            "commit_count": len(shas),
            "commits": shas,
            "commit_count_basis": "git log --name-only path attribution at this snapshot; film-plan commits and reverts are other",
        }
        for name, shas in grouped.items()
    }


def prompt_time(first_line: str) -> str | None:
    match = re.search(r"(\d{1,2}\.\d{1,2}\.\d{4})\s+(\d{2}:\d{2})\s*UTC", first_line)
    if not match:
        return None
    value = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%d.%m.%Y %H:%M").replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def prompt_title(first_line: str) -> str:
    if first_line.lower().startswith("# goal for codex"):
        return "Dry Run goal"
    match = re.search(r"UTC:\s*(.*)$", first_line)
    source = match.group(1) if match else first_line.lstrip("# ")
    source = source.replace("_", " ").replace("`", "")
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", source)
    return " ".join(words[:8]) or "Supervisor prompt"


def prompt_to(first_line: str) -> str:
    low = first_line.lower()
    recipient = ""
    match = re.search(r"supervisor to\s+(.+?)(?:,|:)", low)
    if match:
        recipient = match.group(1)
    if "every" in recipient or "all five" in recipient or "all sessions" in recipient:
        return "all"
    for name in ("inspector", "page", "policy", "analyst"):
        if name in recipient:
            return name
    return "main"


def prompt_rows() -> list[dict[str, str | None]]:
    rows = []
    for path in sorted((ROOT / "prompts").glob("*.md")):
        lines = path.read_text(errors="replace").splitlines()
        first = lines[0] if lines else ""
        rows.append({"t": prompt_time(first), "file": str(path.relative_to(ROOT)), "title": prompt_title(first), "to": prompt_to(first)})
    return rows


def line_count() -> int:
    paths = git("ls-files", "*.py", "*.js", "*.html", "*.css").splitlines()
    return sum(path.read_text(errors="replace").count("\n") for path in (ROOT / name for name in paths))


def rejections() -> list[dict[str, str]]:
    candidates = [
        ("2026-11-14T18:05:00Z", "first pass rejected", "prompts/002-first-pass-rejected.md", "The 245-line first pass had no real simulator or evidence."),
        ("2026-11-14T18:32:00Z", "zero pickups found", "prompts/004-review-of-seed7.md", "The review calls zero pickups the blocker."),
        ("2026-11-14T19:47:00Z", "inspector rejected good bricks", "prompts/014-inspector-rejected.md", "The first real-frame attempt rejected good bricks."),
        ("2026-11-14T19:58:00Z", "rigged fixed-scene tier", "prompts/016-loose-tier-rigged.md", "The one-brick fixed spot was withdrawn as a benchmark."),
        ("2026-11-14T22:10:00Z", "zero of fifty after four hours", "prompts/019-grasp-recipe-measured.md", "The measured recipe prompt records zero of fifty."),
        ("2026-11-15T01:05:00Z", "policy reached zero stages", "prompts/030-policy-train-for-real.md", "The 201-step run reached zero at every stage."),
        ("2026-11-15T01:41:00Z", "live page broken", "prompts/034-live-page-broken.md", "The live page had a console error and lost its first screen."),
    ]
    return [{"t": t, "label": label, "prompt": prompt, "evidence": evidence} for t, label, prompt, evidence in candidates if (ROOT / prompt).is_file()]


def scores() -> list[dict[str, object]]:
    # The one-brick loose tier was withdrawn as rigged in prompt 016.  Keep
    # those historical scores in status/M1.md, but omit them from the page's
    # score series so they cannot be read as benchmark evidence.
    return [
        {"t": "2026-11-14T23:40:39Z", "tier": "touching", "sorted": 217, "bricks": 494, "commit": "2de1ae7", "kind": "held-out", "source": "status/M1.md"},
        {"t": "2026-11-14T23:40:39Z", "tier": "pile", "sorted": 31, "bricks": 386, "commit": "2de1ae7", "kind": "held-out", "source": "status/M1.md"},
        {"t": "2026-11-15T00:53:31Z", "tier": "touching", "sorted": 451, "bricks": 494, "commit": "b9d897c", "kind": "held-out", "source": "data/page/tiers.json"},
        {"t": "2026-11-15T00:53:31Z", "tier": "pile", "sorted": 259, "bricks": 386, "commit": "b9d897c", "kind": "held-out", "source": "data/page/tiers.json"},
    ]


def build(input_path: Path) -> dict:
    payload = json.loads(input_path.read_text())
    now = datetime.now(timezone.utc).replace(microsecond=0)
    commits = commit_rows()
    commit_stats = session_commit_stats(commits)
    prompts = prompt_rows()
    elapsed_hours = round((now - START).total_seconds() / 3600.0, 2)
    payload["updated_utc"] = now.isoformat().replace("+00:00", "Z")
    payload["commit"] = git("rev-parse", "--short", "HEAD")
    payload["commits_total"] = len(commits)
    payload["lines_of_code"] = line_count()
    payload["commits_count_command"] = "git log --reverse --since 2026-11-14T18:00:00Z"
    payload["lines_count_command"] = LINES_COMMAND
    payload["prompts_count"] = len(prompts)
    payload["supervisor_prompts"] = len(prompts)
    payload["prompts_note"] = "Count is generated from prompts/*.md by scripts/night_data.py at each refresh."
    payload["prompts"] = prompts
    payload["hours"]["wall_clock_night"] = elapsed_hours
    payload["hours"]["summed_session_hours"] = payload["hours"].get("total")
    payload["hours"]["basis"] = "wall_clock_night is elapsed UTC duration from the run start; summed_session_hours is the sum of the five recorded tmux session lifetimes."
    payload["boundary"] = BOUNDARY
    payload["authorship"] = AUTHORSHIP
    payload["review"] = dict(REVIEW)
    payload["fast_mode"] = {
        "used_on_sessions": list(SESSION_IDS),
        "measured": True,
        "normal_wall_seconds": 28.53,
        "fast_wall_seconds": 22.42,
        "delta_seconds": 6.11,
        "delta_percent": 21.4,
        "result_identical": True,
        "statement": "Used on all five sessions; the one-run comparison was 28.53 s without the Fast tier and 22.42 s with it, 6.11 s sooner (21.4%), with identical output.",
        "source": "status/fast-mode.md",
    }
    payload["session_ids"] = dict(SESSION_IDS)
    payload["subagents"] = dict(SUBAGENTS)
    payload["subagents_spawned"] = SUBAGENTS["count"]
    payload["subagents_note"] = SUBAGENTS["note"]
    for feature in payload.get("features", []):
        if feature.get("name") == "Fast mode":
            feature.update({
                "evidence": payload["fast_mode"]["statement"],
                "link": "status/fast-mode.md",
                "timing_measured": True,
            })
        elif feature.get("name") == "codex review":
            feature.update({
                "evidence": payload["review"]["statement"],
                "link": "status/codex-review-0415.md",
                "status": payload["review"]["status"],
            })
    for session in payload.get("sessions", []):
        stats = commit_stats.get(session.get("name"))
        if stats:
            session.update(stats)
        name = session.get("name")
        if name in SESSION_IDS:
            session["session_id"] = SESSION_IDS[name]
            session["model"] = next(item["model"] for item in SESSION_DATA if item["name"] == name)
    generated_night_sessions = []
    for session in SESSION_DATA:
        generated = dict(session)
        generated.update(commit_stats[session["name"]])
        generated_night_sessions.append(generated)
    payload["night"] = {
        "start_utc": START.isoformat().replace("+00:00", "Z"),
        "end_utc": now.isoformat().replace("+00:00", "Z"),
        "sessions": generated_night_sessions,
        "commits": commits,
        "prompts": prompts,
        "rejections": rejections(),
        "scores": scores(),
        "scores_note": "The fixed-scene loose tier was withdrawn as rigged; its historical scores are omitted from this benchmark series.",
        "totals": {"commits": len(commits), "prompts": len(prompts), "sessions": len(SESSION_NAMES), "lines_of_code": payload["lines_of_code"], "hours": elapsed_hours},
        "attribution": "Each commit is assigned from git diff-tree --name-only: web to page; inspector paths and data/page/inspector.json to inspector; modal and policy paths to policy; analyst, stations, Nemotron and docs to analyst; the rest to main. The only non-Codex commits are film-plan 44555a4 and 3a8ad76 plus their reverts 606c23e and 9e4bc77; all other commits carry one git author and are lane-assigned by touched files. Counts are derived at every refresh.",
        "git_log_command": "git log --reverse --since 2026-11-14T18:00:00Z",
        "boundary": BOUNDARY,
        "authorship": AUTHORSHIP,
        "review": dict(REVIEW),
        "fast_mode": payload["fast_mode"],
        "session_ids": dict(SESSION_IDS),
        "subagents": dict(SUBAGENTS),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    payload = build(args.input)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "commits": payload["night"]["totals"]["commits"], "prompts": payload["night"]["totals"]["prompts"], "lines_of_code": payload["night"]["totals"]["lines_of_code"], "hours": payload["night"]["totals"]["hours"]}, indent=2))


if __name__ == "__main__":
    main()
