#!/usr/bin/env python3
"""Small dependency-light server for the Dry Run evidence page."""
from __future__ import annotations
import json, os, re, subprocess, time, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
RUNS = Path("/bulk3/runs/dryrun")
REPLAY_DIR = Path(os.environ.get("DRYRUN_REPLAY_DIR", str(RUNS / "goal-page-audit-0312")))
TIER_REPLAY_LOCAL_DIRS = {
    "loose": RUNS / "goal-page-audit-0312/seed-7",
    "messy": RUNS / "m1/replay-touching-seed1001-b9d897c",
    "pile": RUNS / "m1/replay-pile-seed1045-b9d897c",
}
BENCHMARK_TIERS = Path(os.environ.get("DRYRUN_BENCHMARK_TIERS", str(RUNS / "benchmark/tier-summary-64c.json")))
# The page's scene is exported from the compiled MuJoCo station model.  Keep a
# small compatibility list for older deployments while allowing Fly to mount
# the exact release artifact by secret.
VIEWER_SCENE = Path(os.environ.get("DRYRUN_VIEWER_SCENE", str(RUNS / "page-assets/so101-mujoco.glb")))
VIEWER_SCENE_CANDIDATES = (
    VIEWER_SCENE,
    RUNS / "page-assets/so101-mujoco.glb",
    RUNS / "page-assets/so101.glb",
)
INSPECTOR_EVALUATION = Path(os.environ.get("DRYRUN_INSPECTOR_EVALUATION", str(RUNS / "_b2/m2/evaluation.json")))
# Prefer the current rebuild's compact page projection when it exists and is
# populated.  The archived `_b2` artifact is retained only as a compatibility
# fallback; it must never hide a newer measured result on the page.
INSPECTOR_PAGE_CANDIDATES = (
    Path("/bulk3/runs/dryrun/m2_rebuild/final_model/inspector_page.json"),
    INSPECTOR_EVALUATION,
    ROOT / "data" / "inspector_evaluation.json",
)
CODEX_LOG = ROOT / "data" / "codex_log.jsonl"
CODEX_CAST = RUNS / "_b2/codex-casts/main-001.cast"
PROMPT024_EXAMPLES = (
    RUNS / "nemotron_prompt024/examples.jsonl",
    ROOT / "data" / "nemotron_prompt024_examples.jsonl",
)
SNAPSHOT_ARTIFACT = ROOT / "data" / "snapshot.json"
REACH_MAP_ARTIFACT = ROOT / "data" / "reach_map_polar.json"
EVIDENCE_DIR = Path(os.environ.get("DRYRUN_EVIDENCE_DIR", str(RUNS / "page-assets/failure-previews")))
PAGE_DATA_DIR = ROOT / "data" / "page"
STATIC_DIR = WEB / "static"
DOCUMENT_FILES = {
    "codex": ROOT / "CODEX.md",
    "submission": ROOT / "SUBMISSION.md",
    "morning": ROOT / "MORNING.md",
}
STATUS_FILES = {
    "codex-review-0415.md": ROOT / "status" / "codex-review-0415.md",
}

def _html_escape(value: str) -> str:
    import html
    return html.escape(str(value), quote=True)

def _markdown_inline(value: str) -> str:
    """Render the small, safe Markdown subset used by the evidence docs."""
    import html
    escaped = html.escape(value, quote=True)
    code_tokens: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_tokens.append(f"<code>{match.group(1)}</code>")
        return f"\x00CODE{len(code_tokens) - 1}\x00"

    escaped = re.sub(r"`([^`]+)`", stash_code, escaped)

    def link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2).strip()
        # Keep document links useful without allowing javascript/data URLs or
        # exposing arbitrary local filesystem paths through href attributes.
        if target.startswith(("https://", "http://", "/", "prompts/", "status/")):
            href = target
            if target.startswith("prompts/"):
                href = "/" + target
            elif target.startswith("status/"):
                href = "/" + target
            return f'<a href="{html.escape(href, quote=True)}">{label}</a>'
        return label

    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"(?<![\w_])_([^_]+)_(?![\w_])", r"<em>\1</em>", escaped)
    for index, token in enumerate(code_tokens):
        escaped = escaped.replace(f"\x00CODE{index}\x00", token)
    return escaped

def _markdown_to_html(markdown: str) -> str:
    """Render headings, tables, links, lists and fenced code without a dependency."""
    lines = markdown.replace("\r\n", "\n").split("\n")
    output: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    table_rows: list[list[str]] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(part.strip() for part in paragraph).strip()
            if text:
                output.append(f"<p>{_markdown_inline(text)}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            output.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items.clear()

    def flush_table() -> None:
        if not table_rows:
            return
        header, *rows = table_rows
        html_rows = ["<thead><tr>" + "".join(f"<th>{_markdown_inline(cell.strip())}</th>" for cell in header) + "</tr></thead>"]
        if rows:
            html_rows.append("<tbody>" + "".join("<tr>" + "".join(f"<td>{_markdown_inline(cell.strip())}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody>")
        output.append("<div class=\"doc-table-wrap\"><table>" + "".join(html_rows) + "</table></div>")
        table_rows.clear()

    def is_table_separator(text: str) -> bool:
        cells = [cell.strip() for cell in text.strip().strip("|").split("|")]
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph(); flush_list(); flush_table()
            if in_code:
                import html
                output.append("<pre><code>" + html.escape("\n".join(code_lines), quote=False) + "</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph(); flush_list(); flush_table()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        if heading:
            flush_paragraph(); flush_list(); flush_table()
            level = len(heading.group(1))
            output.append(f"<h{level}>{_markdown_inline(heading.group(2))}</h{level}>")
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if bullet:
            flush_paragraph(); flush_table()
            list_items.append(_markdown_inline(bullet.group(1)))
            continue
        if line.lstrip().startswith("|") and "|" in line.strip()[1:]:
            flush_paragraph(); flush_list()
            if is_table_separator(line):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            continue
        if table_rows:
            flush_table()
        paragraph.append(line)
    if in_code:
        import html
        output.append("<pre><code>" + html.escape("\n".join(code_lines), quote=False) + "</code></pre>")
    flush_paragraph(); flush_list(); flush_table()
    return "".join(output)

def document_page(slug: str, prompt_name: str | None = None) -> bytes | None:
    """Render a small readable evidence document without a markdown dependency."""
    if slug == "prompts":
        prompt_dir = ROOT / "prompts"
        links = []
        if prompt_dir.is_dir():
            for prompt in sorted(prompt_dir.glob("*.md")):
                name = prompt.name
                links.append(f'<li><a href="/prompts/{_html_escape(name)}">{_html_escape(name)}</a></li>')
        title, body = "Supervisor prompts", "<ul>" + "".join(links) + "</ul>"
    elif slug == "prompt":
        if not prompt_name or not re.fullmatch(r"[A-Za-z0-9_-]+\.md", prompt_name):
            return None
        prompt = (ROOT / "prompts" / prompt_name).resolve()
        if prompt.parent != (ROOT / "prompts").resolve() or not prompt.is_file():
            return None
        title, body = prompt.name, f"<pre>{_html_escape(prompt.read_text())}</pre>"
    elif slug == "status":
        if not prompt_name or prompt_name not in STATUS_FILES:
            return None
        status = STATUS_FILES[prompt_name]
        if not status.is_file():
            return None
        title, body = status.name, f"<pre>{_html_escape(status.read_text())}</pre>"
    else:
        path = DOCUMENT_FILES.get(slug)
        if path is None or not path.is_file():
            return None
        title = path.name
        body = _markdown_to_html(path.read_text()) if slug == "codex" else f"<pre>{_html_escape(path.read_text())}</pre>"
    nav = '<nav><a href="/">Page</a> · <a href="/codex">Codex log</a> · <a href="/prompts">Prompts</a> · <a href="/submission">Submission</a></nav>'
    page_heading = "" if slug == "codex" else f"<h1>{_html_escape(title)}</h1>"
    return ("<!doctype html><meta charset=\"utf-8\"><title>" + _html_escape(title) +
            " · Dry Run</title><style>body{max-width:960px;margin:40px auto;padding:0 20px;background:#07080a;color:#f4f5f6;font:15px/1.55 system-ui,sans-serif}a{color:#a8e23a}pre{white-space:pre-wrap;font:14px/1.6 ui-monospace,SFMono-Regular,monospace;background:#111419;padding:20px;border-radius:8px;overflow:auto}nav{margin-bottom:28px}h1,h2,h3{line-height:1.15;letter-spacing:-.02em}h1{font-size:36px}h2{font-size:25px;margin-top:38px}h3{font-size:19px;margin-top:28px}p{max-width:75ch}li{margin:5px 0}.doc-table-wrap{overflow-x:auto;margin:18px 0 24px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid #363a40}th{color:#a8e23a;font-weight:650}code{font:13px ui-monospace,SFMono-Regular,monospace;background:#111419;padding:2px 5px;border-radius:3px}</style>" +
            nav + page_heading + body).encode()

def commit() -> str:
    value = os.environ.get("RELEASE_SHA")
    if value: return value[:12]
    try: return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return "local"

def read_jsonl(path: Path):
    if not path.exists(): return []
    rows=[]
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    return rows

def page_payload(name: str) -> dict | None:
    """Read a committed page-session result at request time."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return None
    path = PAGE_DATA_DIR / f"{name}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None

def static_content_type(path: Path) -> str:
    return {".webp": "image/webp", ".png": "image/png", ".svg": "image/svg+xml", ".css": "text/css", ".js": "text/javascript"}.get(path.suffix.lower(), "application/octet-stream")

def _read_snapshot_artifact() -> dict:
    try:
        return json.loads(SNAPSHOT_ARTIFACT.read_text())
    except Exception:
        return {}

def _safe_query_ms(value) -> dict:
    """Keep persisted timing metadata out of the public feature surface."""
    if not isinstance(value, Mapping):
        return {}
    safe = {}
    for key, raw in value.items():
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= number <= 300:
            safe[str(key)] = round(number, 3)
    return safe

def _preview_key(diagnosis: Mapping) -> str:
    raw = f"{diagnosis.get('rollout_id', 'failure')}__{Path(str(diagnosis.get('frame', 'frame.png'))).name}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)

def _merge_artifact_sections(snap: dict) -> dict:
    """Keep recorded local evidence visible when the cloud DB is older."""
    artifact = _read_snapshot_artifact()
    if not snap.get("diagnoses") and artifact.get("diagnoses"):
        snap["diagnoses"] = [dict(row) for row in artifact["diagnoses"]]
        snap["diagnoses_source"] = "local aggregate snapshot"
    if not snap.get("stations") and artifact.get("stations"):
        snap["stations"] = [dict(row) for row in artifact["stations"]]
        snap["worker"] = artifact.get("worker")
        snap["stations_source"] = "local aggregate snapshot"
    if artifact.get("snapshot_at"):
        snap.setdefault("artifact_snapshot_at", artifact["snapshot_at"])
    for row in snap.get("diagnoses") or []:
        row.setdefault("frame_url", "/evidence/failures/" + _preview_key(row))
    return snap

def _artifact_stations() -> dict:
    artifact = _read_snapshot_artifact()
    rows = artifact.get("stations") or []
    if not rows:
        raw = read_jsonl(ROOT / "data" / "stations36.jsonl")
        rows = [{"request_id": i + 1, "sentence": r.get("sentence", ""),
                 "status": "accepted" if r.get("accepted") else "rejected",
                 "result": r.get("expert") or r.get("check"), "error": r.get("reason")}
                for i, r in enumerate(raw)]
    return {"stations": rows, "worker": artifact.get("worker"), "status": "ok",
            "source": "recorded station artifact" if rows else "none"}

def reach_payload() -> dict:
    try:
        payload = json.loads(REACH_MAP_ARTIFACT.read_text())
        cells = payload.get("cells") or []
        summary = {}
        for size in sorted({str(c.get("size", "unknown")) for c in cells}):
            subset = [c for c in cells if str(c.get("size")) == size]
            rates = [float(c.get("pick_rate", 0)) for c in subset]
            threshold = float(payload.get("threshold", .95))
            summary[size] = {"cells": len(subset), "accepted_cells": sum(r >= threshold for r in rates),
                             "mean_pick_rate": round(sum(rates) / len(rates), 4) if rates else None,
                             "min_pick_rate": min(rates) if rates else None}
        return {"status": "ok", "source": "measured reach artifact", "version": payload.get("version"),
                "threshold": payload.get("threshold"), "calibration": payload.get("calibration"),
                "summary": summary, "cells": cells,
                "stage_success": None, "message": "The artifact measures pick and 10 mm lift; stage-by-stage success is not recorded."}
    except Exception:
        return {"status": "unavailable", "source": "none", "message": "Reach-map artifact unavailable.", "summary": {}, "cells": []}

def replay_provenance(payload: dict) -> str:
    """Classify replay evidence without treating placeholder commits as proof."""
    outcome = payload.get("outcome") if isinstance(payload.get("outcome"), dict) else {}
    commit_value = payload.get("commit") or outcome.get("commit")
    return "post-main" if commit_value and str(commit_value).lower() not in {"unknown", "local"} else "archived pre-handoff"

def _external_replay_paths() -> list[Path]:
    """Return replay files mounted for the current deployment."""
    if not REPLAY_DIR.exists():
        return []
    return sorted({*REPLAY_DIR.glob("seed-*/manifest.json"), *REPLAY_DIR.rglob("*.json.gz")})

def _external_post_main() -> bool:
    """Whether the mounted replay set contains a post-handoff artifact."""
    for path in _external_replay_paths():
        try:
            if path.name == "manifest.json":
                payload = json.loads(path.read_text())
            else:
                import gzip
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    payload = json.load(fh)
            if replay_provenance(payload) == "post-main":
                return True
        except Exception:
            continue
    return False

def _embedded_replay(seed: int) -> dict | None:
    """Read the legacy embedded payload only for mesh compatibility."""
    path = ROOT / "data" / f"replay-seed{seed}.jsonl"
    rows = read_jsonl(path)
    return rows[-1] if rows else None

def codex_payload() -> dict:
    """Return the page-safe Codex timeline and truthful cast availability."""
    entries = read_jsonl(CODEX_LOG)
    return {
        "status": "ok" if entries else "unavailable",
        "entries": entries,
        "entry_count": len(entries),
        "cast_available": CODEX_CAST.is_file(),
        "cast_path": CODEX_CAST.name if CODEX_CAST.is_file() else None,
        "message": "Codex log loaded." if entries else "No Codex log is connected.",
    }

def benchmark_tiers() -> dict:
    # Tiger is authoritative when connected; the checked file is an honest
    # fallback for local/offline page builds.
    if os.environ.get("DRYRUN_DB_URL"):
        try:
            from dryrun.telemetry import tier_snapshot
            return tier_snapshot()
        except Exception:
            fallback_error = True
        else:
            fallback_error = None
    else:
        fallback_error = None
    try:
        payload = json.loads(BENCHMARK_TIERS.read_text())
        payload.setdefault("source", "artifact")
        if fallback_error:
            payload["database_error"] = "Tiger telemetry unavailable; showing the checked artifact."
        return payload
    except Exception:
        return {"status": "unavailable", "tiers": [], "message": "No tier benchmark artifact is connected."}

def prompt024_examples() -> dict:
    """Return the two full-sheet examples retained for the analyst page."""
    for path in PROMPT024_EXAMPLES:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        examples = []
        for row in rows[:2]:
            examples.append({
                "id": row.get("id"),
                "tier": row.get("tier"),
                "sheet": row.get("sheet_data_uri"),
                "said": row.get("response", {}).get("diagnosis", row.get("response")),
                "actual": row.get("ground_truth"),
            })
        return {"status": "ok", "count": len(rows), "examples": examples, "source": path.name}
    return {"status": "unavailable", "count": 0, "examples": [], "message": "Prompt 024 examples are not attached to this deployment."}

def snapshot() -> dict:
    try:
        from dryrun.telemetry import dashboard_snapshot
        snap = dashboard_snapshot()
        snap["commit"] = commit(); return _merge_artifact_sections(snap)
    except Exception:
        if SNAPSHOT_ARTIFACT.exists():
            try:
                snap=json.loads(SNAPSHOT_ARTIFACT.read_text()); snap["commit"]=commit(); snap["source"]="local snapshot"; snap["status"]="stale"; snap["query_ms"]=_safe_query_ms(snap.get("query_ms")); return _merge_artifact_sections(snap)
            except Exception: pass
        return {"status":"unavailable","source":"none","reason":"No telemetry aggregate is connected yet.","commit":commit(),"checkpoints":[],"rollups":[],"diagnoses":[],"raw_rows":0,"result_rows":0,"query_ms":{}}

def replay_payload(seed: int) -> dict | None:
    candidates=[REPLAY_DIR/f"seed-{seed}"/"replay.json.gz", REPLAY_DIR/f"seed-{seed}c"/"replay.json.gz"]
    if not _external_post_main():
        candidates.append(ROOT/"data"/f"replay-seed{seed}.jsonl")
    for path in candidates:
        if not path.exists(): continue
        try:
            if path.suffix==".gz":
                import gzip
                with gzip.open(path,"rt",encoding="utf-8") as fh: payload = json.load(fh)
            else:
                rows=read_jsonl(path); payload = rows[-1] if rows else None
            if payload is not None:
                payload.setdefault("provenance", replay_provenance(payload))
            return payload
        except Exception: continue
    if REPLAY_DIR.exists():
        import gzip
        for path in sorted(REPLAY_DIR.rglob("*.json.gz")):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh: payload = json.load(fh)
                if int(payload.get("seed", -1)) != int(seed): continue
                payload.setdefault("provenance", replay_provenance(payload))
                return payload
            except Exception: continue
    return None

def tier_replay_dir(tier: str) -> Path | None:
    """Locate the featured simulated replay for a page tier."""
    tier = str(tier).lower()
    remote_name = {"messy": "touching", "pile": "pile"}.get(tier, tier)
    candidates = [REPLAY_DIR / "tiers" / remote_name]
    if tier == "loose":
        candidates.extend([REPLAY_DIR / "seed-1001", REPLAY_DIR / "seed-7"])
    local = TIER_REPLAY_LOCAL_DIRS.get(tier)
    if local is not None:
        candidates.append(local)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "top-view.mp4").is_file():
            return candidate
    return None

def tier_replay_payload(tier: str) -> dict | None:
    directory = tier_replay_dir(tier)
    if directory is None:
        return None
    path = directory / "replay.json.gz"
    if not path.is_file():
        return None
    try:
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload.setdefault("videos", {"top": "top-view.mp4", "inspection": "inspection-camera.mp4"})
        payload.setdefault("tier", tier)
        payload.setdefault("provenance", replay_provenance(payload))
        return payload
    except Exception:
        return None

def replay_list():
    rows=[]
    mounted_post_main = _external_post_main()
    if not mounted_post_main:
        for path in sorted((ROOT/"data").glob("replay-*.jsonl")):
            try:
                obj=read_jsonl(path)[-1]; rows.append({"seed":obj.get("seed"),"checkpoint":obj.get("checkpoint"),"outcome":obj.get("outcome"),"events":len(obj.get("events",[])),"frames":len(obj.get("frames",[])),"source":"recorded trace","provenance":replay_provenance(obj)})
            except Exception: pass
    if REPLAY_DIR.exists():
        for path in sorted(REPLAY_DIR.glob("seed-*/manifest.json")):
            try:
                obj=json.loads(path.read_text()); obj.setdefault("provenance", replay_provenance(obj))
                if not mounted_post_main or obj["provenance"] == "post-main": rows.append(obj | {"source":"local MuJoCo export"})
            except Exception: pass
        # Benchmark exports are compact gzip traces without a manifest. Keep
        # their scene/result rows visible to the workspace while the featured
        # replay can still come from the full video export above.
        import gzip
        for path in sorted(REPLAY_DIR.rglob("*.json.gz")):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh: obj=json.load(fh)
                if obj.get("seed") is None: continue
                outcome=obj.get("outcome") or {}
                provenance = replay_provenance(obj)
                if not mounted_post_main or provenance == "post-main":
                    rows.append({"seed":obj.get("seed"),"checkpoint":obj.get("checkpoint") or obj.get("controller"),"tier":obj.get("tier"),"outcome":outcome,"events":len(obj.get("events",[])),"frames":len(obj.get("frames",[])),"source":"benchmark replay export","commit":obj.get("commit"),"provenance":provenance})
            except Exception: pass
    unique={int(r.get("seed")):r for r in rows if r.get("seed") is not None}
    return list(unique.values())

def score_payload(snap):
    # A checkpoint without a producing source revision is pre-handoff
    # validation. Keep it out of the public score once a real post-main row
    # exists; the page must never blend it into the new benchmark.
    cps=[row for row in (snap.get("checkpoints") or []) if row.get("source_commit")]
    if not cps:
        # Tier benchmarks are the public score source for this release. The
        # optional telemetry rollup may be absent on Fly; keep that honest
        # empty state from turning into a page-wide API error.
        return {"status":snap.get("status", "ok"),"message":"No telemetry checkpoint is connected; tier benchmark is shown above.","commit":snap.get("commit"),"checkpoints":[],"diagnoses":snap.get("diagnoses",[]),"source":snap.get("source"),"query_ms":_safe_query_ms(snap.get("query_ms")),"raw_rows":snap.get("raw_rows",0),"result_rows":snap.get("result_rows",0),"compression_enabled":snap.get("compression_enabled"),"compression_ratio":snap.get("compression_ratio"),"latest_rollout_at":snap.get("latest_rollout_at"),"snapshot_at":snap.get("snapshot_at"),"artifact_snapshot_at":snap.get("artifact_snapshot_at")}
    latest=dict(cps[-1])
    for key in ("bricks", "correct", "drops", "false_accepts", "false_rejects", "rollouts"):
        if latest.get(key) is not None:
            try: latest[key] = int(latest[key])
            except (TypeError, ValueError): pass
    if latest.get("cycle_time_s") is not None:
        try: latest["cycle_time_s"] = float(latest["cycle_time_s"])
        except (TypeError, ValueError): pass
    latest["accuracy"]=latest["correct"]/latest["bricks"] if latest.get("bricks") else None
    latest["benchmarkable"]=bool(latest.get("source_commit"))
    message = None if latest["benchmarkable"] else "Recorded result has no source commit; validation only, not a benchmark."
    return {"status":snap.get("status", "ok"),"checkpoint":latest,"commit":snap.get("commit"),"data_commit":latest.get("source_commit"),"message":message,"checkpoints":cps,"diagnoses":snap.get("diagnoses",[]),"source":snap.get("source"),"query_ms":_safe_query_ms(snap.get("query_ms")),"raw_rows":snap.get("raw_rows",0),"result_rows":snap.get("result_rows",0),"compression_enabled":snap.get("compression_enabled"),"compression_ratio":snap.get("compression_ratio"),"latest_rollout_at":snap.get("latest_rollout_at"),"snapshot_at":snap.get("snapshot_at"),"artifact_snapshot_at":snap.get("artifact_snapshot_at")}


def inspector_payload() -> dict:
    """Read the compact inspector artifact consumed by the page."""
    path = None
    metrics = None
    for candidate in INSPECTOR_PAGE_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            candidate_metrics = json.loads(candidate.read_text())
        except Exception:
            continue
        # An empty page projection is an incomplete rebuild artifact.  Skip
        # it rather than replacing a usable prior page payload with blanks.
        if candidate_metrics.get("confusion") or candidate_metrics.get("auroc") is not None:
            path, metrics = candidate, candidate_metrics
            break
    if path is None:
        path = INSPECTOR_EVALUATION
    try:
        if metrics is None:
            metrics = json.loads(path.read_text())
        metrics["status"] = "ok"
        metrics["source"] = path.name
        metrics.setdefault("schema", "dryrun.inspector.v1")
        if "score_histogram" not in metrics:
            metrics["score_histogram"] = metrics.get("score_histograms", {})
        return metrics
    except Exception:
        return {
            "schema": "dryrun.inspector.v1", "status": "unavailable",
            "source": INSPECTOR_EVALUATION.name, "message": "Inspector evaluation artifact unavailable.",
            "confusion": {}, "per_defect_recall": {}, "latency": {},
            "score_histograms": {"edges": [], "bins": [], "counts": [], "per_type": {}},
            "score_histogram": {"edges": [], "bins": [], "counts": [], "per_type": {}},
        }

class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, body: bytes, content_type: str, status=200):
        self.send_response(status); self.send_header("Content-Type",content_type); self.send_header("Cache-Control","no-cache"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def send_video_file(self, path: Path):
        """Serve replay video with byte ranges so the inspector can seek."""
        total = path.stat().st_size
        start, end = 0, total - 1
        header = self.headers.get("Range", "")
        if header.startswith("bytes="):
            value = header[6:].split(",", 1)[0].strip()
            try:
                left, right = value.split("-", 1)
                if left:
                    start = int(left)
                    end = int(right) if right else end
                else:
                    length = int(right)
                    start = max(0, total - length)
                if start < 0 or start >= total or end < start:
                    raise ValueError
                end = min(end, total - 1)
            except (TypeError, ValueError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return
        length = end - start + 1
        status = 206 if header.startswith("bytes=") else 200
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            self.wfile.write(fh.read(length))
    def send_json(self,obj,status=200): self.send_bytes(json.dumps(obj,separators=(",",":"),default=str).encode(),"application/json; charset=utf-8",status)
    def do_GET(self):
        parsed=urlparse(self.path); path=unquote(parsed.path)
        if path in ("/codex", "/prompts", "/submission", "/morning"):
            slug = path.strip("/")
            body = document_page(slug)
            return self.send_bytes(body, "text/html; charset=utf-8") if body else self.send_json({"status": "unavailable", "message": "Evidence document is not attached."}, 404)
        if path.startswith("/prompts/"):
            name = path.rsplit("/", 1)[-1]
            body = document_page("prompt", name)
            return self.send_bytes(body, "text/html; charset=utf-8") if body else self.send_json({"status": "unavailable", "message": "Prompt text is not attached."}, 404)
        if path.startswith("/api/page/"):
            name = path.rsplit("/", 1)[-1]
            payload = page_payload(name)
            return self.send_json(payload or {"status": "unavailable", "message": "This page section has no committed result yet."}, 200 if payload else 404)
        if path.startswith("/api/prompts/"):
            name = path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[A-Za-z0-9_-]+\.md", name):
                return self.send_json({"error": "invalid prompt path"}, 400)
            prompt = (ROOT / "prompts" / name).resolve()
            if prompt.parent != (ROOT / "prompts").resolve() or not prompt.is_file():
                return self.send_json({"status": "unavailable", "message": "Prompt text is not attached."}, 404)
            return self.send_bytes(prompt.read_bytes(), "text/plain; charset=utf-8")
        if path.startswith("/status/"):
            name = path.rsplit("/", 1)[-1]
            body = document_page("status", name)
            return self.send_bytes(body, "text/html; charset=utf-8") if body else self.send_json({"status": "unavailable", "message": "Status document is not attached."}, 404)
        if path=="/api/score":
            payload = score_payload(snapshot())
            return self.send_json(payload, 200 if payload.get("status") in {None, "ok"} else 503)
        if path=="/api/benchmark-tiers": return self.send_json(benchmark_tiers())
        if path=="/api/codex": return self.send_json(codex_payload())
        if path=="/api/telemetry":
            payload = snapshot()
            return self.send_json(payload, 200 if payload.get("status") == "ok" else 503)
        if path=="/api/failures": return self.send_json(snapshot().get("diagnoses", []))
        if path=="/api/prompt024": return self.send_json(prompt024_examples())
        if path=="/api/reach": return self.send_json(reach_payload())
        if path in ("/api/inspector", "/api/inspector.json"): return self.send_json(inspector_payload())
        if path=="/api/stations":
            try:
                from dryrun.telemetry import station_snapshot
                result = station_snapshot()
                return self.send_json(result if result.get("stations") else _artifact_stations())
            except Exception as exc:
                # The station queue is an optional analyst-session extension.
                # Keep the rest of the evidence page healthy when an older
                # Tiger schema has not received that migration yet.
                result = _artifact_stations()
                result.setdefault("message", "worker offline")
                return self.send_json(result)
        if path=="/api/rollouts": return self.send_json(replay_list())
        if path=="/api/replays": return self.send_json(replay_list())
        if path=="/api/scenes": return self.send_json(replay_list())
        if path.startswith("/api/tier-replays/"):
            tier = path.rsplit("/", 1)[-1]
            obj = tier_replay_payload(tier) if re.fullmatch(r"[A-Za-z0-9_-]+", tier) else None
            return self.send_json(obj or {"status":"unavailable","message":"That tier replay is not mounted."}, 200 if obj else 404)
        if path.startswith("/api/replays/"):
            try: obj=replay_payload(int(path.rsplit("/",1)[1]))
            except (ValueError,TypeError): obj=None
            return self.send_json(obj or {"status":"unavailable","message":"That rollout is not in the recorded set."}, 200 if obj else 404)
        if path.startswith("/replays/"):
            bits=path.strip("/").split("/"); file=bits[-1]; seed=bits[-2] if len(bits)>1 else ""
            candidates = [REPLAY_DIR/seed/file]
            if seed.isdigit():
                candidates.extend(REPLAY_DIR.glob(f"seed-{seed}*/{file}"))
            for candidate in candidates:
                if candidate.is_file(): return self.send_video_file(candidate)
            return self.send_json({"status":"unavailable","message":"Camera video is available in the local replay export."},404)
        if path.startswith("/tier-replay/"):
            bits = path.strip("/").split("/")
            tier, name = (bits[-2], bits[-1]) if len(bits) >= 3 else ("", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", tier) or name not in {"top-view.mp4", "inspection-camera.mp4"}:
                return self.send_json({"error":"invalid tier replay path"}, 400)
            directory = tier_replay_dir(tier)
            candidate = directory / name if directory is not None else None
            if candidate is not None and candidate.is_file():
                return self.send_video_file(candidate)
            return self.send_json({"status":"unavailable","message":"That tier replay video is not mounted."},404)
        if path.startswith("/replay-video/"):
            bits=path.strip("/").split("/")
            name=bits[-1]
            seed_text=bits[-2] if len(bits) >= 3 else "7"
            candidates=[]
            if seed_text.isdigit():
                candidates.extend(REPLAY_DIR.glob(f"seed-{seed_text}*/{name}"))
            candidates.append(REPLAY_DIR/name)
            for candidate in candidates:
                if candidate.is_file(): return self.send_video_file(candidate)
            try:
                seed_record = replay_payload(int(seed_text)) if seed_text.isdigit() else None
                key = "top" if name == "top-view.mp4" else "inspection"
                encoded = (seed_record or {}).get("video_data", {}).get(key)
                if encoded:
                    import base64
                    return self.send_bytes(base64.b64decode(encoded),"video/mp4")
            except Exception:
                pass
            return self.send_json({"status":"unavailable","message":"Camera video is available in the local replay export."},404)
        if path.startswith("/evidence/failures/"):
            key = path.rsplit("/", 1)[-1]
            if not key or key in {".", ".."} or "/" in key or "\\" in key:
                return self.send_json({"error":"invalid evidence path"}, 400)
            candidate = EVIDENCE_DIR / key
            if candidate.is_file() and candidate.resolve().parent == EVIDENCE_DIR.resolve():
                return self.send_bytes(candidate.read_bytes(), "image/png")
            return self.send_json({"status":"unavailable","message":"Failure frame is not mounted for this deployment."},404)
        if path=="/api/health": return self.send_json({"status":"ok","commit":commit(),"replay_count":len(replay_list())})
        if path in ("/","/index.html"):
            return self.send_bytes((WEB/"index.html").read_bytes(),"text/html; charset=utf-8")
        if path.startswith("/static/"):
            relative = path[len("/static/"):].lstrip("/")
            candidate = (STATIC_DIR / relative).resolve()
            if candidate.is_file() and STATIC_DIR.resolve() in candidate.parents:
                return self.send_bytes(candidate.read_bytes(), static_content_type(candidate))
            return self.send_json({"status": "unavailable", "message": "Static evidence is not attached."}, 404)
        if path in ("/favicon.svg", "/favicon.ico"): return self.send_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#12253d"/><path d="M8 22 16 7l8 15-8-4z" fill="#a7e34b"/></svg>',"image/svg+xml")
        if path in ("/mujoco-station.glb", "/so101.glb"):
            for candidate in VIEWER_SCENE_CANDIDATES:
                if candidate.is_file(): return self.send_bytes(candidate.read_bytes(),"model/gltf-binary")
            try:
                encoded=(_embedded_replay(7) or {}).get("mesh_data")
                if encoded:
                    import base64
                    return self.send_bytes(base64.b64decode(encoded),"model/gltf-binary")
            except Exception:
                pass
            return self.send_json({"status":"unavailable","message":"The compressed SO-101 mesh bundle is present in the local run assets."},404)
        self.send_json({"error":"not found"},404)
    def do_POST(self):
        if self.path == "/api/stations":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                sentence = str(body.get("sentence", "")).strip()
                from dryrun.telemetry import enqueue_station_request
                request_id = enqueue_station_request(sentence)
                return self.send_json({"ok": True, "request_id": request_id, "status": "pending", "worker": "station-worker"}, 202)
            except Exception:
                return self.send_json({"error": "invalid station request"}, 400)
        if self.path!="/api/defect": return self.send_json({"error":"not found"},404)
        try:
            body=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
            body.update({"created_at":time.time(),"status":"queued","verdict":"pending"})
            p=ROOT/"data"/"defects.jsonl"; p.parent.mkdir(exist_ok=True); p.open("a").write(json.dumps(body,separators=(",",":"))+"\n")
            result={"ok":True,"defect_id":f"custom-{int(time.time()*1000)}","verdict":"pending","message":"Saved. A rollout will run when the inspector worker is connected."}
            # A deployed Modal URL is the preferred live worker. Local
            # development can opt into the same path with
            # DRYRUN_LIVE_ROLLOUT=1; Fly remains honest when neither exists.
            modal_url = os.environ.get("DRYRUN_MODAL_URL")
            if modal_url and body.get("drawing", "").startswith("data:image"):
                try:
                    import urllib.request
                    req = urllib.request.Request(modal_url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=900) as response:
                        live = json.loads(response.read())
                    result.update(live)
                    result["message"] = "Live Modal inspector complete." + (" One-brick rollout recorded." if live.get("run_id") else " Set rollout=true to record a one-brick replay.")
                    return self.send_json(result, 201)
                except Exception:
                    result["message"] = "Saved; live Modal worker unavailable."
            # On the full local box, use the exact render-trained OpenVINO
            # inspector. Fly remains honest when its optional model is absent.
            drawing=body.get("drawing","")
            if drawing.startswith("data:image"):
                try:
                    import base64, io
                    import numpy as np
                    from PIL import Image
                    from dryrun.defects import render_drawing_frame
                    from dryrun.inspector import Inspector
                    raw=base64.b64decode(drawing.split(",",1)[1]); arr=np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
                    frame=render_drawing_frame(arr,seed=int(body.get("seed",0)))
                    decision=Inspector().inspect(frame); heat=Image.fromarray(decision["heatmap"])
                    buf=io.BytesIO(); heat.save(buf,format="PNG")
                    result.update({"verdict":decision["verdict"],"score":decision["score"],"threshold":decision["threshold"],"heatmap":"data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode(),"message":"Inspector result from the rendered custom brick. The live sort is queued."})
                    if os.environ.get("DRYRUN_LIVE_ROLLOUT") == "1":
                        from dryrun.live_defect import run_custom_defect
                        result.update(run_custom_defect(body))
                        result["message"] = "Live local inspector and one-brick MuJoCo rollout complete."
                except Exception:
                    result["message"]="Saved; local inspector unavailable."
            return self.send_json(result,201)
        except Exception: return self.send_json({"error":"invalid defect"},400)
    def log_message(self,*args): pass

if __name__=="__main__":
    port=int(os.environ.get("PORT","8787")); print(f"Dry Run dashboard http://localhost:{port}"); ThreadingHTTPServer(("0.0.0.0",port),Handler).serve_forever()
