#!/usr/bin/env python3
"""Small dependency-light server for the Dry Run evidence page."""
from __future__ import annotations
import json, os, subprocess, time, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
RUNS = Path("/bulk3/runs/dryrun")
REPLAY_DIR = Path(os.environ.get("DRYRUN_REPLAY_DIR", str(RUNS / "_b2/replays/seed-7c")))
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

def codex_payload() -> dict:
    """Return the page-safe Codex timeline and truthful cast availability."""
    entries = read_jsonl(CODEX_LOG)
    return {
        "status": "ok" if entries else "unavailable",
        "entries": entries,
        "entry_count": len(entries),
        "cast_available": CODEX_CAST.is_file(),
        "cast_path": str(CODEX_CAST) if CODEX_CAST.is_file() else None,
        "message": "Codex log loaded." if entries else "No Codex log is connected.",
    }

def snapshot() -> dict:
    try:
        from dryrun.telemetry import dashboard_snapshot
        snap = dashboard_snapshot()
        snap["commit"] = commit(); return snap
    except Exception as exc:
        p=ROOT/"data"/"snapshot.json"
        if p.exists():
            try:
                snap=json.loads(p.read_text()); snap["commit"]=commit(); snap["source"]="local snapshot"; return snap
            except Exception: pass
        return {"status":"unavailable","source":"none","reason":"No telemetry aggregate is connected yet.","error":str(exc),"commit":commit(),"checkpoints":[],"rollups":[],"diagnoses":[],"raw_rows":0,"result_rows":0,"query_ms":{}}

def replay_payload(seed: int) -> dict | None:
    candidates=[REPLAY_DIR/f"seed-{seed}"/"replay.json.gz", REPLAY_DIR/f"seed-{seed}c"/"replay.json.gz", ROOT/"data"/f"replay-seed{seed}.jsonl"]
    for path in candidates:
        if not path.exists(): continue
        try:
            if path.suffix==".gz":
                import gzip
                with gzip.open(path,"rt",encoding="utf-8") as fh: return json.load(fh)
            rows=read_jsonl(path); return rows[-1] if rows else None
        except Exception: continue
    return None

def replay_list():
    rows=[]
    for path in sorted((ROOT/"data").glob("replay-*.jsonl")):
        try:
            obj=read_jsonl(path)[-1]; rows.append({"seed":obj.get("seed"),"checkpoint":obj.get("checkpoint"),"outcome":obj.get("outcome"),"events":len(obj.get("events",[])),"frames":len(obj.get("frames",[])),"source":"recorded trace"})
        except Exception: pass
    if REPLAY_DIR.exists():
        for path in sorted(REPLAY_DIR.glob("seed-*/manifest.json")):
            try: rows.append(json.loads(path.read_text()) | {"source":"local MuJoCo export"})
            except Exception: pass
    unique={int(r.get("seed")):r for r in rows if r.get("seed") is not None}
    return list(unique.values())

def score_payload(snap):
    cps=snap.get("checkpoints") or []
    if not cps: return {"status":"unavailable","message":"No benchmark checkpoint has been recorded in telemetry yet.","commit":snap.get("commit"),"checkpoints":[]}
    latest=dict(cps[-1]); latest["accuracy"]=latest["correct"]/latest["bricks"] if latest.get("bricks") else None
    latest["benchmarkable"]=bool(latest.get("source_commit"))
    message = None if latest["benchmarkable"] else "Recorded result has no source commit; validation only, not a benchmark."
    return {"status":"ok","checkpoint":latest,"commit":snap.get("commit"),"data_commit":latest.get("source_commit"),"message":message,"checkpoints":cps,"diagnoses":snap.get("diagnoses",[]),"source":snap.get("source"),"query_ms":snap.get("query_ms"),"raw_rows":snap.get("raw_rows",0),"result_rows":snap.get("result_rows",0),"compression_enabled":snap.get("compression_enabled"),"compression_ratio":snap.get("compression_ratio"),"latest_rollout_at":snap.get("latest_rollout_at")}


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
        metrics["source"] = str(path)
        metrics.setdefault("schema", "dryrun.inspector.v1")
        if "score_histogram" not in metrics:
            metrics["score_histogram"] = metrics.get("score_histograms", {})
        return metrics
    except Exception as exc:
        return {
            "schema": "dryrun.inspector.v1", "status": "unavailable",
            "source": str(INSPECTOR_EVALUATION), "message": str(exc),
            "confusion": {}, "per_defect_recall": {}, "latency": {},
            "score_histograms": {"edges": [], "bins": [], "counts": [], "per_type": {}},
            "score_histogram": {"edges": [], "bins": [], "counts": [], "per_type": {}},
        }

class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, body: bytes, content_type: str, status=200):
        self.send_response(status); self.send_header("Content-Type",content_type); self.send_header("Cache-Control","no-cache"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def send_json(self,obj,status=200): self.send_bytes(json.dumps(obj,separators=(",",":"),default=str).encode(),"application/json; charset=utf-8",status)
    def do_GET(self):
        parsed=urlparse(self.path); path=unquote(parsed.path)
        if path=="/api/score": return self.send_json(score_payload(snapshot()))
        if path=="/api/codex": return self.send_json(codex_payload())
        if path=="/api/telemetry": return self.send_json(snapshot())
        if path=="/api/failures": return self.send_json(snapshot().get("diagnoses", []))
        if path in ("/api/inspector", "/api/inspector.json"): return self.send_json(inspector_payload())
        if path=="/api/stations":
            try:
                from dryrun.telemetry import station_snapshot
                return self.send_json(station_snapshot())
            except Exception as exc:
                # The station queue is an optional analyst-session extension.
                # Keep the rest of the evidence page healthy when an older
                # Tiger schema has not received that migration yet.
                return self.send_json({"stations": [], "worker": None, "status": "ok", "message": "worker offline"})
        if path=="/api/rollouts": return self.send_json(replay_list())
        if path=="/api/replays": return self.send_json(replay_list())
        if path=="/api/scenes": return self.send_json(replay_list())
        if path.startswith("/api/replays/"):
            try: obj=replay_payload(int(path.rsplit("/",1)[1]))
            except (ValueError,TypeError): obj=None
            return self.send_json(obj or {"status":"unavailable","message":"That rollout is not in the recorded set."}, 200 if obj else 404)
        if path.startswith("/replays/"):
            bits=path.strip("/").split("/"); file=bits[-1]; seed=bits[-2] if len(bits)>1 else ""
            candidate=REPLAY_DIR/seed/file
            if candidate.is_file(): return self.send_bytes(candidate.read_bytes(),"video/mp4")
            return self.send_json({"status":"unavailable","message":"Camera video is available in the local replay export."},404)
        if path.startswith("/replay-video/"):
            candidate=REPLAY_DIR/path.rsplit("/",1)[1]
            if candidate.is_file(): return self.send_bytes(candidate.read_bytes(),"video/mp4")
            try:
                seed_record = replay_payload(7)
                name = path.rsplit("/",1)[1]
                key = "top" if name == "top-view.mp4" else "inspection"
                encoded = (seed_record or {}).get("video_data", {}).get(key)
                if encoded:
                    import base64
                    return self.send_bytes(base64.b64decode(encoded),"video/mp4")
            except Exception:
                pass
            return self.send_json({"status":"unavailable","message":"Camera video is available in the local replay export."},404)
        if path=="/api/health": return self.send_json({"status":"ok","commit":commit(),"replay_count":len(replay_list())})
        if path in ("/","/index.html"):
            return self.send_bytes((WEB/"index.html").read_bytes(),"text/html; charset=utf-8")
        if path=="/favicon.svg": return self.send_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#12253d"/><path d="M8 22 16 7l8 15-8-4z" fill="#a7e34b"/></svg>',"image/svg+xml")
        if path=="/so101.glb":
            candidate=RUNS / "page-assets/so101.glb"
            if candidate.is_file(): return self.send_bytes(candidate.read_bytes(),"model/gltf-binary")
            try:
                encoded=(replay_payload(7) or {}).get("mesh_data")
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
            except Exception as exc:
                return self.send_json({"error": f"invalid station request: {exc}"}, 400)
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
                except Exception as exc:
                    result["message"] = "Saved; live Modal worker unavailable: " + str(exc)
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
                except Exception as exc:
                    result["message"]="Saved; local inspector unavailable: "+str(exc)
            return self.send_json(result,201)
        except Exception as exc: return self.send_json({"error":f"invalid defect: {exc}"},400)
    def log_message(self,*args): pass

if __name__=="__main__":
    port=int(os.environ.get("PORT","8787")); print(f"Dry Run dashboard http://localhost:{port}"); ThreadingHTTPServer(("0.0.0.0",port),Handler).serve_forever()
