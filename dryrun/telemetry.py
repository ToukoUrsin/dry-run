"""The single telemetry boundary used by benchmark and replay export."""
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping
try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:
    psycopg = None
    Jsonb = lambda x: x

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "data" / "schema.sql"
DEFAULT_URL = "postgresql://postgres:dryrun@127.0.0.1:55439/dryrun"

def db_url() -> str:
    return os.environ.get("DRYRUN_DB_URL", DEFAULT_URL)

def _require_db():
    if psycopg is None:
        raise RuntimeError("install psycopg[binary] to use DRYRUN_DB_URL")

def _sql_statements(text: str) -> list[str]:
    statements, buf, in_do = [], [], False
    for line in text.splitlines(True):
        buf.append(line)
        if line.lstrip().startswith("DO $$"): in_do = True
        if in_do and line.strip() == "END $$;":
            statements.append("".join(buf)); buf, in_do = [], False
        elif not in_do and line.rstrip().endswith(";"):
            statements.append("".join(buf)); buf = []
    if buf and "".join(buf).strip(): statements.append("".join(buf))
    return statements

def setup(url: str | None = None) -> None:
    _require_db()
    with psycopg.connect(url or db_url(), autocommit=True) as conn:
        # Migrate the early smoke schema without losing its raw trace.
        cols = {r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='rollout_telemetry'").fetchall()}
        if cols:
            if "joint" in cols: conn.execute("ALTER TABLE rollout_telemetry ALTER COLUMN joint DROP NOT NULL")
            for name, typ in (("checkpoint", "TEXT NOT NULL DEFAULT 'expert'"), ("q0", "DOUBLE PRECISION NOT NULL DEFAULT 0"), ("q1", "DOUBLE PRECISION NOT NULL DEFAULT 0"), ("q2", "DOUBLE PRECISION NOT NULL DEFAULT 0"), ("q3", "DOUBLE PRECISION NOT NULL DEFAULT 0"), ("q4", "DOUBLE PRECISION NOT NULL DEFAULT 0"), ("q5", "DOUBLE PRECISION NOT NULL DEFAULT 0"), ("gripper", "DOUBLE PRECISION NOT NULL DEFAULT 0"), ("event_data", "JSONB")):
                if name not in cols: conn.execute(f"ALTER TABLE rollout_telemetry ADD COLUMN {name} {typ}")
            if "joint" in cols:
                conn.execute("""UPDATE rollout_telemetry SET q0=COALESCE((joint->>'q0')::double precision,0),q1=COALESCE((joint->>'q1')::double precision,0),q2=COALESCE((joint->>'q2')::double precision,0),q3=COALESCE((joint->>'q3')::double precision,0),q4=COALESCE((joint->>'q4')::double precision,0),q5=COALESCE((joint->>'q5')::double precision,0),gripper=COALESCE((joint->>'q5')::double precision,0) WHERE joint IS NOT NULL""")
        conn.execute("DROP MATERIALIZED VIEW IF EXISTS rollout_1s CASCADE")
        conn.execute("DROP MATERIALIZED VIEW IF EXISTS benchmark_checkpoint CASCADE")
        result_cols = {r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='rollout_results'").fetchall()}
        pk_def = conn.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid=to_regclass('rollout_results') AND contype='p'").fetchone()
        if result_cols and ("finished_at" not in result_cols or (pk_def and "finished_at" not in pk_def[0])):
            # This table was created by the rejected smoke schema and has no
            # durable benchmark result; replace it with the partitionable form.
            if conn.execute("SELECT count(*) FROM rollout_results").fetchone()[0] == 0:
                conn.execute("DROP TABLE rollout_results CASCADE")
        for statement in _sql_statements(SCHEMA.read_text()): conn.execute(statement)

def _timestamp(value):
    if value is None: return datetime.now(timezone.utc)
    if isinstance(value, datetime): return value
    if isinstance(value, (float, int)): return datetime.fromtimestamp(value, timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

def record_rollout(samples: Iterable[Mapping], events: Iterable[Mapping], outcome: Mapping, *, rollout_id: str | None = None, seed: int = 0, checkpoint: str = "expert", url: str | None = None) -> str:
    """Persist one real rollout and its result; safe to call from benchmark."""
    _require_db(); rid = rollout_id or f"seed-{seed}-{int(time.time() * 1000)}"
    sample_rows, event_rows, out = list(samples), list(events), dict(outcome)
    with psycopg.connect(url or db_url()) as conn:
        with conn.cursor() as cur:
            # Benchmark retries use a stable rollout id.  Avoid duplicating
            # the 30 Hz hypertable rows when the same completed rollout is
            # sent again (for example after a failed aggregate refresh).
            cur.execute("SELECT 1 FROM rollout_results WHERE rollout_id=%s LIMIT 1", (rid,))
            if cur.fetchone() is not None:
                return rid
            for row in sample_rows:
                q = (list(row.get("qpos", row.get("joint", [0] * 6))) + [0] * 6)[:6]
                cur.execute("""INSERT INTO rollout_telemetry
                    (ts,rollout_id,seed,checkpoint,q0,q1,q2,q3,q4,q5,gripper,event,event_data,outcome)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (_timestamp(row.get("ts")), rid, seed, checkpoint, *[float(x) for x in q], float(row.get("gripper", q[5])), row.get("event"), json.dumps(row.get("event_data")) if row.get("event_data") is not None else None, None))
            for event in event_rows:
                q = (list(event.get("qpos", [0] * 6)) + [0] * 6)[:6]
                cur.execute("""INSERT INTO rollout_telemetry
                    (ts,rollout_id,seed,checkpoint,q0,q1,q2,q3,q4,q5,gripper,event,event_data,outcome)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (_timestamp(event.get("ts")), rid, seed, checkpoint, *[float(x) for x in q], float(event.get("gripper", q[5])), event.get("name", "event"), json.dumps(dict(event)), None))
            cur.execute("""INSERT INTO rollout_results
                (finished_at,rollout_id,seed,checkpoint,bricks,correct,drops,false_accepts,false_rejects,cycle_time_s,failure_reason,outcome)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING""",
                (datetime.now(timezone.utc), rid, seed, checkpoint, int(out.get("bricks", 0)), int(out.get("correct", out.get("successes", 0))), int(out.get("drops", 0)), int(out.get("false_accepts", 0)), int(out.get("false_rejects", 0)), out.get("cycle_time_s"), out.get("failure_reason"), json.dumps(out)))
        conn.commit()
    return rid

def record_diagnosis(diagnosis: Mapping, *, url: str | None = None) -> str:
    """Persist one validated camera-only analyst result for the page."""
    _require_db()
    required = ("rollout_id", "model", "version", "frame_paths", "failure_stage", "cause", "frame", "scene_features", "next_scenes")
    missing = [name for name in required if name not in diagnosis]
    if missing:
        raise ValueError("diagnosis missing: " + ",".join(missing))
    rid, version = str(diagnosis["rollout_id"]), str(diagnosis["version"])
    with psycopg.connect(url or db_url()) as conn:
        conn.execute(
            """INSERT INTO failure_diagnoses
               (rollout_id,model,analyst_version,frame_paths,failure_stage,cause,evidence_frame,scene_features,next_scenes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (rollout_id, analyst_version) DO UPDATE SET
                 model=EXCLUDED.model, frame_paths=EXCLUDED.frame_paths,
                 failure_stage=EXCLUDED.failure_stage, cause=EXCLUDED.cause,
                 evidence_frame=EXCLUDED.evidence_frame,
                 scene_features=EXCLUDED.scene_features, next_scenes=EXCLUDED.next_scenes""",
            (rid, str(diagnosis["model"]), version, Jsonb(list(diagnosis["frame_paths"])),
             str(diagnosis["failure_stage"]), str(diagnosis["cause"]), str(diagnosis["frame"]),
             Jsonb(dict(diagnosis["scene_features"])), Jsonb(list(diagnosis["next_scenes"]))),
        )
        conn.commit()
    return rid

def enqueue_station_request(sentence: str, *, url: str | None = None) -> int:
    """Queue a sentence for the local Astra station worker."""
    sentence = str(sentence).strip()
    if not 1 <= len(sentence) <= 300:
        raise ValueError("sentence must contain 1 to 300 characters")
    _require_db()
    with psycopg.connect(url or db_url()) as conn:
        row = conn.execute("INSERT INTO station_requests (sentence) VALUES (%s) RETURNING request_id", (sentence,)).fetchone()
        conn.commit()
    return int(row[0])

def station_snapshot(*, url: str | None = None) -> dict:
    """Return the station gallery and worker heartbeat for the page."""
    _require_db()
    with psycopg.connect(url or db_url()) as conn:
        rows = conn.execute("SELECT request_id,created_at,sentence,status,result,error,finished_at FROM station_requests ORDER BY created_at DESC LIMIT 200").fetchall()
        heartbeat = conn.execute("SELECT worker_name,seen_at,status,detail FROM worker_heartbeat ORDER BY seen_at DESC LIMIT 1").fetchone()
    gallery = [{"request_id": r[0], "created_at": r[1].isoformat() if hasattr(r[1], "isoformat") else r[1], "sentence": r[2], "status": r[3], "result": r[4], "error": r[5], "finished_at": r[6].isoformat() if hasattr(r[6], "isoformat") else r[6]} for r in rows]
    worker = None if heartbeat is None else {"worker_name": heartbeat[0], "seen_at": heartbeat[1].isoformat() if hasattr(heartbeat[1], "isoformat") else heartbeat[1], "status": heartbeat[2], "detail": heartbeat[3]}
    return {"stations": gallery, "worker": worker}

def worker_heartbeat(name: str, status: str, detail: Mapping | None = None, *, url: str | None = None) -> None:
    _require_db()
    with psycopg.connect(url or db_url()) as conn:
        conn.execute("""INSERT INTO worker_heartbeat (worker_name,seen_at,status,detail) VALUES (%s,now(),%s,%s)
                       ON CONFLICT (worker_name) DO UPDATE SET seen_at=EXCLUDED.seen_at,status=EXCLUDED.status,detail=EXCLUDED.detail""", (name, status, Jsonb(dict(detail or {}))))
        conn.commit()

def claim_station_request(*, url: str | None = None) -> tuple[int, str] | None:
    _require_db()
    with psycopg.connect(url or db_url()) as conn:
        row = conn.execute("""WITH next AS (SELECT request_id FROM station_requests WHERE status='pending' ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
                             UPDATE station_requests s SET status='running',started_at=now() FROM next WHERE s.request_id=next.request_id RETURNING s.request_id,s.sentence""").fetchone()
        conn.commit()
    return None if row is None else (int(row[0]), str(row[1]))

def finish_station_request(request_id: int, status: str, result: Mapping | None = None, error: str | None = None, *, url: str | None = None) -> None:
    if status not in {"accepted", "rejected", "error"}: raise ValueError("invalid station status")
    _require_db()
    with psycopg.connect(url or db_url()) as conn:
        conn.execute("UPDATE station_requests SET status=%s,result=%s,error=%s,finished_at=now() WHERE request_id=%s", (status, Jsonb(dict(result or {})) if result is not None else None, error, int(request_id)))
        conn.commit()

def refresh(url: str | None = None) -> None:
    _require_db()
    with psycopg.connect(url or db_url(), autocommit=True) as conn:
        conn.execute("CALL refresh_continuous_aggregate('rollout_1s', NULL, NULL)")
        conn.execute("CALL refresh_continuous_aggregate('benchmark_checkpoint', NULL, NULL)")

def dashboard_snapshot(url: str | None = None) -> dict:
    """Read only aggregate tables for the public page and deployment snapshot."""
    _require_db()
    with psycopg.connect(url or db_url()) as conn:
        started = time.perf_counter()
        rollups = conn.execute("SELECT bucket,rollout_id,seed,checkpoint,q0,q1,q2,q3,q4,q5,gripper,samples,joint_effort,status FROM rollout_1s ORDER BY bucket DESC LIMIT 240").fetchall()
        aggregate_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        summary = conn.execute("SELECT checkpoint,sum(bricks),sum(correct),sum(drops),sum(false_accepts),sum(false_rejects),avg(cycle_time_s),sum(rollouts) FROM benchmark_checkpoint GROUP BY checkpoint ORDER BY checkpoint").fetchall()
        summary_ms = (time.perf_counter() - started) * 1000
        # Keep the aggregate score tied to the source revision that produced
        # the newest result. Older rows may predate commit provenance, so the
        # API returns None instead of attributing them to the page release.
        source_rows = conn.execute("""
            SELECT DISTINCT ON (checkpoint) checkpoint, NULLIF(outcome->>'commit', '')
            FROM rollout_results
            ORDER BY checkpoint, finished_at DESC
        """).fetchall()
        source_commits = {str(row[0]): row[1] for row in source_rows}
        # The page's row count is itself an aggregate: the API never scans the
        # 30 Hz hypertable. `samples` is the exact count represented by each
        # one-second continuous aggregate row.
        raw_count = int(conn.execute("SELECT COALESCE(sum(samples),0) FROM rollout_1s").fetchone()[0] or 0)
        result_count = int(conn.execute("SELECT count(*) FROM rollout_results").fetchone()[0] or 0)
        compression = conn.execute("SELECT COALESCE(sum(before_compression_total_bytes),0), COALESCE(sum(after_compression_total_bytes),0) FROM hypertable_compression_stats('rollout_telemetry')").fetchone()
        hypertable = conn.execute("SELECT compression_enabled FROM timescaledb_information.hypertables WHERE hypertable_name='rollout_telemetry'").fetchone()
        latest_rollout = conn.execute("SELECT max(finished_at) FROM rollout_results").fetchone()[0]
        diagnosis_rows = conn.execute("""SELECT created_at,rollout_id,model,analyst_version,frame_paths,
                   failure_stage,cause,evidence_frame,scene_features,next_scenes
                   FROM failure_diagnoses ORDER BY created_at DESC LIMIT 100""").fetchall()
        keys = ("bucket","rollout_id","seed","checkpoint","q0","q1","q2","q3","q4","q5","gripper","samples","joint_effort","status")
        rows = []
        for row in rollups:
            item = dict(zip(keys, row)); item["bucket"] = item["bucket"].isoformat() if hasattr(item["bucket"], "isoformat") else item["bucket"]; rows.append(item)
        checkpoints = [{"checkpoint": r[0], "bricks": r[1], "correct": r[2], "drops": r[3], "false_accepts": r[4], "false_rejects": r[5], "cycle_time_s": r[6], "rollouts": r[7], "source_commit": source_commits.get(str(r[0]))} for r in summary]
    before, after = [int(x or 0) for x in compression]
    diagnoses = [{"created_at": r[0].isoformat() if hasattr(r[0], "isoformat") else r[0], "rollout_id": r[1], "model": r[2], "version": r[3], "frame_paths": r[4], "failure_stage": r[5], "cause": r[6], "frame": r[7], "scene_features": r[8], "next_scenes": r[9]} for r in diagnosis_rows]
    # Station requests are an optional analyst-session extension. Keep the
    # telemetry dashboard readable against a cloud schema created before that
    # extension; an absent queue means an honest empty station gallery.
    try:
        stations = station_snapshot(url=url)
    except Exception as exc:
        if psycopg is None or not isinstance(exc, psycopg.errors.UndefinedTable):
            raise
        stations = {"stations": [], "worker": None}
    return {"status":"ok","source":"timescale","rollups":rows,"checkpoints":checkpoints,"diagnoses":diagnoses,"stations":stations["stations"],"worker":stations["worker"],"raw_rows":raw_count,"result_rows":result_count,"compression_enabled":bool(hypertable and hypertable[0]),"uncompressed_bytes":before,"compressed_bytes":after,"compression_ratio":round(before/after,3) if after else None,"latest_rollout_at":latest_rollout.isoformat() if hasattr(latest_rollout,"isoformat") else latest_rollout,"query_ms":{"rollup_aggregate":round(aggregate_ms,3),"checkpoint_aggregate":round(summary_ms,3)},"snapshot_at":datetime.now(timezone.utc).isoformat()}

def counts(url: str | None = None) -> dict:
    snap = dashboard_snapshot(url); return {"telemetry_rows":snap["raw_rows"],"result_rows":snap["result_rows"],"aggregate_rows":len(snap["rollups"]),"compression":"enabled"}

def ingest_replay(path: str | Path, url: str | None = None) -> str:
    """Ingest one exported MuJoCo replay into the local writer."""
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as fh: replay = json.load(fh)
    else:
        replay = json.loads(path.read_text().splitlines()[-1])
    base = datetime.now(timezone.utc)
    samples = [{"ts": base.timestamp() + float(f["t"]), "qpos": list(f.get("qpos", [])) + [float(f.get("gripper", 0))], "gripper": float(f.get("gripper", 0))} for f in replay.get("frames", [])]
    events = [{"ts": base.timestamp() + float(e.get("t", 0)), **e} for e in replay.get("events", [])]
    return record_rollout(samples, events, replay.get("outcome", {}), rollout_id=f"replay-seed-{replay.get('seed', 0)}", seed=int(replay.get("seed", 0)), checkpoint=replay.get("checkpoint", "expert"), url=url)

def sync(source_url: str | None = None, target_url: str | None = None) -> dict:
    """Copy complete local rollouts to Tiger Cloud, idempotently by rollout id."""
    _require_db(); source_url = source_url or DEFAULT_URL; target_url = target_url or os.environ.get("DRYRUN_TIGER_DB_URL")
    if not target_url: raise RuntimeError("DRYRUN_TIGER_DB_URL is not set")
    setup(target_url)
    copied = 0
    with psycopg.connect(source_url) as source, psycopg.connect(target_url) as target:
        ids = {r[0] for r in target.execute("SELECT DISTINCT rollout_id FROM rollout_results").fetchall()}
        results = source.execute("SELECT finished_at,rollout_id,seed,checkpoint,bricks,correct,drops,false_accepts,false_rejects,cycle_time_s,failure_reason,outcome FROM rollout_results ORDER BY finished_at").fetchall()
        for result in results:
            rid = result[1]
            if rid in ids: continue
            rows = source.execute("SELECT ts,rollout_id,seed,checkpoint,q0,q1,q2,q3,q4,q5,gripper,event,event_data,outcome FROM rollout_telemetry WHERE rollout_id=%s ORDER BY ts", (rid,)).fetchall()
            rows = [tuple(list(row[:12]) + [Jsonb(row[12]) if row[12] is not None else None, Jsonb(row[13]) if row[13] is not None else None]) for row in rows]
            with target.cursor() as cur:
                cur.executemany("""INSERT INTO rollout_telemetry (ts,rollout_id,seed,checkpoint,q0,q1,q2,q3,q4,q5,gripper,event,event_data,outcome) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows)
                result = tuple(list(result[:-1]) + [Jsonb(result[-1])])
                cur.execute("""INSERT INTO rollout_results (finished_at,rollout_id,seed,checkpoint,bricks,correct,drops,false_accepts,false_rejects,cycle_time_s,failure_reason,outcome) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", result)
            copied += 1; ids.add(rid)
        target.commit()
    refresh(target_url)
    return {"copied_rollouts": copied, "target": "Tiger Cloud", "stats": remote_stats(target_url)}

def remote_stats(url: str | None = None) -> dict:
    """Numbers shown on the page, all read from Tiger's catalog and aggregates."""
    _require_db(); started = time.perf_counter()
    with psycopg.connect(url or db_url()) as conn:
        raw = int(conn.execute("SELECT COALESCE(sum(samples),0) FROM rollout_1s").fetchone()[0] or 0)
        rollouts = int(conn.execute("SELECT count(*) FROM rollout_results").fetchone()[0] or 0)
        aggregates = int(conn.execute("SELECT count(*) FROM rollout_1s").fetchone()[0] or 0)
        checkpoint_rows = int(conn.execute("SELECT count(*) FROM benchmark_checkpoint").fetchone()[0] or 0)
        diagnosis_rows = conn.execute("SELECT count(*) FROM failure_diagnoses").fetchone()[0]
        compression = conn.execute("SELECT COALESCE(sum(before_compression_total_bytes),0), COALESCE(sum(after_compression_total_bytes),0) FROM hypertable_compression_stats('rollout_telemetry')").fetchone()
        hypertable = conn.execute("SELECT compression_enabled FROM timescaledb_information.hypertables WHERE hypertable_name='rollout_telemetry'").fetchone()
    elapsed = (time.perf_counter() - started) * 1000
    uncompressed, compressed = [int(x or 0) for x in compression]
    return {"telemetry_rows": raw, "rollouts": rollouts, "aggregate_rows": aggregates, "checkpoint_rows": checkpoint_rows, "diagnosis_rows": diagnosis_rows, "compression_enabled": bool(hypertable and hypertable[0]), "uncompressed_bytes": uncompressed, "compressed_bytes": compressed, "compression_ratio": round(uncompressed / compressed, 3) if compressed else None, "catalog_query_ms": round(elapsed, 3)}

def write_rollout(seed: int = 7, seconds: float = 2.0, url: str | None = None):
    start = datetime.now(timezone.utc); n = int(seconds * 30)
    samples = [{"ts":start.timestamp()+i/30,"qpos":[.01*i,-.02*i,.03*i,0,0,.2],"gripper":.2} for i in range(n)]
    return record_rollout(samples,[{"name":"start","ts":start}],{"status":"success","bricks":0,"correct":0},seed=seed,url=url), n

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("cmd",choices=["setup","write","counts","refresh","snapshot","ingest","sync","remote-stats"]); p.add_argument("--seed",type=int,default=7); p.add_argument("--seconds",type=float,default=2.0); p.add_argument("--out",type=Path); p.add_argument("--replay",type=Path); a=p.parse_args()
    if a.cmd=="setup": setup(); print(json.dumps({"schema":"ok","db":"configured"}))
    elif a.cmd=="write": print(json.dumps(dict(zip(("rollout_id","rows"),write_rollout(a.seed,a.seconds)))))
    elif a.cmd=="refresh": refresh(); print(json.dumps({"refreshed":True}))
    elif a.cmd=="counts": print(json.dumps(counts()))
    elif a.cmd=="ingest": print(json.dumps({"rollout_id": ingest_replay(a.replay)}))
    elif a.cmd=="sync": print(json.dumps(sync()))
    elif a.cmd=="remote-stats": print(json.dumps(remote_stats()))
    else:
        snap=dashboard_snapshot();
        if a.out: a.out.write_text(json.dumps(snap,indent=2,default=str))
        print(json.dumps(snap,indent=2,default=str))

if __name__=="__main__": main()
