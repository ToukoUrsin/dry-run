CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE TABLE IF NOT EXISTS rollout_telemetry (
  ts TIMESTAMPTZ NOT NULL,
  rollout_id TEXT NOT NULL,
  seed INTEGER NOT NULL,
  joint JSONB NOT NULL,
  event TEXT,
  outcome JSONB
);
SELECT create_hypertable('rollout_telemetry','ts',if_not_exists => TRUE);
ALTER TABLE rollout_telemetry SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'rollout_id');
DO $$ BEGIN
  PERFORM add_compression_policy('rollout_telemetry', INTERVAL '1 day');
EXCEPTION WHEN OTHERS THEN NULL;
END $$;
CREATE MATERIALIZED VIEW IF NOT EXISTS rollout_1s
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 second', ts) AS bucket, rollout_id, seed,
       avg((joint->>'q0')::double precision) AS q0,
       avg((joint->>'q1')::double precision) AS q1,
       count(*) AS samples,
       max(outcome->>'status') AS status
FROM rollout_telemetry GROUP BY bucket, rollout_id, seed;
DO $$ BEGIN
  PERFORM add_continuous_aggregate_policy('rollout_1s', start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '1 minute');
EXCEPTION WHEN OTHERS THEN NULL;
END $$;
