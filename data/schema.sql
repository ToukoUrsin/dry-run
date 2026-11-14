CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS rollout_telemetry (
  ts TIMESTAMPTZ NOT NULL,
  rollout_id TEXT NOT NULL,
  seed INTEGER NOT NULL,
  checkpoint TEXT NOT NULL DEFAULT 'expert',
  q0 DOUBLE PRECISION NOT NULL,
  q1 DOUBLE PRECISION NOT NULL,
  q2 DOUBLE PRECISION NOT NULL,
  q3 DOUBLE PRECISION NOT NULL,
  q4 DOUBLE PRECISION NOT NULL,
  q5 DOUBLE PRECISION NOT NULL,
  gripper DOUBLE PRECISION NOT NULL,
  event TEXT,
  event_data JSONB,
  outcome JSONB
);
SELECT create_hypertable('rollout_telemetry','ts',if_not_exists => TRUE);
ALTER TABLE rollout_telemetry SET (timescaledb.compress = true, timescaledb.compress_segmentby = 'rollout_id,checkpoint');
DO $$ BEGIN
  PERFORM add_compression_policy('rollout_telemetry', INTERVAL '10 minutes');
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

CREATE MATERIALIZED VIEW IF NOT EXISTS rollout_1s
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 second', ts) AS bucket, rollout_id, seed, checkpoint,
       avg(q0) AS q0, avg(q1) AS q1, avg(q2) AS q2, avg(q3) AS q3,
       avg(q4) AS q4, avg(q5) AS q5, avg(gripper) AS gripper,
       count(*) AS samples,
       sum(abs(q0)+abs(q1)+abs(q2)+abs(q3)+abs(q4)+abs(q5)) AS joint_effort,
       max(outcome->>'status') AS status
FROM rollout_telemetry
GROUP BY bucket, rollout_id, seed, checkpoint;
DO $$ BEGIN
  PERFORM add_continuous_aggregate_policy('rollout_1s', start_offset => INTERVAL '7 days', end_offset => INTERVAL '1 second', schedule_interval => INTERVAL '1 minute');
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS rollout_results (
  finished_at TIMESTAMPTZ NOT NULL,
  rollout_id TEXT NOT NULL,
  seed INTEGER NOT NULL,
  checkpoint TEXT NOT NULL,
  bricks INTEGER NOT NULL,
  correct INTEGER NOT NULL,
  drops INTEGER NOT NULL DEFAULT 0,
  false_accepts INTEGER NOT NULL DEFAULT 0,
  false_rejects INTEGER NOT NULL DEFAULT 0,
  cycle_time_s DOUBLE PRECISION,
  failure_reason TEXT,
  outcome JSONB NOT NULL,
  PRIMARY KEY (finished_at, rollout_id)
);
SELECT create_hypertable('rollout_results','finished_at',if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS rollout_results_checkpoint_idx ON rollout_results (checkpoint, finished_at DESC);

CREATE MATERIALIZED VIEW IF NOT EXISTS benchmark_checkpoint
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', finished_at) AS bucket, checkpoint,
       sum(bricks) AS bricks, sum(correct) AS correct, sum(drops) AS drops,
       sum(false_accepts) AS false_accepts, sum(false_rejects) AS false_rejects,
       avg(cycle_time_s) AS cycle_time_s, count(*) AS rollouts
FROM rollout_results GROUP BY bucket, checkpoint;
DO $$ BEGIN
  PERFORM add_continuous_aggregate_policy('benchmark_checkpoint', start_offset => INTERVAL '30 days', end_offset => INTERVAL '1 minute', schedule_interval => INTERVAL '1 minute');
EXCEPTION WHEN OTHERS THEN NULL;
END $$;
