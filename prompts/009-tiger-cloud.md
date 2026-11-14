# Supervisor to the page session, Sat 14.11.2026 19:20 UTC: Tiger Cloud is ready

A free Tiger Cloud service exists: name `dryrun`, service id `w0nfn5a4t8`, region us-east-1, PostgreSQL 18.6, TimescaleDB 2.32.0, toolkit 1.26.0. I tested it: hypertable, continuous aggregate, compression and a compression policy all work. It is empty.

The connection URL (with password) is in hsec as `DRYRUN_TIGER_DB_URL`. Never print it, never write it to a file in the repo or a log.

- Run something against it: `hsec exec --only DRYRUN_TIGER_DB_URL -- bash -c 'DRYRUN_DB_URL=$DRYRUN_TIGER_DB_URL uv run python -m <module>'`
- Give it to the Fly app: `hsec exec --only DRYRUN_TIGER_DB_URL --only FLY_API_TOKEN -- bash -c 'fly secrets set DRYRUN_DB_URL="$DRYRUN_TIGER_DB_URL" -a dryrun-junction-2026'`

What to do:

1. Apply `data/schema.sql` to the cloud service unchanged. The same file must work on the local container and on the cloud.
2. Limits: shared compute and 750 MB of storage. Thirty-hertz joint rows for every rollout will not fit raw for long. Measure bytes per rollout after compression, then decide: compress chunks early (minutes, not days), and keep raw rows for the held-out benchmark runs and the rollouts shown on the page; for the rest keep the aggregates and events. Write the numbers and the choice in `status/M3.md`. The local container keeps everything.
3. Writers on this box write to the local container during runs (fast) and a sync command copies to the cloud: `uv run python -m dryrun.telemetry sync`. It must be safe to run twice.
4. The public page reads only from the cloud service's continuous aggregates. Drop the snapshot mode once that works; keep the "last rollout received at <time>" line so a visitor sees the data is live.
5. The page shows what the database does, in numbers read from it: rows, rollouts, compression ratio (from `hypertable_compression_stats`), aggregate refresh time, query time for the dashboard queries. This is the Tiger Data entry, so these numbers must be real and re-readable.
6. The Tiger CLI is installed and logged in (`~/.local/bin/tiger`, `tiger service list`). Tiger also ships an MCP server (`tiger mcp`); if it helps you inspect the service from Codex, use it and log that in `CODEX.md`.

Do not create more services (two free ones is the cap) and do not change the plan.
