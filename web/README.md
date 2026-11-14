# Dry Run dashboard

Run locally with `PORT=8898 DRYRUN_DB_URL=postgresql://... python web/app.py` and open `/`. `/api/score` and `/api/telemetry` read the Timescale continuous aggregate when configured; without a database they report `unavailable` instead of inventing a score. `/api/defect` records a custom defect request as JSONL for the next rollout.

The deployed smoke-test URL is https://dryrun-junction-2026.fly.dev/.
