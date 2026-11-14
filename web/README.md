# Dry Run dashboard

Run locally with `PORT=8898 DRYRUN_DB_URL=postgresql://... python web/app.py` and open `/`. `/api/score` and `/api/telemetry` read the Timescale continuous aggregate when configured; without a database they report `unavailable` instead of inventing a score. `/api/defect` records a custom defect request as JSONL for the next rollout. `/api/codex` reads the saved `data/codex_log.jsonl` timeline and reports whether the optional local asciinema cast exists; it never claims a cast that is not on the run volume.

The deployed smoke-test URL is https://dryrun-junction-2026.fly.dev/.
