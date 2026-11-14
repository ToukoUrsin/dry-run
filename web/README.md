# Dry Run dashboard

Run `python web/app.py` and open http://localhost:8787. The zero-dependency server exposes `/api/score`, `/api/rollouts`, and `/api/defect`; set `PORT` to change the port. The dashboard reads benchmark numbers from `data/benchmark.json` and keeps custom defects as JSONL.
