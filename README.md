# Dry Run

Dry Run is a seedable CPU simulator for the Junction 2026 brick-sorting station. It generates camera-facing scenes, runs a scripted demonstrator, and scores a fixed benchmark.

```bash
python -m dryrun.sim scene 42 --json
python benchmark.py --seeds 200 --output runs/benchmark.json
python -m dryrun.render_run 7
```

The XML station is intentionally small and portable. It provides the same table, bowls and camera contract used by later inspector and policy stages; it does not claim a calibrated physical SO-101.
