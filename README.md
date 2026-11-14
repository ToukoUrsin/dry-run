# Dry Run

Dry Run is an evidence-first simulator for the Junction SO-101 brick-sorting station. The repository now contains a vendored official SO-101 model, a real MuJoCo station generator, contact-physics rendering, an Anomalib/OpenVINO inspector experiment, Timescale telemetry plumbing, a LeRobot-format dataset experiment, a Nemotron comparison, and a public dashboard.

The project is still unfinished. `status/M1.md` through `status/M6.md` are authoritative; each says open where its proof gate is not met. The current physics expert scores zero on the 200-seed pile benchmark, so no learned-policy or Nemotron uplift claim is treated as a finished result.

## Reproducible checks

```bash
uv run python -c 'import mujoco; print(mujoco.__version__)'
uv run python -m dryrun.mjcf_station check assets/station.xml
uv run python -m dryrun.mjcf_station batch --seeds 32
MUJOCO_GL=egl uv run python -m dryrun.render --seed 7 --out /bulk2/runs/dryrun/m1/seed7.mp4
MUJOCO_GL=egl uv run python -m dryrun.benchmark --seeds 200
```

The local page is `DRYRUN_DB_URL=... PORT=8898 python web/app.py`. The deployed smoke-test page is https://dryrun-junction-2026.fly.dev/; its public score remains unavailable until a remote Timescale URL is configured. Generated videos, model checkpoints and datasets live under `/bulk2/runs/dryrun/`, not in git.
