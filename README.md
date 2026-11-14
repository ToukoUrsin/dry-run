# Dry Run

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101
brick-sorting station. It is a place to test perception, control, failure
analysis and telemetry before a real arm moves. The simulator is the product
being demonstrated; it does not make a claim about a physical arm.

The status files are the source of truth for results. A statement belongs in
the submission only when its status file has an exact command and output that
can be rerun. Open milestones are deliberately called open.

## What is in the repository

- `assets/SO101/` contains the official SO-101 model, meshes and licence.
- `dryrun/mjcf_station.py` builds seeded stations with Duplo geometry, bowls,
  cameras and defect geometry.
- `dryrun/render.py` and `dryrun/benchmark.py` run the MuJoCo expert and
  record physical events and scores.
- `dryrun/inspector*.py` and `dryrun/defects.py` contain the good-only
  inspection experiment and its held-out evaluation.
- `dryrun/telemetry.py`, `data/schema.sql` and `web/` provide the telemetry
  writer and dashboard.
- `dryrun/nemotron*.py`, `dryrun/curriculum*.py` and `dryrun/stations*.py`
  contain the failure analyst, curriculum comparison and sentence-to-station
  machinery. Their proof is in `status/M5.md`.

## Reproduce the checks

Run these from the repository. Keep videos, checkpoints and other large
outputs under `/bulk2/runs/dryrun/`; the repository keeps source and small
text records.

```bash
uv run python -c 'import mujoco; print(mujoco.__version__)'
uv run python -m dryrun.mjcf_station check assets/station.xml
uv run python -m dryrun.mjcf_station batch --seeds 32
MUJOCO_GL=egl uv run python -m dryrun.render --seed 7 --out /bulk2/runs/dryrun/m1/seed7.mp4
python -m dryrun benchmark --seeds 200
```

For the Nemotron run, use the secret without printing it:

```bash
hsec exec --only NVIDIA_API_KEY -- uv run python -m dryrun.nemotron --help
```

The local page uses `DRYRUN_DB_URL` and port `8898`. The public deployment is
`https://dryrun-junction-2026.fly.dev/`; it must show an explicit empty or
snapshot state when a live database result is unavailable.

## Milestone evidence

| Milestone | State and proof |
| --- | --- |
| M1 station and expert | [status/M1.md](status/M1.md) |
| M2 inspector | [status/M2.md](status/M2.md) |
| M3 telemetry and page | [status/M3.md](status/M3.md) |
| M4 learned policy | [status/M4.md](status/M4.md) |
| M5 Nemotron and stations | [status/M5.md](status/M5.md) |
| M6 polish and exports | [status/M6.md](status/M6.md) |

The command and commit recorded in each status file are authoritative. The
dashboard and these documents must not substitute a placeholder number for a
missing run.
