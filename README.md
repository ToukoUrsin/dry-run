# Dry Run

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101
brick-sorting station. It is a place to test perception, control, failure
analysis and telemetry before a real arm moves. The simulator is the product
being demonstrated; it does not make a claim about a physical arm.

The status files and `data/page/*.json` are the source of truth for results. A
statement belongs in the submission only when its status file has an exact
command and output that can be rerun. Open milestones are deliberately called
open.

The live page is https://dryrun-junction-2026.fly.dev/. Its station, failure,
inspector, analyst, and Codex views are simulator evidence, and the learned
policy gate remains open because the current ACT maps reach no physical stage.
The Nemotron curriculum comparison is measured-expert tuning evidence rather
than a learned-policy result.

## What is in the repository

- `assets/SO101/` contains the official SO-101 model, meshes and licence.
- `dryrun/mjcf_station.py` builds seeded stations with Duplo geometry, bowls,
  cameras and defect geometry.
- `dryrun/render.py` and `dryrun/benchmark.py` run the MuJoCo expert and
  record physical events and scores.
- `dryrun/inspector*.py` and `dryrun/defects.py` contain the good-only
  inspection experiments and held-out evaluations. The shipped PatchCore is
  adapted from real good present frames; the synthetic-render-only model was
  rejected on real crops, so the exact Anomalib synthetic-training gate remains
  open (see `status/M2.md`).
- `dryrun/telemetry.py`, `data/schema.sql` and `web/` provide the telemetry
  writer and dashboard.
- `dryrun/nemotron*.py`, `dryrun/curriculum*.py`, `dryrun/stations*.py` and
  `dryrun/worker.py` contain the failure analyst, curriculum comparison,
  sentence-to-station machinery and bounded local worker. Their proof is in
  `status/M5.md`; `data/codex_log.jsonl` is the page-readable build ledger.

The fixed-scene loose tier was withdrawn after review because it repeated one
brick at one position for every seed. Its historical scores are retained only
as failure evidence and are not benchmark or training results. Current page
evidence uses seeded multi-brick scenes; the measured touching and pile rows
are recorded in `data/page/codex.json` and `status/M5.md`.

The measured grasp recipe and cycle timings were supplied as prompt artifacts
in `prompts/019-grasp-recipe-measured.md` and `prompts/021-reference/`.
Codex ported and integrated that recipe into the station, rollout API, scoring
and replay paths; the station and analyst artifacts document which runs are
diagnostic or benchmark evidence.

## Reproduce the checks

Run these from the repository. Keep new videos, checkpoints and other large
outputs under `/bulk3/runs/dryrun/`; historical proofs may reference archived
paths. The live page is https://dryrun-junction-2026.fly.dev/.

```bash
uv run python -c 'import mujoco; print(mujoco.__version__)'
uv run python -m dryrun.mjcf_station check assets/station.xml
uv run python -m dryrun.mjcf_station batch --seeds 32
MUJOCO_GL=egl uv run python -m dryrun.render --seed 7 --out /bulk3/runs/dryrun/m1/seed7.mp4
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m dryrun.benchmark --seeds 50 \
  --tier pile --workers 4 --without-inspector \
  --output /bulk3/runs/dryrun/benchmark/readme-pile50.json
```

For the Nemotron run, use the secret without printing it:

```bash
hsec exec --only NVIDIA_API_KEY -- uv run python -m dryrun.nemotron --help
```

The completed 448 px diagnostic score and its confusion table are in
`/bulk3/runs/dryrun/nemotron_prompt024/score-final-merged/score.json`; the
measured curriculum comparison and 36-sentence station rollout are proven in
`status/M5.md`.

The local page uses `DRYRUN_DB_URL` and port `8898`. The command output and
the linked status files are the source of benchmark numbers; the page must
show an explicit empty or snapshot state when a live database result is
unavailable.

The Codex evidence page records the four blocked historical review attempts and
the fresh review's zero actionable findings in `status/codex-review-0415.md`; its
one-run Fast comparison is in `status/fast-mode.md`.

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
