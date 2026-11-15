# Dry Run

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101
brick-sorting station. It is a place to test perception, control, failure
analysis and telemetry before a real arm moves. The simulator is the product
being demonstrated; it does not make a claim about a physical arm.

The status files are the source of truth for results. A statement belongs in
the submission only when its status file has an exact command and output that
can be rerun. Open milestones are deliberately called open.

The public deployment was checked directly on 2026-11-15 at 02:45 UTC. It was
healthy at commit `bbf30f4`; Tiger-backed `/api/score` reported the current
`expert-v2-b9d897c` checkpoint at 1090/1262 bricks (`0.8637083994`), and the
tier endpoint exposed both the supervisor reference rows and the newer pushing
expert rows. These are simulator results. The learned-policy gate remains open:
the current ACT maps reach no physical stage, and the Nemotron curriculum
comparison is expert tuning evidence rather than a learned-policy result.

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
outputs under `/bulk2/runs/dryrun/` as required by `AGENTS.md`; historical
proofs may reference the archived `/bulk3/runs/dryrun/` paths.

```bash
uv run python -c 'import mujoco; print(mujoco.__version__)'
uv run python -m dryrun.mjcf_station check assets/station.xml
uv run python -m dryrun.mjcf_station batch --seeds 32
MUJOCO_GL=egl uv run python -m dryrun.render --seed 7 --out /bulk2/runs/dryrun/m1/seed7.mp4
# The withdrawn loose benchmark is intentionally not a current score source;
# use the recorded seeded multi-brick runs in status/M1.md and status/M5.md.
```

For the Nemotron run, use the secret without printing it:

```bash
hsec exec --only NVIDIA_API_KEY -- uv run python -m dryrun.nemotron --help
```

The completed 448 px diagnostic score and its confusion table are in
`/bulk3/runs/dryrun/nemotron_prompt024/score-final-merged/score.json`; the
measured curriculum comparison and 36-sentence station rollout are proven in
`status/M5.md`.

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
