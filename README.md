# Dry Run

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101
brick-sorting station. Five parallel sessions built the seeded station,
measured expert, real-frame inspector, page and telemetry, policy evaluation,
and Nemotron failure analyst before a real arm moves. The simulator is the
product being demonstrated; no real-arm performance is claimed.

The live page is https://dryrun-junction-2026.fly.dev/. Station opens a
recorded replay, Scenes and Compare expose seeded outcomes, Build a station
runs the checked sentence worker, and Codex shows the five-session timeline.
The fixed one-brick loose tier was withdrawn; current benchmark evidence uses
seeded multi-brick scenes.

## Final numbers

- Reference/pushing expert: **380/382 loose (99.5%)**, **451/494 touching (91.3%)**, and **259/386 pile (67.1%)**.
- The real-crop inspector accepts **328/328 good** bricks and detects **241/241 defective** bricks, with 0 false rejects and 0 false accepts.
- The final camera-only diagnostic puts **4/100** bricks in the right bowl (39 reached, 33 gripped, 20 presented); it is not a benchmark.
- After the fix, the learned policy reaches **5/20** one-brick scenes and still sorts **0/20**; the earlier canonical result sorted **0/1,309 bricks** across 200 held-out scenes.
- The valid 448 px Nemotron diagnostic gets **131/300 (43.7%)**. Since 295/300 failures are `no_clear_grasp`, always answering `no_clear_grasp` scores **98.3%**, above Nemotron.
- Five real Codex reviews found **28 findings** and mapped them to **10 fixes**. codex review found why the learned policy scored zero: every action chunk was one action repeated, paired with the next frame. After the fix, the learned policy reaches **5/20** one-brick scenes and still sorts **0/20**; the earlier canonical result sorted **0/1,309 bricks** across 200 held-out scenes.

The source of each number is the corresponding `data/page/*.json` file and
its status record. The balanced four-stage test is withdrawn: 63 of its 84
sheets were rendered too small to judge. The valid headline uses only the 448 px
Prompt 024 sheets; the 32 px run is withdrawn because the brick was about two
pixels wide.

## Run locally in three commands

From the repository root:

```bash
# 1. Check the seeded station and closed-form IK.
uv run python -m dryrun.mjcf_station check assets/station.xml

# 2. Run one honest seeded multi-brick benchmark; keep output outside git.
MUJOCO_GL=egl PYTHONPATH=. uv run python -m dryrun.benchmark --seeds 50 \
  --tier pile --workers 4 --without-inspector \
  --output /bulk3/runs/dryrun/benchmark/readme-pile50.json

# 3. Start the local page, then open http://127.0.0.1:8898/.
PORT=8898 PYTHONPATH=. uv run python web/app.py
```

Large videos, images, manifests and checkpoints belong under
`/bulk3/runs/dryrun/`; JSON records and status notes remain in the repository.
The page uses `DRYRUN_DB_URL` when telemetry is available and shows an explicit
recorded or offline state when it is not.

## Evidence map

- Station, expert and benchmark: [status/M1.md](status/M1.md)
- Inspector and camera-only diagnostics: [status/M2.md](status/M2.md) and [data/page/inspector.json](data/page/inspector.json)
- Page and telemetry: [status/M3.md](status/M3.md) and [status/M6.md](status/M6.md)
- Learned policy: [status/M4.md](status/M4.md) and [data/page/policy.json](data/page/policy.json)
- Nemotron and sentence stations: [status/M5.md](status/M5.md) and [data/page/analyst.json](data/page/analyst.json)
- Five-session record and review findings: [CODEX.md](CODEX.md) and [data/page/codex.json](data/page/codex.json)
