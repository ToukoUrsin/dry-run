# Submission

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101 brick-sorting station. It tests perception, control, failure analysis and telemetry before a real arm moves; it makes no real-arm performance claim. The status files and `data/page/*.json` are the source of every number.

## Codex

The Codex page snapshot records 274 attributed commits, five sessions, 9.23 elapsed goal-mode hours, 11,667 tracked code lines and 45 prompt files. The five streams owned simulation/expert work, inspection, page/telemetry, policy, and Nemotron/stations; their models, evidence and session paths are in [CODEX.md](CODEX.md) and [data/page/codex.json](data/page/codex.json). The exact boundary is stated there: the supervisor wrote prompts and Codex ported the two reference sorters.

## Nemotron

NVIDIA Nemotron (`nvidia/nemotron-3.5-super-vl-120b-a12b`) receives camera-only failure evidence and returns a stage, evidence sentence, scene change and selected panel. The 448 px sheet run is 131/300 stage decisions correct (`43.6667%`, 44% rounded) in [data/page/analyst.json](data/page/analyst.json); the earlier 32 px run scored 9% and is withdrawn because the brick was only about two pixels wide. The larger sheets, four key frames and strict schema are the change. Nemotron diagnoses failures and proposes scenes; it is not a learned policy and its 9/24 versus 15/30 curriculum comparison is measured-expert tuning only.

## Tiger Data

Tiger Cloud stores raw telemetry in a Timescale hypertable with compression and serves continuous aggregates (`rollout_1s` and `benchmark_checkpoint`). The page reads those aggregates rather than raw rows or hand-written scores. The current page data records 17,439 raw telemetry rows, 357 rollout outcomes, 16,334 aggregate rows, three checkpoint rows and compression ratio 2.2; the source and refresh command are in [data/page/tiger.json](data/page/tiger.json). Diagnosis and station views retain recorded evidence and explicit offline states when a live table is unavailable.

## Intel

The simulated station uses the official SO-101 model, seeded Duplo bodies, bowls and top/inspection cameras. The OpenVINO PatchCore inspector trains on 588 real good present frames and evaluates 569 real bricks: 328 good and 241 defective, with 0% false rejects and 0% false accepts; four-thread inference is 216.9 ms median per pair and one-thread inference is 471.8 ms. These figures come from [data/page/inspector.json](data/page/inspector.json). The fixed-scene loose tier is withdrawn; current benchmark evidence uses seeded multi-brick scenes.
