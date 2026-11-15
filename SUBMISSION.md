# Submission

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101 brick-sorting station. It tests perception, control, failure analysis and telemetry before a real arm moves; it makes no real-arm performance claim. The status files and `data/page/*.json` are the source of every number.

## Codex

The Codex page records over 300 attributed commits, five sessions, the wall-clock night duration and the summed session hours; its generated exact counts are in [data/page/codex.json](data/page/codex.json). The five streams owned simulation/expert work, inspection, page/telemetry, policy, and Nemotron/stations; their exact session IDs and evidence are in [CODEX.md](CODEX.md). The fresh review completed with zero actionable findings after four earlier blocked attempts, and the one-run Fast comparison was 28.53 s versus 22.42 s with identical output. The authorship boundary is stated there.

## Nemotron

NVIDIA Nemotron (`nvidia/nemotron-3.5-super-vl-120b-a12b`) receives camera-only failure evidence and returns a stage, evidence sentence, scene change and selected panel. The 448 px sheet run is 131/300 stage decisions correct (`43.6667%`, 44% rounded) in [data/page/analyst.json](data/page/analyst.json). The per-stage table shows that 295/300 failures (`98.3333%`) are `no_clear_grasp`, so a constant guess scores 98.3333%; Nemotron is below it at 43.6667%. The earlier 32 px run scored 9% and is withdrawn because the brick was only about two pixels wide. Nemotron diagnoses failures and proposes scenes; it is not a learned policy and its 9/24 versus 15/30 curriculum comparison is measured-expert tuning only.

## Tiger Data

Tiger Cloud stores raw telemetry in a Timescale hypertable with compression and serves continuous aggregates (`rollout_1s`, `benchmark_checkpoint`, tier statistics, failure events and totals). The page reads those aggregates rather than raw rows or hand-written scores. The current page data records 580,704 measured 30 Hz rows, 401 rollout outcomes (400 held-out plus one smoke row), 19,729 aggregate rows, three checkpoint rows and compression ratio 2.176; the source and refresh command are in [data/page/tiger.json](data/page/tiger.json). Diagnosis and station views retain recorded evidence and explicit offline states when a live table is unavailable.

## Intel

The simulated station uses the official SO-101 model, seeded Duplo bodies, bowls and top/inspection cameras. The OpenVINO PatchCore inspector trains on 588 real good present frames and evaluates 569 real bricks: 328 good and 241 defective, with 0% false rejects and 0% false accepts. Inference is 216.9 ms median per camera frame at four threads (about 0.4 s per brick, two frames) and 471.8 ms per camera frame at one thread (about 0.94 s per brick). These figures come from [data/page/inspector.json](data/page/inspector.json). The fixed-scene loose tier is withdrawn; current benchmark evidence uses seeded multi-brick scenes.
