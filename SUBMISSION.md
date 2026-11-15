# Submission

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101
brick-sorting station. It tests perception, control, failure analysis and
telemetry before a real arm moves; no real-arm performance is claimed. Numbers
come from the status files and `data/page/*.json` artifacts.

## What was built

Five parallel sessions built the seeded station, measured expert, real-frame
inspector, page and telemetry, policy evaluation, and Nemotron analyst. The
Reference/pushing expert: **380/382 loose (99.5%)**, **451/494 touching
(91.3%)**, and **259/386 pile (67.1%)**.
The fixed one-brick loose tier is withdrawn; current benchmark evidence uses
seeded multi-brick scenes.

## Review

Five real Codex review runs found 28 findings and ten named fix commits;
codex review found why the learned policy scored zero: every action chunk was
one action repeated, paired with the next frame. After the fix it reaches 9 of
policy reaches **5/20** one-brick scenes and still sorts **0/20**; the earlier
canonical result sorted **0/1,309 bricks** across 200 held-out scenes. See
[CODEX.md](CODEX.md), [status/codex-review-0415.md](status/codex-review-0415.md)
and [data/page/codex.json](data/page/codex.json).

## Final measured numbers

- **Inspector:** on 569 held-out real bricks, 328/328 good bricks were accepted and 241/241 defective bricks were detected, with 0 false rejects and 0 false accepts.
- **Camera-only sorter:** the final diagnostic placed 4/100 bricks in the right bowl (39 reached, 33 gripped, 20 presented); this is not a benchmark.
- **Learned policy:** after the fix, the policy reaches **5/20** one-brick scenes and still sorts **0/20**; the earlier canonical result sorted **0/1,309 bricks** across 200 held-out scenes.
- **Nemotron diagnostic:** the valid 448 px run got 131/300 strict stage decisions correct (43.7%). Since 295/300 failures are `no_clear_grasp`, always answering `no_clear_grasp` scores 98.3%, above Nemotron; the balanced four-stage test is withdrawn because 63/84 sheets were too small to judge.

Nemotron diagnoses camera evidence and proposes scenes; it is not a learned
policy. No real-arm run has been performed. Full commands, confusion matrices,
and source commits are linked from [status/M5.md](status/M5.md),
[data/page/analyst.json](data/page/analyst.json),
[data/page/policy.json](data/page/policy.json), and
[data/page/inspector.json](data/page/inspector.json).

## Runtime and telemetry

The page reads recorded simulator evidence and Tiger aggregates, preserving
explicit offline states when a live table is unavailable. The measured Fast
comparison was 28.53 seconds versus 22.42 seconds with identical output.
