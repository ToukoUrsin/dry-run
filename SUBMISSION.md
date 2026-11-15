# Submission

Dry Run is an evidence-first MuJoCo reproduction of the Junction SO-101
brick-sorting station. It tests perception, control, failure analysis and
telemetry before a real arm moves; no real-arm performance is claimed. Numbers
come from the status files and `data/page/*.json` artifacts.

## What was built

Five parallel sessions built the seeded station, measured expert, real-frame
inspector, page and telemetry, policy evaluation, and Nemotron analyst. The
reference expert is correct on 203/205 bricks across 40 loose scenes (99.0%).
The fixed one-brick loose tier is withdrawn; current benchmark evidence uses
seeded multi-brick scenes.

## Review

Five real Codex review runs found 28 findings and ten named fix commits;
codex review found why the learned policy scored zero: every action chunk was
one action repeated, paired with the next frame. After the fix it reaches 9 of
20 bricks; the latest policy evaluation reaches 10/20 and grasps 2/20, but
still sorts 0/20 and has no lifts, presentations or drops. See
[CODEX.md](CODEX.md), [status/codex-review-0415.md](status/codex-review-0415.md)
and [data/page/codex.json](data/page/codex.json).

## Final measured numbers

- **Inspector:** on 569 held-out real bricks, 328/328 good bricks were accepted and 241/241 defective bricks were detected, with 0 false rejects and 0 false accepts.
- **Camera-only sorter:** the final diagnostic placed 4/100 bricks in the right bowl (39 reached, 33 gripped, 20 presented); this is not a benchmark.
- **Learned policy:** the latest evaluation reached 10/20 one-brick scenes and grasped 2/20, but sorted 0/20, with 0 lifts, presentations or drops.
- **Balanced Nemotron:** 45/84 strict stage decisions were correct (53.6%) across 21 examples per class. Chance and the largest-class baseline are both 25%; it gets 19/21 lifted-then-lost cases but calls 18/21 touched-without-lift and 19/21 wrong-bowl cases lifted-then-lost.

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
