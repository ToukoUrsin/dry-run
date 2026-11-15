# Submission

Dry Run is submitted as a simulator-first answer to the Intel station task.
The status files linked below contain the commands, outputs and commit for
each claim. Anything still marked open is described as open here as well.

## Intel

The station is a MuJoCo scene built around the official SO-101 model, seeded
Duplo bodies, two bowls and top and inspection cameras. The scripted expert is
allowed simulator state as a demonstrator; policy and inspector interfaces are
vision and joint-state interfaces. The fixed-scene loose tier was withdrawn as
rigged; no score from it is used as a benchmark or training result. Seeded
multi-brick touching and pile evidence, the remaining physics gates, and the
station checks are recorded in [status/M1.md](status/M1.md) and
[status/M5.md](status/M5.md), while inspection evidence is in
[status/M2.md](status/M2.md). This entry does not claim that a real arm has
been tested. [MORNING.md](MORNING.md) is the real-station check.

## Codex

The build used parallel Codex sessions for simulation, inspection, page and
analysis work. The session log is [CODEX.md](CODEX.md); it records the model,
session id, action and proof command for work that actually ran. The public
page, replay and telemetry evidence is linked from [status/M3.md](status/M3.md)
and [status/M6.md](status/M6.md).

## Nemotron

The failure analyst uses NVIDIA's
`nvidia/nemotron-3.5-super-vl-120b-a12b` through
`https://integrate.api.nvidia.com/v1`. It receives camera frames from failed
rollouts, emits validated stage, cause, evidence-frame, scene-feature and
next-scene fields, and stores the request and response. The measured accuracy
and curriculum comparison, including the version/change/accuracy table, are
in [status/M5.md](status/M5.md). The final 448 px run is 131/300 stage
decisions correct (`0.436667`); its 32 px predecessor is explicitly
withdrawn. The same status file proves the 9/24 versus 15/30 measured
curriculum result and the 36-sentence station rollout. A result is presented
as an improvement only when that file shows the command and output.

## Tiger Data

Telemetry is written through the shared `DRYRUN_DB_URL` interface and the
schema in `data/schema.sql`. The page is intended to read aggregates rather
than hand-written values. [status/M3.md](status/M3.md) records the actual
database, row, aggregate, compression and page checks; it also names any
remaining local-only or snapshot limitation.

## Modal

Modal is an optional execution backend for long benchmark, dataset and policy
runs. No spend or remote result is claimed unless a status file records the
exact command, container, duration and estimated cost. Local proof remains
valid and reproducible when Modal credentials are unavailable.
