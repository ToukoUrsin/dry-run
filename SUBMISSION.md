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

The current public Tiger-backed score is 1090/1262 (`0.8637083994`) for the
`b9d897c` pushing expert. The first-screen tier card still mounts the
supervisor-verified `64c4919` reference row; the API also exposes the newer
loose, touching and pile rows. These remain simulator measurements.

The shipped inspector is a PatchCore model adapted from 588 real good present
frames. The synthetic-render-only model rejected all 294 good real crops, so
the exact Anomalib good-render gate in the goal remains open; see [status/M2.md](status/M2.md).

## Codex

The build used parallel Codex sessions for simulation, inspection, page and
analysis work. The session log is [CODEX.md](CODEX.md); it records the model,
session id, action and proof command for work that actually ran. The public
page, replay and telemetry evidence is linked from [status/M3.md](status/M3.md)
and [status/M6.md](status/M6.md).

The measured reference recipe was supplied in prompt files and ported by
Codex; no external implementation is claimed.

## Nemotron

The failure analyst uses NVIDIA's
`nvidia/nemotron-3.5-super-vl-120b-a12b` through
`https://integrate.api.nvidia.com/v1`. It receives camera frames from failed
rollouts and stores each request and response. The earlier v4 diagnostic used
validated stage, cause, evidence-frame, scene-feature and next-scene fields.
The current Prompt 024 run uses 448 px sheets and a strict four-field schema
(stage, visible support, scene change and evidence panel); it got 131/300 stage
decisions correct (`0.436667`). Its 32 px predecessor is explicitly withdrawn.
The measured accuracy, schema/version table and curriculum comparison are in
[status/M5.md](status/M5.md). The 9/24 versus 15/30 comparison is measured
expert tuning (`helps=false`), not a learned-policy improvement; no such
learned-policy comparison is claimed while the policy reaches no physical
stage.

## Tiger Data

Telemetry is written through the shared `DRYRUN_DB_URL` interface and the
schema in `data/schema.sql`. The page reads aggregate views rather than
hand-written values. The current direct audit reported 20,723 raw rows, 401
result rows and compression ratio 2.2; the dashboard caps returned diagnoses
at 100 while the database retains 304. [status/M3.md](status/M3.md) records
the commands and the older snapshot limitations.

## Modal

Modal is an optional execution backend for long benchmark, dataset and policy
runs. No spend or remote result is claimed unless a status file records the
exact command, container, duration and estimated cost. Local proof remains
valid and reproducible when Modal credentials are unavailable.
