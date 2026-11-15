# Shared session contracts

The main session owns station generation, the MuJoCo expert, the benchmark, demos, policy and Nemotron loop. The page session owns `web/`, replay export, telemetry and Fly. The inspector session owns defect geometry, inspection renders, Anomalib and OpenVINO.

`dryrun.mjcf_station.station_xml(seed, sentence, count=None)` returns a complete MJCF string. `layout_bricks(seed, sentence, count=None)` returns ordered `BrickSpec` labels; body `brick_N` corresponds to index `N`. A consumer must load the returned XML in MuJoCo and call `mj_forward` before reading poses.

`dryrun.render.rollout(model, data, seed, writer=None, specs=None)` steps the real physics from the supplied simulator state. It returns `{seed, bricks, successes, score, events}`; each event records physical contact pickup, pad contacts, final pose, bowl membership and success. It never teleports or welds a body.

The benchmark calls the inspector at the point where a brick is held at the inspection camera. The inspector returns a bowl decision and timing; the benchmark records that decision in the event. The rollout calls the telemetry writer after each physics step or at its 30 Hz sampling boundary. If an integration is not wired, the owning status file remains open and names the missing call.

## Inspector (inspector session)

`dryrun.inspector.Inspector(model_path=..., threshold=None, threads=1)` loads
the OpenVINO model once. `inspect(frame)` (also `__call__`) accepts an RGB
`uint8` or float `H x W x 3` NumPy frame and returns
`{"score": float, "threshold": float, "verdict": "good"|"defective", "heatmap": uint8[H,W]}`.
The heat map is resized to the input frame; callers do not need ground truth.
The benchmark calls this at `present` and records the verdict before `drop`.

`dryrun.defects.render_drawing_frame(drawing, seed=0, size="2x2", color="red")`
accepts a PNG path or an `H x W x 3/4` NumPy drawing (RGBA alpha is preserved),
places it on the visible face of a seeded brick at the inspection point, and
returns the RGB `uint8[384,384,3]` frame to pass to `Inspector`.

The rebuilt inspector shot uses the prompt-020 brick centre
`INSPECTION_PRESENT_POSE = [0.1715, -0.1009, 0.1300]` metres and
`INSPECTION_CAMERA_POSITION = [0.1194, -0.1430, 0.2296]` metres: a fixed
close camera with `xyaxes="0.6290 -0.7774 0 0.6449 0.5218 0.5584"`, 40 degree
field of view, and a 384x384 RGB frame. `render_inspection` and
`render_inspection_frame` use the same camera
configuration; the rollout calls the latter after holding the brick at that
pose. Both functions use a fixed brick crop: arm and bowl geometry remains in
the physical rollout and top/free cameras, but is excluded from this crop so
the judged top face and its defects cannot be occluded. Underside marks are
outside this shot and are reported separately rather than silently treated as
visible.

`dryrun.live_defect.inspect_custom_defect(payload)` is the fast Modal boundary.
`payload` has a PNG `drawing` data URI and optional `seed`, `size`, and `color`;
it returns `rendered_rgb` (`uint8[H,W,3]`), `heatmap` (`uint8[H,W]`),
`score`, `threshold`, `verdict` (`good` or `defective`), and measured
`latency_ms`. The process caches the OpenVINO graph. `inspection_json(...)`
converts the two arrays to PNG data URIs for HTTP. The Modal endpoint returns
this inspection result by default; adding `rollout: true` also returns the
compressed one-brick replay and its `run_id`.

The page can render the measured inspector artifacts directly from
`/bulk2/runs/dryrun/m2/evaluation.json` (confusion, per-defect recall, held-out
types, OpenVINO version and one/four-thread latency) and
`/bulk3/runs/dryrun/m2/contact_sheet_real_station.png`. A present event carries
the scalar `score`, selected `threshold` and the heat-map array; callers that
persist it should write a PNG beside the replay and retain its path in the
event row. The pure `render_drawing_frame` function is the payload boundary for
an on-demand worker or Modal endpoint.

The page/API inspector contract is `schema: "dryrun.inspector.v1"` from
`GET /api/inspector` (also `/api/inspector.json`) and
`/bulk2/runs/dryrun/m2/inspector_page.json`. It contains `confusion` with
`tp`, `tn`, `fp`, `fn` and false-accept/reject rates,
`per_defect_recall` keyed by each generated defect type (including the two
held-out types), `latency` keyed by OpenVINO thread count with `median_ms`,
`p95_ms`, and `n`, and `score_histograms` with shared `edges`, lower-bound
`bins`, all-sample `counts`, and `per_type` counts. The detailed evaluation
remains at `evaluation.json`; the page projection is emitted beside it and
never invents a histogram when the detailed score rows are unavailable.

## Camera-only expert (prompt 032)

`dryrun.perception` is the camera boundary. `top_camera_calibration` reads the
compiled `top_view` camera pose, axes and field of view; `detect_bricks` accepts
only an RGB top-camera image plus calibration and returns a centre, yaw, size,
colour and `ok`/`unclear` status. It uses a 960 px calibration (the live
renderer must provide at least that resolution), corrects brick-top parallax,
masks calibrated bowl discs, and can receive an arm mask projected from the
six measured joints. The evaluator may join detections to `BrickSpec` for
error measurement; `CameraExpert` never does.

`dryrun.camera_expert.CameraExpert` subclasses `MeasuredExpert` but does not
call its truth-reading constructor. Its controller input is `top`,
`inspection`, and six joint readings. It keeps estimated detections, uses the
measured arm kinematics and bowl calibration, and routes an absent inspector
verdict to the defective bowl. Camera event brick ids are opaque pick tokens;
the rollout wrapper may map them to a nearest true body for scoring logs only.
The sorter itself does not read `data.xpos`, `data.xmat`, contacts, `env.specs`,
or defect labels.

The page-owned inspector artifact is `data/page/inspector.json`. It carries
`updated_utc`, the measurement `commit`, a stranger-facing `summary`, model
and training provenance, real present-frame confusion, held-out recall rows,
`not_visible` accounting, CPU latency when measured, eight compressed examples,
and the `modes` rows for ground-truth, ground-truth-pose plus inspector, and
camera-only runs. The page must omit a section whose JSON file is absent.
End-to-end mode rows may add `tier`, `scenes`, `bricks`, `correct`, `verdicts`,
`pushes`, `false_accepts`, `false_rejects`, `score`, and a stranger-facing
`summary`; a camera-only sweep records `stage_totals`, `timeout_seeds`, and
its artifact path. The optional `mark_sweep` projection has `sizes_mm`,
`contrasts`, 40-frame rows with `kind`, `size_mm`, `contrast`, `recall`, and
the separate geometry-only `visibility_accounting`. `speed` records measured
one/four-thread frame latency and the two-frame cycle context.

## Replay export format

`uv run python -m dryrun.replay_export --seed N --out DIR` writes a compressed `replay.json.gz` plus MuJoCo camera videos. The JSON contains `scene.bodies` (parent, primitive or mesh geoms, sizes, and RGBA), `scene.cameras`, `scene.meshes`, 30 Hz `frames`, `events`, and `outcome`. A frame pose is `[x, y, z, qw, qx, qy, qz]`; positions use `1e-4 m` quantisation and quaternion components use signed int16 scale (`32767`). The pose clock in `frame.t` is authoritative. Videos carry `fps=10`, `source_pose_hz=30`, and `frame_stride=3` because each camera frame is rendered from every third pose.

Event names are `approach`, `grasp`, `lift`, `at_inspection`, `present`, `verdict`, `release`, `drop`, `in_bowl`, `dropped`, and `failed`. Failure events include a `reason`; brick events include `brick` and the event frame.

## Failure analyst and sentence stations (analyst)

`dryrun.nemotron` selects six to eight local frames around a failed rollout,
including both `top` and `inspection` cameras. The request contains those
images only; event-log labels and scene truth are used later for scoring. A
validated response has `failure_stage`, one-sentence `cause`, a supplied
`frame`, `scene_features`, and `next_scenes`. Each next scene has generator
fields `tier`, `brick_count`, `spacing`, `yaw_spread`, `positions`, and
`lighting`. Requests, raw responses, retries and final responses are saved
under `/bulk3/runs/dryrun/m5/diagnoses/`.

`dryrun.telemetry.record_diagnosis(row)` writes the validated result to
`failure_diagnoses`; `dashboard_snapshot()` and `/api/failures` expose the
frame paths, sentence, scene features and next-scene batch to the page. The
row is keyed by `(rollout_id, analyst_version)` so retries do not duplicate a
diagnosis.

`dryrun.stations.generate_and_check(...)` turns each plain sentence into MJCF,
keeps every XML and rejection reason in its JSONL record, and optionally sends
accepted stations through the public `dryrun.rollout.run` expert API. The
worker passes the checked XML as `station_path` together with the sentence, so
the expert runs exactly the station that passed the load, intersection, IK and
camera checks rather than regenerating a tier scene.

The page's `POST /api/stations` accepts a sentence of at most 300 characters
and inserts `station_requests`; it never treats the sentence as a command.
`dryrun.worker --once` claims one pending request, writes the station gallery
record and expert result, and updates `worker_heartbeat`. The page reads these
rows through `GET /api/stations` and labels the worker online or offline.

For the workspace views, the page consumes `diagnoses`, `stations`, and
`worker` from the aggregate snapshot. Replay files remain under the run
volume; only compact scene/result rows belong in the database. The current
deployment keeps honest recorded/offline states when a live worker or remote
database is absent.

## Prompt 024 high-resolution failure analyst

`dryrun.nemotron_real` sends one 896x896 sheet containing four 448px rollout
panels (`before_grasp`, `end_closing`, `presentation`, `end_cycle`) per failed
brick. The request contains only the sheet and requires exactly
`failure_stage`, `visible_support`, `scene_change`, and `evidence_panel`; the
simulator event label is joined only after the response. The allowed stage set
is `no_clear_grasp`, `missed_grasp`, `slipped`, `wrong_verdict`,
`dropped_outside_bowl`, `pushed_out_of_reach`, and `other`.

`dryrun.nemotron_prompt024_run` retains each request, raw response, retry, and
terminal row under `/bulk3/runs/dryrun/`. After a run,
`dryrun.nemotron_prompt024_score` ID-aligns the 300 event-derived labels and
writes `predictions.jsonl`, denominator-300 stage accuracy, per-stage counts,
tier breakdown, and a confusion table with an explicit `unresolved` column.
The merged run at `/bulk3/runs/dryrun/nemotron_prompt024/score-final-merged/`
is 300/300 valid with stage accuracy `0.436667`; the withdrawn 32px and 64px
analyst artifacts are never used by this contract. The measured curriculum
comparison is recorded in `status/M5.md` and keeps the suggested and random
arms separate until a learned-policy checkpoint exists.

The current Astra sentence audit is 36 requests, with every accepted station
checked for MuJoCo load, rest intersections, reach and camera coverage before
the exact XML is sent through `dryrun.rollout.run`; its retained summary is
`/bulk3/runs/dryrun/stations-final-audit-2344/summary.json`.

## Rollout API (main, published 2026-11-14)

`dryrun.env.StationEnv(seed, tier=None)` is the public MuJoCo environment. `reset()` returns an observation. `step(action)` accepts six finite joint position targets in the order shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper, advances the 2 ms simulator at the public 30 Hz boundary, and returns `(observation, reward, done, info)`. `observe()` returns exactly `{"top": uint8[H,W,3], "inspection": uint8[H,W,3], "joints": float32[6]}`. `expert_action()` may read simulator truth and returns one six-target action. `result()` returns good-bowl, wrong-bowl, dropped brick ids and cycle times.

`dryrun.rollout.run(seed, controller, inspector=None, record=None, tier=None)` uses this same `StationEnv.step` path for expert and learned controllers. `controller` can be `"expert"`, a callable taking the observation, or an object with `act(observation)`. `inspector(image)` is called at inspection presentation events. `record(frame)` receives every step with timestamp, observation, action and recent events. The event rows use the names `reach`, `grasp`, `lift`, `present`, `verdict`, `drop`, `push`, `miss`, `slip`, `ik_fail`, and `unstable`, with brick id and simulator time when applicable.

## Telemetry boundary

The benchmark and replay exporter call `dryrun.telemetry.record_rollout(samples, events, outcome, seed=seed, checkpoint="expert")`. Samples are 30 Hz rows with `ts`, six joint values and `gripper`; events and the final outcome are written with the same rollout ID. The writer targets the local `DRYRUN_DB_URL` container during runs.

`uv run python -m dryrun.telemetry sync` copies completed local rollout IDs to the Tiger Cloud URL held in `DRYRUN_TIGER_DB_URL`. It skips IDs already present, so repeating it is safe. `remote-stats` reads Tiger's hypertable catalog, compression stats, and aggregate counts. The dashboard reads `rollout_1s` and `benchmark_checkpoint`; it publishes raw row count, compression ratio, and last rollout timestamp, and only exposes an aggregate query timing when it is at most 300 ms.

Benchmark telemetry writes each episode in one transaction to Tiger. Raw sample
rows carry `rollout_id`, `seed`, `tier`, `checkpoint`, `expert_version`, `ts`,
`q0..q5`, `target0..target5`, `gripper`, and `gripper_effort`; event rows
(`reach`, `grasp`, `lift`, `present`, `verdict`, `drop`, `push`, `miss`, `slip`,
`ik_fail`, `unstable`) live in `rollout_events`. `rollout_results` carries the
episode's `bricks`, `correct`, `false_accepts`, `false_rejects`, `pushes`, and
`cycle_time_s`. The tier API reports aggregate rows by `tier` and
`expert_version`, with `source: "timescale"` when Tiger is answering.
The full-rate held-out loader uses PostgreSQL `COPY` once per episode; its
telemetry-only mode skips camera rasterization while preserving MuJoCo's 30 Hz
physics and joint/action samples.

## Page session request from Soren review

Main provides `benchmark.py --export-replays DIR` rows for every held-out seed and controller. Each replay row is JSONL with `controller`, `seed`, `tier`, brick count, correct count, event rows, and scene values from `BrickSpec` (size, x, y, yaw, defect, color); the directory contains a top-camera thumbnail per row. Replay files and thumbnails live outside the database under the run directory or Fly volume. The page displays the commit that produced each row and marks missing live workers honestly.
## Policy records and analyst scene batches

`data/policy_scores.jsonl` contains one row per `(controller, checkpoint,
tier, seed)` with `controller`, `checkpoint`, `seed`, `tier`, `bricks`,
`correct`, `score`, `events`, `stages`, `closest_approach_mm`, `commit`,
`image_size`, `open_loop_ticks`, and `max_delta`. The page's policy-score API
returns `{schema: "dryrun.policy-scores.v1", rows, checkpoints}`; each
`checkpoints` entry groups rows by controller/checkpoint and exposes
`reached`, `grasped`, `lifted`, `presented`, `dropped`, `right_bowl`, and the
median `closest_approach_mm`. The page uses these rows for checkpoint curves
and scene cards. Featured replay binaries stay under
`/bulk3/runs/dryrun/<run>/` and are keyed by the same seed/checkpoint pair.

The real policy dataset records both cameras at `image_size=224` (the full
top view, with no crop), six joint values, the inspector verdict, and a
60-tick ACT action chunk. The bounded fallback controller consumes the first
30 actions open-loop, replans at the 30 Hz camera refresh, and clamps each
joint target to `max_delta=0.06` radians per tick; the earlier 64 px/201-step
run remains a labelled before result.

Analyst scene batches are JSONL objects of the form
`{"seed": 123, "tier": "messy", "sentence": "...", "source": "nemotron"}`.
`source` is `nemotron` or `random`; unknown keys are rejected. Nemotron and
equal-size random batches are trained from the same starting checkpoint and
scored on the fixed held-out set.

## Analyst/session request

Loose-tier scenes use the normal seeded sentence contract: `layout_bricks(seed, TIERS["loose"])` returns 3–7 bricks with seed-dependent positions, sizes, colours, defects and symmetry-equivalent yaw branches. The benchmark never edits generated XML or positions. The Prompt 021 reference expert plans clear grasps from live MuJoCo geometry with scratch-state IK.

## Main request after loose-tier review

Loose is not valid when a benchmark edits an XML position or repeats one brick. The station owner exposes seed-driven 3–7 brick loose scenes, with both sizes and yaw, inside the measured polar envelope. Main consumes those specs unchanged; the benchmark must not patch positions.

## Measured loose expert recipe (prompt 019)

`dryrun.expert.MeasuredExpert` is the single reference expert used through `StationEnv.step`. Its TCP is `(0.0225, 0, -0.086)` in the gripper body; fixed cycle targets use the measured neutral, presentation, and bowl-drop poses. Events are physical `grasp`, `lift`, `present`, `drop`, `push`, `miss`, `slip`, `ik_fail`, and `unstable` rows with brick and simulator time. A `push` is a recovery action that separates a blocked cluster, not a failed rollout.

Loose scenes are 3–7 seeded bricks in radius `0.12–0.24 m`, azimuth `±0.85 rad` about `(0.0388, 0)`, with 78 mm/95 mm centre spacing and side bowls at `(-0.0093, ±0.2657)`. The expert only queues bricks with a clear geometry-and-IK grasp plan. The exact pad and contact values are in the vendored SO-101 XML and `status/M1.md`.

## Modal mapped policy and demo records

`modal/rollouts.py` fans expert seeds out with `rollout_seed.map`. Each task
stores a `dryrun-demo-shards` Dict value keyed by `run_id/tier/seed`; successful
cycle frames contain the public top and inspection images (JPEG), six-joint
state, six-joint target action, verdict and event tail. The local entrypoint
writes a manifest and JSONL rows under `/bulk3/runs/dryrun/m4/modal-map/`.
`modal/merge_demos.py` is the single writer that converts those values into a
LeRobot 0.7 dataset on the `dryrun-demos` volume. `modal/policy_eval.py` uses
the same `.map` fan-out and writes one row per seed to the `dryrun-policy-rows`
Dict, with `--download` materializing rows under the same `/bulk3` run
directory. A bounded map is infrastructure evidence only; no short-horizon
or archived expert result is a held-out benchmark score.
