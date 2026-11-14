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

The rebuilt inspector shot uses `INSPECTION_PRESENT_POSE = [0.28, 0.0, 0.16]`
metres and `INSPECTION_CAMERA_POSITION = [0.28, -0.10, 0.23]` metres: a
0.122 m fixed camera distance, 34 degree field of view, and a 384x384 RGB
frame. `render_inspection` and `render_inspection_frame` use the same camera
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
under `/bulk2/runs/dryrun/m5/diagnoses/`.

`dryrun.telemetry.record_diagnosis(row)` writes the validated result to
`failure_diagnoses`; `dashboard_snapshot()` and `/api/failures` expose the
frame paths, sentence, scene features and next-scene batch to the page. The
row is keyed by `(rollout_id, analyst_version)` so retries do not duplicate a
diagnosis.

`dryrun.stations.generate_and_check(...)` turns each plain sentence into MJCF,
keeps every XML and rejection reason in its JSONL record, and optionally sends
accepted stations through the shared `dryrun.render.rollout` expert API.

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

## Rollout API (main, published 2026-11-14)

`dryrun.env.StationEnv(seed, tier=None)` is the public MuJoCo environment. `reset()` returns an observation. `step(action)` accepts six finite joint position targets in the order shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper, advances the 2 ms simulator at the public 30 Hz boundary, and returns `(observation, reward, done, info)`. `observe()` returns exactly `{"top": uint8[H,W,3], "inspection": uint8[H,W,3], "joints": float32[6]}`. `expert_action()` may read simulator truth and returns one six-target action. `result()` returns good-bowl, wrong-bowl, dropped brick ids and cycle times.

`dryrun.rollout.run(seed, controller, inspector=None, record=None, tier=None)` uses this same `StationEnv.step` path for expert and learned controllers. `controller` can be `"expert"`, a callable taking the observation, or an object with `act(observation)`. `inspector(image)` is called at inspection presentation events. `record(frame)` receives every step with timestamp, observation, action and recent events. The event rows use the names `reach`, `grasp`, `lift`, `present`, `verdict`, `drop`, `miss`, `slip`, `ik_fail`, and `unstable`, with brick id and simulator time when applicable.

## Telemetry boundary

The benchmark and replay exporter call `dryrun.telemetry.record_rollout(samples, events, outcome, seed=seed, checkpoint="expert")`. Samples are 30 Hz rows with `ts`, six joint values and `gripper`; events and the final outcome are written with the same rollout ID. The writer targets the local `DRYRUN_DB_URL` container during runs.

`uv run python -m dryrun.telemetry sync` copies completed local rollout IDs to the Tiger Cloud URL held in `DRYRUN_TIGER_DB_URL`. It skips IDs already present, so repeating it is safe. `remote-stats` reads Tiger's hypertable catalog, compression stats, aggregate counts, and query time. The dashboard reads `rollout_1s` and `benchmark_checkpoint`, and displays the raw row count, compression ratio, aggregate query time, and last rollout timestamp as database health metadata.

## Page session request from Soren review

Main provides `benchmark.py --export-replays DIR` rows for every held-out seed and controller. Each replay row is JSONL with `controller`, `seed`, `tier`, brick count, correct count, event rows, and scene values from `BrickSpec` (size, x, y, yaw, defect, color); the directory contains a top-camera thumbnail per row. Replay files and thumbnails live outside the database under the run directory or Fly volume. The page displays the commit that produced each row and marks missing live workers honestly.
## Policy records and analyst scene batches

`data/policy_scores.jsonl` contains one row per `(controller, checkpoint,
tier, seed)` with `controller`, `checkpoint`, `seed`, `tier`, `bricks`,
`correct`, `score`, `events`, and `commit`. The page uses these rows for
checkpoint curves and scene cards. Featured replay binaries stay under
`/bulk2/runs/dryrun/<run>/` and are keyed by the same seed/checkpoint pair.

Analyst scene batches are JSONL objects of the form
`{"seed": 123, "tier": "messy", "sentence": "...", "source": "nemotron"}`.
`source` is `nemotron` or `random`; unknown keys are rejected. Nemotron and
equal-size random batches are trained from the same starting checkpoint and
scored on the fixed held-out set.

## Analyst/session request

Loose-tier scenes use the normal seeded sentence contract: `layout_bricks(seed, TIERS["loose"])` returns 3–8 bricks with seed-dependent positions, sizes, colours, defects and symmetry-equivalent yaw branches. The benchmark never edits generated XML or positions. `data/reach_map_polar.json` is the measured physics pick/lift map; only cells with at least 95% measured pick rate are eligible, and closed-form pad IK rejects cells outside a 5% joint-limit margin at pickup and presentation.

## Main request after loose-tier review

Loose is not valid when a benchmark edits an XML position or repeats one brick. The station owner must expose seed-driven 3–8 brick loose scenes, with both sizes and yaw, whose placements come from the measured polar reach map. Main will consume those specs unchanged; the benchmark must not patch positions. The closed-form SO-101 IK and numeric cross-check should define the accepted work-area cells and the present pose before new loose results are published.
