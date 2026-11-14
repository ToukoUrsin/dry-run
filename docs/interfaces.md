# Shared session contracts

The main session owns station generation, the MuJoCo expert, the benchmark, demos, policy and Nemotron loop. The page session owns `web/`, replay export, telemetry and Fly. The inspector session owns defect geometry, inspection renders, Anomalib and OpenVINO.

`dryrun.mjcf_station.station_xml(seed, sentence, count=None)` returns a complete MJCF string. `layout_bricks(seed, sentence, count=None)` returns ordered `BrickSpec` labels; body `brick_N` corresponds to index `N`. A consumer must load the returned XML in MuJoCo and call `mj_forward` before reading poses.

`dryrun.render.rollout(model, data, seed, writer=None, specs=None)` steps the real physics from the supplied simulator state. It returns `{seed, bricks, successes, score, events}`; each event records physical contact pickup, pad contacts, final pose, bowl membership and success. It never teleports or welds a body.

The benchmark calls the inspector at the point where a brick is held at the inspection camera. The inspector returns a bowl decision and timing; the benchmark records that decision in the event. The rollout calls the telemetry writer after each physics step or at its 30 Hz sampling boundary. If an integration is not wired, the owning status file remains open and names the missing call.

## Replay export format

`uv run python -m dryrun.replay_export --seed N --out DIR` writes a compressed `replay.json.gz` plus MuJoCo camera videos. The JSON contains `scene.bodies` (parent, primitive or mesh geoms, sizes, and RGBA), `scene.cameras`, `scene.meshes`, 30 Hz `frames`, `events`, and `outcome`. A frame pose is `[x, y, z, qw, qx, qy, qz]`; positions use `1e-4 m` quantisation and quaternion components use signed int16 scale (`32767`). The pose clock in `frame.t` is authoritative. Videos carry `fps=10`, `source_pose_hz=30`, and `frame_stride=3` because each camera frame is rendered from every third pose.

Event names are `approach`, `grasp`, `lift`, `at_inspection`, `present`, `verdict`, `release`, `drop`, `in_bowl`, `dropped`, and `failed`. Failure events include a `reason`; brick events include `brick` and the event frame.

## Rollout API (main, published 2026-11-14)

`dryrun.env.StationEnv(seed, tier=None)` is the public MuJoCo environment. `reset()` returns an observation. `step(action)` accepts six finite joint position targets in the order shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper, advances the 2 ms simulator at the public 30 Hz boundary, and returns `(observation, reward, done, info)`. `observe()` returns exactly `{"top": uint8[H,W,3], "inspection": uint8[H,W,3], "joints": float32[6]}`. `expert_action()` may read simulator truth and returns one six-target action. `result()` returns good-bowl, wrong-bowl, dropped brick ids and cycle times.

`dryrun.rollout.run(seed, controller, inspector=None, record=None, tier=None)` uses this same `StationEnv.step` path for expert and learned controllers. `controller` can be `"expert"`, a callable taking the observation, or an object with `act(observation)`. `inspector(image)` is called at inspection presentation events. `record(frame)` receives every step with timestamp, observation, action and recent events. The event rows use the names `reach`, `grasp`, `lift`, `present`, `verdict`, `drop`, `miss`, `slip`, `ik_fail`, and `unstable`, with brick id and simulator time when applicable.
