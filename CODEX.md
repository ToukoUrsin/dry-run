# Codex use log

The first pass used parallel Codex sessions, but its status claims were rejected because they described stand-ins as completed systems. Further entries must record only verifiable work and exact proof commands. No milestone is currently complete.

| 2026-11-14 | Real MuJoCo M1 build | gpt-6.1-sol (root) | Installed MuJoCo 4.0.1 in the uv environment, vendored official SO-101 MJCF/STL/licence, built a physics scene with free-joint bricks, and rendered `/bulk2/runs/dryrun/m1/seed7e.mp4`. Score is not claimed until benchmark evidence and render review are complete. |
| 2026-11-14 | Spatial geometry and render review | gpt-6-astra (`/root/astra_geometry`) | Assigned official MJCF geometry, Duplo dimensions/studs, defect geometry, sentence-generated stations, automated validity checks, and render review per `prompts/003-astra-for-3d.md`; results pending. |

2026-11-14 18:18 UTC — gpt-6-astra spatial pass (`/root/astra_geometry`): generated `dryrun/mjcf_station.py` and `dryrun/render_mujoco.py`, added the official SO-101 include path wrapper, and rebuilt `assets/station.xml`. Geometry includes measured Duplo 2x2/2x4 bodies (32/64 × 32 × 19.2 mm), 8 mm radius/3.2 mm-high studs with missing/chipped variants, marker/sticker/discolouring geoms, open 16-segment bowls, and physical top/inspection camera mounts. Sentence generation records accepted/rejected MJCFs and checks MuJoCo load, negative contacts, damped IK reach for every brick and bowl, and camera sight lines. Proof commands: `uv run python -m dryrun.mjcf_station check assets/station.xml` -> `loaded=true`, `brick_count=8`, `intersections=[]`, all 10 IK targets `ok=true`, all 3 cameras `sees_station=true`, `pass=true`; `uv run python -m dryrun.mjcf_station batch --seeds 32 --out-dir /bulk2/runs/dryrun/m1/stations32c --record /bulk2/runs/dryrun/m1/stations32c.jsonl` -> `attempted=32 passed=32 rejected=0`; `MUJOCO_GL=egl uv run python -m dryrun.render_mujoco assets/station.xml --out-dir /bulk2/runs/dryrun/m1/frames` wrote and Astra inspected `free_view.png`, `top_view.png`, `inspection_view.png` (all show the arm, studs, bowls, and brick defects). A 500-step actuator check moved a joint by 0.799998 rad (`nq=62`, `nu=6`).

2026-11-14 18:47 UTC — Astra grasp review (`prompts/004-review-of-seed7.md`): inspected real SO-101 jaw frames and measured pad contact. The wrapper now disables gripper mesh collisions, uses only fixed/moving thin high-friction box pads (`condim=4`, friction 1.8), raises gripper actuator force to ±12, and the station uses elliptic contacts (`impratio=10`, `noslip_iterations=3`). `dryrun/grasp_test.py` runs one 2x2 Duplo on an empty table over ten position/yaw trials, logs pad-only contact force, and re-solves the measured brick pose after approach and again before closing; IK solves restore qpos so planning does not teleport the physics state. Proof run: `MUJOCO_GL=egl uv run python -m dryrun.grasp_test --out /bulk2/runs/dryrun/m1/grasp_retry2.jsonl --xml /bulk2/runs/dryrun/m1/grasp_retry2.xml` -> 8/10 lifts, 6/10 both-pad contact, peak pad normal forces up to 56.5 N. This remains open; the two hard positions still lose contact. A temporary larger-pad variant was measured and rejected: 0/10 lifts, 5/10 both-pad contact.
| 2026-11-14 | Seed-7 grasp review | gpt-6-astra + gpt-6.1-sol | Added elliptic contacts, noslip iterations, high-friction condim-4 jaw pads, pad-midpoint IK, contact-force logging, and closed-loop jaw stopping. Varied calibration is 8/10 lifted; M1 stays open. |
| 2026-11-14 | Fly deployment | gpt-6.1-sol (root) | Deployed the static dashboard as `dryrun-junction-2026`; public smoke test returned HTML at https://dryrun-junction-2026.fly.dev/. Public score is honestly unavailable until a remote Timescale URL is supplied. |

| 2026-11-14 | Inspector M2 | gpt-6.1-astra + gpt-6.1-sol | Replaced the 580-image overlay harness with seeded MuJoCo inspection renders, good-only Anomalib training, PaDiM/PatchCore/EfficientAD comparison, OpenVINO CPU timing, held-out defect evaluation and the `Inspector`/photo interfaces. The built-in image-generation tool was unavailable in this session, so the sticker art and table surface are deterministic seeded raster assets generated locally and recorded under the M2 run. |
| 2026-11-14 | M2 review gate | codex review (gpt-6.1-sol) | Ran `codex review --uncommitted` before the M2 status commit. The review worker could not start because its Chrome MCP requested approval while this session's policy is `approval=never`; no review findings were returned. Local compile, Inspector, photo CLI and benchmark smoke checks passed. |

2026-11-14 19:00 UTC — Astra geometry correction: the seed-7 bowl overlap was real; the old 246 mm outside diameter was larger than the station spec. `_bowl` now emits 150 mm outside-diameter open bowls and the checker explicitly rejects bowl-centre distance below 150 mm or negative good/defective bowl contacts. Proof: `uv run python -m dryrun.mjcf_station check assets/station.xml` -> `bowl_overlap=[]`, `intersections=[]`, all IK/camera checks pass; a deliberately overlapping copy returned non-empty `bowl_overlap` and failed; `uv run python -m dryrun.mjcf_station batch --seeds 32 --out-dir /bulk2/runs/dryrun/m1/stations_bowlfix --record /bulk2/runs/dryrun/m1/stations_bowlfix.jsonl` -> 32/32 passed. EGL free/top/inspection frames were rerendered to `/bulk2/runs/dryrun/m1/frames_bowlfix/` and inspected: bowls are separate and visibly within the table/arm scene.

## Session ledger for this run

The task path is the stable session identifier available in the shared
worktree. A platform UUID is not copied from another session and is therefore
not guessed here. The five work streams named in `prompts/010` are listed so a
reviewer can follow ownership; each status file is the proof for its feature.

| Session identifier | Model assignment | Feature used | Proof or record |
| --- | --- | --- | --- |
| `/root` | `gpt-6-astra` for the main spatial stream, with `gpt-6.1-sol` build work recorded above | MuJoCo station, rollout API, expert and benchmark | `status/M1.md`; `docs/interfaces.md` |
| `/root/inspector` | `gpt-6.1-sol` work stream from `prompts/010` | Defects, good-only inspector and OpenVINO | `status/M2.md` |
| `/root/page` | `gpt-6.1-sol` work stream from `prompts/010` | Replay export, telemetry, dashboard and deployment | `status/M3.md`, `status/M6.md` |
| `/root/policy` | `gpt-6.1-sol` work stream from `prompts/010` | LeRobot data, policy training and policy benchmark | `status/M4.md` |
| `/root/analyst_impl` | `gpt-6.1-sol` inherited analyst stream | Nemotron failure analysis, scored curriculum and sentence stations | `status/M5.md` |

The documentation pass ran as `/root/docs_status` on `gpt-6.1-sol`: it rewrote
`README.md`, `SUBMISSION.md`, `FILM_PLAN.md` and `MORNING.md` to link every
claim to status evidence and ran `git diff --check` on those files. The
analyst and Astra streams must append their request/response paths and exact
review output here when their proof runs finish. A `codex review` is required
before the commit that includes those changes; the root session owns that
final review because it sees all shared paths.

2026-11-14 20:00 UTC — Main rollout API and tier benchmark (`gpt-6-astra`, ultra): published `dryrun.env.StationEnv` and `dryrun.rollout.run` with camera/state-only observations, exact 30 Hz substep accumulation, six-target actions, recorder cycle markers, inspector hook and the shared event vocabulary. Added fixed 80/70/50 loose/messy/pile benchmark partitioning and page replay/thumbnail contract in `docs/interfaces.md`. Smoke proof: `MUJOCO_GL=egl uv run python - <<'PY' ... StationEnv.reset/step ... PY` returned uint8 top/inspection frames and float32[6] joints; loose five-seed benchmark was 1/5 (honest open gate; marginal generated cells remain under station-owner review).

2026-11-14 19:51 UTC — M5/Soren integration (`/root`): ran `codex review --uncommitted` before integration; Codex could not inspect because its MCP runner requested approval under `approval=never`. Manual gates `git diff --check` and `.venv/bin/python -m compileall -q dryrun web modal modal_live_defect.py` passed. Added camera-only diagnosis storage, equal curriculum hook, 36-sentence station gallery, bounded station queue/worker heartbeat, and `data/codex_log.jsonl`; proof is in `status/M5.md`.
