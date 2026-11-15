# Codex use log

The first pass used parallel Codex sessions, but its status claims were rejected because they described stand-ins as completed systems. Further entries record only verifiable work and exact proof commands. Open gates remain labeled open; completed analyst and station evidence is recorded below.

## Judge card

Dry Run's goal was to build an evidence-first simulator and page for the
Junction SO-101 station before a real arm moves. At the generated 03:13 UTC
snapshot, the repository contained 274 attributed commits, 5 sessions, 9.23
elapsed goal-mode hours, 11,667 tracked code lines and 45 prompt files. Counts
are generated from `git log --name-only` by `scripts/night_data.py`; the page
snapshot is `data/page/codex.json`.

| Session | Model | Owns | Commits |
| --- | --- | --- | ---: |
| `/root` | gpt-6-astra plus gpt-6.1-sol | simulator, expert, benchmark, replay | 140 |
| `/root/inspector` | gpt-6.1-sol | defects and OpenVINO inspection | 13 |
| `/root/page` | gpt-6.1-sol | viewer, telemetry, deployment | 56 |
| `/root/policy` | gpt-6.1-sol | demonstrations, ACT, policy evaluation | 33 |
| `/root/analyst_impl` | gpt-6.1-sol plus Nemotron | failure analysis, stations, evidence docs | 28 |

The four film-plan commits are kept in the `other` lane, so the five session
lanes sum to 270 and the total is 274. This is why an older page lane count
must not be compared with a hand-counted main total.

## What went wrong and how it was caught

- **First pass rejected (18:05).** Prompt 002 records that a 245-line stub claimed milestones without a simulator.
  Later MuJoCo checks replaced the stub with a runnable scene and evidence.
- **Zero pickups (18:32).** The seed-7 review in Prompt 004 found no useful grasps.
  Pad-contact trials made the failure measurable before more tuning.
- **Inspector rejected good bricks (19:47).** Prompt 014 records the first real-frame model rejecting good parts.
  Real-frame retraining and a held-out evaluation replaced that attempt.
- **Loose tier rigged (19:58).** Prompt 016 found one brick at one fixed spot for every seed.
  Its scores were withdrawn from benchmark and training use.
- **Zero of fifty (22:10).** Prompt 019 records no successful lifts after four hours with the measured recipe.
  The prompt supplied the measured boundary and recipe used for the replacement expert.
- **Policy reached zero stages (01:05).** The first 201-step policy run reached no physical stage, as Prompt 030 records.
  The run remains a failure instead of being presented as a success.
- **Live page broken (01:41).** Prompt 034 found a console error and a missing first screen.
  The browser repair added a smoke gate for desktop and phone views.

| 2026-11-14 | Real MuJoCo M1 build | gpt-6.1-sol (root) | Installed MuJoCo 4.0.1 in the uv environment, vendored official SO-101 MJCF/STL/licence, built a physics scene with free-joint bricks, and rendered `/bulk2/runs/dryrun/m1/seed7e.mp4`. Score is not claimed until benchmark evidence and render review are complete. |
| 2026-11-14 | Spatial geometry and render review | gpt-6-astra (`/root/astra_geometry`) | Assigned official MJCF geometry, Duplo dimensions/studs, defect geometry, sentence-generated stations, automated validity checks, and render review per `prompts/003-astra-for-3d.md`; proof is recorded in `status/M1.md` and `status/M5.md`. |

2026-11-14 18:18 UTC — gpt-6-astra spatial pass (`/root/astra_geometry`): generated `dryrun/mjcf_station.py` and `dryrun/render_mujoco.py`, added the official SO-101 include path wrapper, and rebuilt `assets/station.xml`. Geometry includes measured Duplo 2x2/2x4 bodies (32/64 × 32 × 19.2 mm), 8 mm radius/3.2 mm-high studs with missing/chipped variants, marker/sticker/discolouring geoms, open 16-segment bowls, and physical top/inspection camera mounts. Sentence generation records accepted/rejected MJCFs and checks MuJoCo load, negative contacts, damped IK reach for every brick and bowl, and camera sight lines. Proof commands: `uv run python -m dryrun.mjcf_station check assets/station.xml` -> `loaded=true`, `brick_count=8`, `intersections=[]`, all 10 IK targets `ok=true`, all 3 cameras `sees_station=true`, `pass=true`; `uv run python -m dryrun.mjcf_station batch --seeds 32 --out-dir /bulk2/runs/dryrun/m1/stations32c --record /bulk2/runs/dryrun/m1/stations32c.jsonl` -> `attempted=32 passed=32 rejected=0`; `MUJOCO_GL=egl uv run python -m dryrun.render_mujoco assets/station.xml --out-dir /bulk2/runs/dryrun/m1/frames` wrote and Astra inspected `free_view.png`, `top_view.png`, `inspection_view.png` (all show the arm, studs, bowls, and brick defects). A 500-step actuator check moved a joint by 0.799998 rad (`nq=62`, `nu=6`).

2026-11-14 18:47 UTC — Astra grasp review (`prompts/004-review-of-seed7.md`): inspected real SO-101 jaw frames and measured pad contact. The wrapper now disables gripper mesh collisions, uses only fixed/moving thin high-friction box pads (`condim=4`, friction 1.8), raises gripper actuator force to ±12, and the station uses elliptic contacts (`impratio=10`, `noslip_iterations=3`). `dryrun/grasp_test.py` runs one 2x2 Duplo on an empty table over ten position/yaw trials, logs pad-only contact force, and re-solves the measured brick pose after approach and again before closing; IK solves restore qpos so planning does not teleport the physics state. Proof run: `MUJOCO_GL=egl uv run python -m dryrun.grasp_test --out /bulk2/runs/dryrun/m1/grasp_retry2.jsonl --xml /bulk2/runs/dryrun/m1/grasp_retry2.xml` -> 8/10 lifts, 6/10 both-pad contact, peak pad normal forces up to 56.5 N. This remains open; the two hard positions still lose contact. A temporary larger-pad variant was measured and rejected: 0/10 lifts, 5/10 both-pad contact.
| 2026-11-14 | Seed-7 grasp review | gpt-6-astra + gpt-6.1-sol | Added elliptic contacts, noslip iterations, high-friction condim-4 jaw pads, pad-midpoint IK, contact-force logging, and closed-loop jaw stopping. Varied calibration is 8/10 lifted; M1 stays open. |
| 2026-11-14 | Fly deployment (historical) | gpt-6.1-sol (root) | Deployed the first static dashboard as `dryrun-junction-2026`; its initial smoke test returned HTML at https://dryrun-junction-2026.fly.dev/. The score was unavailable then because no remote Timescale URL was supplied; the later Tiger-backed deployment is audited below. |

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

## Features used and their proof

| Feature | Session and model | Evidence |
| --- | --- | --- |
| Goal mode | `/root`, gpt-6-astra | `prompts/001-goal.md`; the objective and boundary are recorded in the goal log. |
| Parallel sessions and delegation | all five paths above | `prompts/010-ultra-and-five-sessions.md`; the session ledger above records ownership. |
| Ultra effort | all five streams, gpt-6.1-sol and gpt-6-astra | `prompts/010-ultra-and-five-sessions.md`; the supervisor assigned ultra effort to every stream. |
| Fast service tier | all five streams | The launch records use `service_tier=fast`; no with/without timing comparison was measured, so no timing claim is made. `CODEX.md` and the session logs are the evidence. |
| Astra spatial reasoning | `/root/astra_geometry` and `/root/curriculum_astra`, gpt-6-astra | `prompts/003-astra-for-3d.md`; `status/M1.md`; `status/M5.md`. |
| Nemotron visual analysis | `/root/analyst_impl`, gpt-6.1-sol plus NVIDIA Nemotron | `dryrun/nemotron.py`; `data/page/analyst.json`; `status/M5.md`. |
| Models by job | Astra for geometry/spatial checks; Sol for build, page, inspector and policy; Nemotron for visual failure analysis | `prompts/003-astra-for-3d.md`; the five-session ledger; `data/page/analyst.json`. |
| Browser and public smoke checks | `/root/page`, gpt-6.1-sol | `scripts/smoke_public.py`; page captures and console/error output are under `/bulk3/runs/dryrun/page-shots/`. |
| AGENTS.md and secret/telemetry boundaries | `/root` and the named streams | `AGENTS.md`; `status/M5.md` records the hsec and telemetry boundaries used. |
| Codex review | `/root`, gpt-6.1-sol | `/bulk3/runs/dryrun/codex-review-goal012-final.log`; no findings were returned because sandbox/MCP inspection was unavailable, so this is not a green review. |
| Image generation | Codex desktop app, gpt-6.1-sol, codex-cli 0.169.0-alpha.11.1 | **WORKED:** built-in `image_gen` made `table-wood.png` and `defect-stickers.png` (1254x1254); proof `/bulk2/runs/dryrun/codex-features/imagegen/*.png`, `imagegen/mac-try.log`. Server cliproxy did not expose the tool and its sandbox failed (`imagegen/try1.log`), so earlier textures were code-generated. |
| Sites | Codex desktop app, live APIs | **WORKED:** published owner-only results Site `https://dryrun-results.jarmo1.chatgpt.site` (`appgprj_6af9206a560c8191bc222fffa878783e`); corrected `/api/inspector` to `/api/page/inspector` after the stale synthetic result. Proof `sites/run.jsonl`, `run2.jsonl`, `site-index.html`, `site.png`. |
| Documents | Codex desktop app | **WORKED with caveat:** wrote `dryrun-test-report.docx`; its stale inspector number and placeholder layout image were wrong, and LibreOffice render was killed. Real render: `docs/dryrun-test-report.pdf`, `page-1.png`; proof `docs/run.jsonl`, `run2.jsonl`. |

**Boundary:** Everything outside `prompts/` was written by Codex. The
supervisor wrote the prompts, and two reference sorters under
`prompts/021-reference` and `prompts/027-reference`, which Codex ported into
`dryrun/expert.py`.

2026-11-15 03:10 UTC — Feature audit: image generation, Sites and Documents were exercised with the caveats above. Proof is under `/bulk2/runs/dryrun/codex-features/`.

2026-11-14 20:00 UTC — Main rollout API and tier benchmark (`gpt-6-astra`, ultra): published `dryrun.env.StationEnv` and `dryrun.rollout.run` with camera/state-only observations, exact 30 Hz substep accumulation, six-target actions, recorder cycle markers, inspector hook and the shared event vocabulary. Added fixed 80/70/50 loose/messy/pile benchmark partitioning and page replay/thumbnail contract in `docs/interfaces.md`. Smoke proof: `MUJOCO_GL=egl uv run python - <<'PY' ... StationEnv.reset/step ... PY` returned uint8 top/inspection frames and float32[6] joints. The loose benchmark number was withheld: the seeded multi-brick generator and closed-form reach gate were still being replaced, so no score from that run is evidence.

2026-11-14 19:51 UTC — M5/Soren integration (`/root`): ran `codex review --uncommitted` before integration; Codex could not inspect because its MCP runner requested approval under `approval=never`. Manual gates `git diff --check` and `.venv/bin/python -m compileall -q dryrun web modal modal_live_defect.py` passed. Added camera-only diagnosis storage, equal curriculum hook, 36-sentence station gallery, bounded station queue/worker heartbeat, and `data/codex_log.jsonl`; proof is in `status/M5.md`.

## Codex review — 2026-11-14

Command: `codex review --uncommitted` (run from `/home/ubuntu/junction/dryrun`).
The review process started but could not inspect the repository: its command runner
failed on a sandbox mount before repository commands ran. It also logged that
Chrome DevTools required approval while the project approval policy was `never`.
No review findings were produced; this is not a clean review result.

## Historical page rounds and proof (2026-11-14; superseded)

The page rounds are preserved under `/bulk3/runs/dryrun/page-shots/`; each round
has laptop and phone captures. The changes recorded here are the ones that are
actually present in the source and were followed by browser checks:

| Round | Change | Evidence |
| --- | --- | --- |
| 2 | Added the real seed-7 MuJoCo replay, embedded top/inspection videos, and the compressed SO-101 mesh asset. | `page-shots/round-2/` |
| 3 | Added the public smoke gate and aggregate-backed inspector/API states. | `page-shots/round-3/` |
| 4 | Reworked the viewer with a Z-up camera, lighting/shadows, smooth bowls, rounded bricks, and synchronized cameras. | `page-shots/round-4/` |
| 5 | Added Scenes/Compare workspace views, hid unavailable views, and selected the failed replay honestly. | `page-shots/round-5/` |
| 6 | Split the long report into hash-routed workspace views and made viewer readiness deterministic for the smoke test. | `page-shots/round-6/` |
| 7 | Kept Station as the first workspace view and reran the public gate at both judge sizes. | `page-shots/round-7/` |
| 8 | Replaced the long report navigation with a persistent workspace sidebar, separate failure/defect/station/Codex routes, responsive mobile tabs, a tighter first viewport, and a level Z-up viewer. | `/bulk3/runs/dryrun/page-shots/d083bf9-public-final/` |

The initial deployment proof used commit `d083bf9` and the clean-worktree smoke run:

```text
uv run --project /home/ubuntu/junction/dryrun --no-sync \
  python scripts/smoke_public.py --url https://dryrun-junction-2026.fly.dev \
  --screenshot-dir /bulk3/runs/dryrun/page-shots/d083bf9-public-final
-> 1440px non_background_pixels=203593, videos=[4,4], scroll_width=1440,
   errors=[]; 390px non_background_pixels=84526, videos=[4,4],
   scroll_width=390, errors=[]
```

At that historical snapshot the page and telemetry were live, with 304
camera-only Nemotron diagnosis rows (all version `v4-both-chain-fewshot`) and
the station gallery evidence described below. The fixed-scene loose run was
never used as a benchmark or training result.

The measured recipe from `prompts/019-grasp-recipe-measured.md` and
`prompts/020-present-pose-and-what-changes.md` was subsequently ported by Codex
and integrated. The current page uses the seed-1001 replay; the old failed
replay remains historical evidence only.

2026-11-14 22:34 UTC — Nemotron v4 completion (`/root/analyst_impl`): the 300-row camera-only diagnostic set reached 300 valid responses after retry-5; ID-aligned stage/cause/combined accuracy is 0.090000/0.000000/0.045000. This is diagnostic evidence from the withdrawn loose-tier failures, never a benchmark or training score. Proof: `/bulk3/runs/dryrun/nemotron_accuracy_v4/score.json`, source `8b3d3a4`.
2026-11-14 23:04 UTC — Recipe-compatible station gallery (`/root/analyst_impl`): 36 distinct sentence requests were checked; 32 passed and four rest-intersection rejections were retained. The expert rerun was pending at this checkpoint and was completed in the later post-reference audit recorded below. Proof: `/bulk3/runs/dryrun/stations36-reference-check/summary.json`.

2026-11-14 23:05 UTC — Supervisor reference expert handoff (`prompts/021-reference-expert.md`): the earlier measured expert was found broken and its rollout numbers are withdrawn. Main owns the port of the external reference expert and the protected physics files; analyst evidence here does not claim the external reference run as a repository score.

2026-11-14 23:22 UTC — Reference expert integrated (`64c4919`): the measured
external recipe in `prompts/019-grasp-recipe-measured.md` and
`prompts/021-reference/` replaced the slow experimental expert. The station
camera mount, episode tick limit, fresh presentation render and ground-truth
correct-bowl scoring were fixed. The old fixed-scene loose release was later
withdrawn from benchmark use; its historical artifacts remain only for audit.

2026-11-14 23:43 UTC — `codex review --uncommitted` was attempted after the
Prompt 022 main changes. The review worker could not inspect the repository
because Blender MCP requested approval while this session's policy is
`approval=never`; no review findings were returned.

2026-11-15 00:16 UTC — The featured replay exporter was completed at
`bcee980`: it stops when the reference expert enters `done` and writes free,
top, and inspection videos plus 12 linked 256 px presentation crops. The
clean seed-1000 replay sorts 6/6 with 12 verdict events; evidence is under
`/bulk3/runs/dryrun/m1/reference-replay-seed1000-bcee980/`.

2026-11-15 00:45 UTC — Main session added the explicit 3--12 scene-count
contract at `0d0b556` and recorded it in `0339c52`. Seeded 12-brick fixtures
load as MuJoCo stations under `/bulk3/runs/dryrun/m1/explicit12-station.xml`;
the full suite passed 21 tests.

2026-11-15 02:18 UTC — The final 448 px Nemotron analyst page artifact was
published from the merged strict run: 300/300 valid responses, 131 correct
stage decisions, accuracy `0.436667`, full confusion matrix and messy/pile
breakdown. The withdrawn 32 px result remains explicitly excluded. Proof:
`data/page/analyst.json` and `/bulk3/runs/dryrun/nemotron_prompt024/score-final-merged/score.json`.

2026-11-15 02:22 UTC — The Codex night snapshot and station page data were
refreshed. The station gallery records 12 varied accepted page examples and
three unsupported requests; the complete Astra sentence audit is in M5.
Proof: `data/page/codex.json`, `data/page/stations.json`, and
`status/M5.md`.

2026-11-15 02:24 UTC — The page snapshot generator was rerun through the
current repository history. It records every prompt file, commit attribution,
session ledger and benchmark-safe touching/pile score series; fixed-scene loose
scores are omitted. Proof command: `PYTHONPATH=. .venv/bin/python
scripts/night_data.py`.

## Current public deployment audit — 2026-11-15 02:45 UTC

Direct requests to the live deployment returned the following values. The
endpoint payloads and status files are the source for these numbers; this log
does not turn them into a physical-arm claim.

```text
curl -fsSL https://dryrun-junction-2026.fly.dev/api/health
-> {"status":"ok","commit":"bbf30f4","replay_count":1}

curl -fsSL https://dryrun-junction-2026.fly.dev/api/score
-> checkpoint=expert-v2-b9d897c, source_commit=b9d897c,
   correct=1090, bricks=1262, rollouts=200, accuracy=0.8637083994,
   false_accepts=1, false_rejects=0, cycle_time_s=10.0500579325

curl -fsSL https://dryrun-junction-2026.fly.dev/api/benchmark-tiers
-> source=timescale; 64c4919 loose=382/382, b9d897c loose=380/382;
   64c4919 messy=217/494, b9d897c messy=451/494;
   64c4919 pile=31/386, b9d897c pile=259/386

curl -fsSL https://dryrun-junction-2026.fly.dev/api/telemetry
-> source=timescale, raw_rows=20,723, result_rows=401,
   diagnoses returned=100, compression_enabled=true, compression_ratio=2.2

curl -fsSL https://dryrun-junction-2026.fly.dev/api/page/policy
-> commit=bbf30f4, crop224 ACT at 30,000/100,000 steps,
   0/100 physical stages at the 10k, 20k and 30k maps; status=running
```

The public score is a Tiger-backed simulator checkpoint. The current ACT run
has no physical-stage success, so no learned-policy benchmark or
Nemotron-versus-random policy comparison is claimed. The current browser smoke
capture and screenshot paths remain in `status/M6.md`.
