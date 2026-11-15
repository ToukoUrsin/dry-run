# Supervisor to all five sessions, Sun 15.11.2026 01:14 UTC: the page is one good screen and five empty tabs

I opened every tab of the live page at 01:15 UTC. Station is good. The rest: Benchmark shows dashes and "Waiting for TimescaleDB"; Replays has one row; Scenes has one card; Inspector shows the withdrawn first attempt (161 / 239 / 535 / 1265); Method is three sentences. Meanwhile the work exists on disk: 32 stations made from sentences, 300 Nemotron diagnoses, an inspector with a confusion table, 774 demonstration episodes, three tiers, 33 supervisor prompts, five session recordings. A judge who clicks once past the first screen sees nothing of it.

Fix: **each session publishes its own results as one JSON file; the page session draws them.** Nobody but the page session edits `web/`. Nobody but the owner edits a JSON file.

## Contract

- Files: `data/page/<name>.json`, committed. Every file has `"updated_utc"`, `"commit"` (the commit the numbers were measured on), and `"summary"`: one plain sentence with the main number, written for a stranger.
- Pictures: at most 12 per section, WebP, 640 px wide, under 60 KB each, in `web/static/evidence/<name>/`, referenced by relative path. These are page content and go in git. Nothing else binary does.
- Only measured numbers. A thing not measured is left out of the file, not written as null or "pending".
- First version of every file by **02:10 UTC**, even if a number is still the earlier one. Update the file whenever a number changes; the page reads the files at request time (`/api/page/<name>`), so no page edit is needed.

| File | Owner | Holds |
|---|---|---|
| `tiers.json` | main | per tier and expert version (`64c4919` = before, new = after): scenes, bricks, correct, per cent, pushes, median cycle seconds per brick, failures by event. `replays`: id, tier, seed, score for one loose, one touching and one pile replay with pushes. |
| `tiger.json` | main | raw rows, rollouts, aggregate rows, compression ratio, query milliseconds, read live from Tiger Cloud when the page asks (prompt 031). |
| `inspector.json` | inspector | model, number of good training frames, test frames, the 2 × 2 table, false accepts and false rejects in per cent, recall per defect type with a `held_out` flag, CPU milliseconds per frame (1 and 4 threads, OpenVINO). `examples`: 8 real `present` frames with truth, verdict, score and the anomaly map. `modes`: the three rows of prompt 032 (ground truth / ground-truth poses + inspector / cameras only) as they land. The withdrawn first attempt goes in `earlier` with one line saying why it failed. |
| `policy.json` | policy | dataset (episodes, frames, format, image size, loads in LeRobot: yes). `runs`: name, image size, steps, frames, GPU, minutes, and per checkpoint the stage counts of prompt 030 plus closest approach in mm. OpenVINO milliseconds for the policy. |
| `analyst.json` | analyst | model id, failures analysed, frame size, stage accuracy per prompt version (the 32 px run stays as the earlier row), `examples`: 6 failures with the 2 × 2 sheet, the simulator's stage, what Nemotron said. The batch comparison if measured. |
| `stations.json` | analyst | attempted, accepted, rejected by reason. `items`: 12 stations with the sentence, a render, and the checks passed. |
| `codex.json` | analyst | sessions (name, model, effort, hours, commits), Codex features used with one line of evidence each, number of supervisor prompts, a timeline of the real turning points with UTC time and commit (first pass rejected 18:05, zero pickups, the measured recipe 22:10, first sorted brick, 382/382, pushes), and the boundary line: the grasp recipe and two reference experts were prototyped by the supervising agent outside Codex (`prompts/019`, `021`, `027`). |

## Page session

After the arm mesh (prompt 029 item 1, still first):

1. One scrolling page in story order, the nav jumps to anchors: Station (as now) → Messier piles (before → after by tier, touching and pile replays) → Inspector → Cameras only → Learned policy → Nemotron → Stations from a sentence → Every rollout in Tiger Data → Built with Codex → What is simulated.
2. Each section: the `summary` sentence as its headline, one chart or table, the pictures, the commit in small type. Charts drawn from the JSON (score by checkpoint, stage funnel, recall per defect type, before → after bars).
3. A section whose file is missing is not drawn. No dashes, no "waiting for", no empty tables. Delete the Benchmark, Replays and Scenes tabs as they are now.
4. Draw-your-own-defect stays, inside the Inspector section.
5. Target: sections live by 02:45 UTC with whatever files exist. Then open every section at 1440 and 390 px yourself.

Feature freeze is 03:00 UTC: after that only numbers in the JSON files change.
