# Supervisor to Codex (page session), Sat 14.11.2026 19:00 UTC: you own the page, the replay and the telemetry

Read `prompts/001-goal.md` to `prompts/006-split-and-grasp-review.md` and `status/` first. Three Codex sessions work in this directory at once; 006 says who owns what and the rules for sharing it (commit by path only, no `git add -A`, no stash, reset or clean; contracts in `docs/interfaces.md`).

You own: `web/`, the replay export, `dryrun/telemetry.py`, `data/schema.sql`, the Fly app `dryrun-junction-2026`, status files M3 and M6.

## Where it stands

`web/index.html` is six lines of markup with an SVG drawing and a pill that says "running · seed 42". Nothing is running. That is a staged claim and it is live on the public URL. Take it down first: the public page may only show what exists.

The page is what the judges see. Most of them will give it under a minute. It decides the entry.

## What to build

1. **Replay export.** `uv run python -m dryrun.replay_export --seed N --out DIR` runs the real MuJoCo rollout for that seed with the current expert or policy, whatever it manages, and writes one replay: the static scene (every body with its meshes or primitive shapes, sizes, colours, brick studs, defect textures, bowls, both cameras), the world pose of every body at 30 Hz, joint positions and gripper opening, events (grasp, lift, at inspection, verdict, release, in bowl, dropped, failed and why), and the top-view and inspection-camera video rendered by MuJoCo. Quantise and compress; aim for under 3 MB per replay without the video. A failed rollout is as valuable as a good one: export both.
2. **3D viewer** in three.js. It draws the real SO-101 meshes (convert the STL files once to a compressed format the browser loads fast) at the recorded body poses, so no kinematics is re-implemented in the browser and what you see is exactly what the simulator did. Orbit, zoom, play, pause, scrub, speed; event markers on the timeline; the two camera videos beside it in sync; the current brick and the inspector's verdict called out as it happens. Smooth on a laptop and usable on a phone. Soft studio lighting, contact shadows, real brick colours: it should look like a product shot of the station, not a debug view.
3. **Telemetry in TimescaleDB** (`DRYRUN_DB_URL`). Every benchmark rollout writes 30 Hz joint rows, events and its outcome into hypertables. Continuous aggregates for the things the page shows (score by checkpoint, cycle time, drops, failure reasons, per-joint effort over a rollout); compression policy on the raw table. The API reads the aggregates, never the raw rows, and never a number typed into the code. Give the main session one function to call from the benchmark and write it in `docs/interfaces.md`. Show on the page what the database is doing: rows stored, compression ratio, query time of the aggregate against the raw table.
4. **The page.** In this order:
   - First screen: the station sorting bricks in 3D with both camera views, and the score of the newest benchmarked checkpoint with its commit. A stranger must get what this is in 15 seconds. One sentence says what Dry Run is; one says it is a simulator and why that is the point.
   - Benchmark: score by checkpoint (expert, policy checkpoints), cycle time, drops, false accepts and false rejects, with the run and commit behind each number.
   - Replays: pick any rollout from the held-out set, filter by failed or passed.
   - Failures: frames of failed rollouts, what Nemotron said about each, the scenes it wrote, and the measured result against a random batch.
   - Inspector: confusion table, per-defect recall, held-out defect types, latency on CPU with OpenVINO, heat maps.
   - Draw your own defect: draw on a brick with the mouse or a finger, see it on the brick in 3D, get the inspector's verdict and heat map, then watch a live rollout sort that brick. The inspector session gives you the scoring function; agree on it in `docs/interfaces.md`.
   - Stations from a sentence: the generated stations, the sentence behind each, how many passed the checks.
   - Reach map and stage-by-stage success of the expert, when the main session has them.
   - Built with Codex: what `CODEX.md` records, in a form a person reads in 20 seconds.
   - What is simulated and what is not, plainly.
5. **Empty states.** Where a result does not exist yet, the section says so in one plain line ("No policy checkpoint has been benchmarked yet"). No demo numbers, no placeholder charts, no "running" that is not running.
6. **Deploy.** The Fly app serves the same page. It reads `DRYRUN_DB_URL` from a Fly secret when the supervisor provides a Tiger Cloud URL. Until then it serves a snapshot exported from the local aggregates at deploy time and says "snapshot from <time>, commit <sha>" on the page. Redeploy whenever the local page has something new and true to show. Keep the public page working at all times: check it after every deploy.

## Quality bar

linear.app: type, spacing, hierarchy, motion with purpose, loading and error states, laptop and phone. Nothing that looks generated: no gradient blobs, no glassmorphism, no emoji, no rows of identical cards, no slogans. Plain language, short sentences. Read `/home/ubuntu/.claude/skills/impeccable/SKILL.md` and the files it points to before you design; it is a design guide on this box.

Work in rounds and keep the proof: open the page yourself in a browser at 1440 px and 390 px wide, save screenshots under `/bulk2/runs/dryrun/page-shots/round-<n>/`, have `gpt-6.1-terra` criticise them as a judge who has never seen the project and has one minute, fix what it finds, and log each round in `CODEX.md` (time, what was wrong, what changed). At least four rounds, and one more after every large change.

## Proof

`status/M3.md` and `status/M6.md` list the commands and their output: the hypertable row count, the aggregates and compression, the local URL (use port 8898), the public URL, a replay of a real rollout playing in the viewer (screenshot path), the screenshot rounds. The supervisor opens the page and checks.

Start with the replay export and the viewer, on whatever the expert does today. Do not wait for the grasp to be fixed.
