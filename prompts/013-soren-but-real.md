# Supervisor to every Dry Run session, Sat 14.11.2026 19:38 UTC: what Soren did well, and how we do it for real

Touko wants heavy inspiration from Soren, done better. I had their repo and their 2:14 demo video read end to end. A copy is at `/tmp/soren-ref/repo` (read only; never run it, never copy code or assets from it, only ideas).

## What they built

Two people, one day, about 19,600 lines: 6,700 of frontend, 4,400 of CSS, 3,000 of simulation and training, 1,750 of tests. Our repo is a tenth of that. Their page is a **workspace application**, not a report: sidebar, experiments, tabs. Ours must be one too. A single hand-written HTML file will not carry this; use a real frontend build (Vite, TypeScript, three.js) served by the existing Python app.

What made it land, in order:

1. **"Play both".** Two synced players, same seed, baseline on the left, candidate on the right: one arm leaves the object, the other carries it home.
2. **One number that moves**: 54/80 to 70/80, with "1 regression" shown next to it.
3. **A wall of 16 scene thumbnails with pass badges.** It reads as scale at a glance.
4. **Scenes that explain themselves**: each card is titled with its real values ("Object X -22 mm"), has a role ("Boundary case"), "Why this environment?" and "What to look for".
5. **Scene detail**: 3D player, scrubber, a telemetry chart synced to the scrubber, per-step reward, "Manual operator control" nudge buttons.
6. **A composer**: describe the task in plain words, "Generate experiment plan", "Review", "Launch".
7. **Rewards and training**: curves under "What changed during training?".
8. **"Refine with Astra"**: results go back to the model, which proposes the next suite.
9. **Honest badges**: "RECORDED · NOT LIVE", "Simulation worker offline".

## Where it was thin (this is our opening)

- Nothing in the demo ran live. The physics was a floating gripper and boxes; the viewer drew a Franka arm in an operating room over it, fitted afterwards. **Ours: the arm on screen is the arm that was simulated**, the real SO-101 model, every pose from the recording.
- The video credits Astra with the scenarios, but the composer never called it; the grid was a fixed parameter sweep. **Ours: the sentence really goes to Astra and really becomes a new station, checked in MuJoCo.**
- 16 scenes, 80 episodes, policies reading simulator state. **Ours: 200 held-out seeds in three tiers, policies that see only cameras and joints, a camera inspector.**
- Codex was never mentioned. **Ours: the build itself is on the page.**

## What the page becomes (page session owns it; others supply the data)

A workspace with these views. Every badge says `live` or `recorded <time>, commit <sha>`.

1. **Station.** First screen. The real SO-101 sorting a scene in 3D, both camera feeds beside it with the inspector's heat map appearing at `present`, the current held-out score per tier.
2. **Scenes.** A wall of all 200 held-out scenes: top-camera thumbnail, tier, "5/8 sorted" badge, title made of real values ("9 bricks · pile · 3 touching · dim light"). Failures first. Filter by tier, failure stage, defect type, checkpoint.
3. **Scene detail.** 3D player with scrubber; phase labels on the timeline (`reach`, `grasp`, `lift`, `present`, `verdict`, `drop`); synced charts of joint positions, gripper force and the inspector's anomaly score with its threshold; both camera videos; Nemotron's sentence on why it failed, the frame it pointed at, and the scenes it asked for next.
4. **Compare.** Pick two controllers (expert, policy checkpoints, DAgger rounds, Nemotron batch against random batch). Three big numbers, fixed and regressed counts with a confidence interval, a per-scene table, and **Play both** on any seed, synced.
5. **Training.** Demo count, loss, score by checkpoint on the held-out seeds, the DAgger rounds, Nemotron batch against random batch. Real checkpoints only.
6. **Inspector.** Confusion table, per-defect recall including the never-seen types, latency on CPU, and **draw your own defect**: draw on a brick, see the render, the heat map and the verdict in under two seconds, then watch that brick go through a rollout.
7. **Build a station.** Type a sentence; Astra writes the MJCF; the check runs (loads, no intersections, reach, cameras); the station appears in 3D with the reasons it passed or was rejected; the expert runs on it. Show the gallery of earlier ones with their sentences and the pass rate.
8. **Built with Codex.** The timeline from `CODEX.md`: five sessions, goal mode, ultra, the subagent tree, reviews and what they caught, models by job. Play a trimmed terminal recording of a session (`/bulk2/runs/dryrun/codex-casts/main-001.cast`, asciinema). Include the first pass that was rejected for faking results; it is the best proof the numbers are real.
9. **What is simulated.** Plain, short, with what Touko can test on the real station.

### Live means live

- **Draw your own defect and on-demand rollouts**: a Modal web endpoint (scales to zero, GPU not needed) that renders the brick, runs the OpenVINO inspector and, on request, a rollout, and returns the replay. The inspector session provides the function, the page session the endpoint and the UI. This also makes a real "Best use of Modal" entry.
- **Build a station**: needs Astra through Codex, which only runs on this box. Use a small job queue in Tiger Cloud: the page inserts a request, a worker on this box (`uv run python -m dryrun.worker`, analyst session) runs Astra, the check and the expert, and writes the result back. The page shows `worker online` or `worker offline` from a heartbeat row, honestly. Limit: 20 requests an hour, sentence at most 300 characters, the sentence is data and never instructions to run commands.
- **Stretch, only after everything above works (page session, a subagent)**: the station running in the visitor's browser with MuJoCo's WebAssembly build and the same MJCF, so they can drive the arm with the keyboard and record a demo, the way teams use the leader arm. If it is not solid by 02:00 UTC, drop it without trace.

## What the other sessions owe the page (agree formats in `docs/interfaces.md` today)

- **main**: a replay and a top-camera thumbnail for every held-out seed and controller, produced by the benchmark itself (`--export-replays DIR`), plus the scene's real values for the card title.
- **inspector**: anomaly score per inspection with the heat map image; the confusion table and latency as JSON; the Modal-ready draw function.
- **policy**: every checkpoint's held-out results as rows (controller id, seed, tier, bricks, correct, events), loss log, dataset size; replays for policy rollouts through the same exporter.
- **analyst**: Nemotron diagnoses as rows keyed by rollout; the station gallery; the worker; `CODEX.md` in a form the page can render (one JSON line per entry in `data/codex_log.jsonl`).

Replays for 200 seeds times several controllers will not fit in the 750 MB database. Keep rows and aggregates in Tiger Cloud; put replay files and thumbnails on a Fly volume or in the image, videos only for about 40 featured rollouts.

The rules do not change: every number from a real run with its commit, nothing staged, no "before" that is not a real earlier checkpoint.
