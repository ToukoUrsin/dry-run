# Goal for Codex: Dry Run

This file is the prompt a supervisor gave Codex on Sat 14.11.2026 18:00 UTC. Everything else in this repo is written by Codex. Keep it that way: the claim "built entirely in Codex" must stay true, and no other tool or person writes code here.

Deadline: Sun 15.11.2026 06:00 UTC for a finished, deployed, documented project (submission closes 08:00 UTC). Touko is asleep for most of it. Decide everything reversible yourself. Do not stop to report; write status files instead.

## What we are building

Junction 2026 has a challenge by Ignite Next × Intel, "Stop Simulating, Start Moving". Each team gets one station: an SO-101 robot arm, a leader arm for teleoperation, a top-view camera, an inspection camera, a box of Duplo bricks (the team marks some as defective themselves), and two bowls, "good" and "defective". Teams record a few dozen teleoperated demos, fine-tune a vision-language-action policy in Physical AI Studio on cloud GPUs, train an Anomalib model that tells good bricks from defective ones at the inspection point, optimise both with OpenVINO and run them on an Intel laptop. The arm picks each brick, shows it to the inspection camera and drops it in the right bowl, on its own. Stretch: faster cycles, messier piles, new defect types.

Every team gets one arm, one weekend and a few dozen demos. **Dry Run is that exact station as a simulator**: the place to make ten thousand demos, find where the policy and the inspector fail, and fix them before the real arm moves. We enter it under the Intel challenge on purpose as the sim-first answer, and we aim to win the tech challenges: Best Use of Codex (OpenAI), Nemotron (NVIDIA), Tiger Data, and if a GPU turns up, Verda or Modal.

Reference to beat: Soren (github.com/kevinxie06/soren-oai), a finalist at the GPT-6 Astra SF hackathon. Its weaknesses are our spec: policies saw privileged simulator state (no vision), 16 scenarios and 80 episodes, a hand-picked study instead of a held-out benchmark, "old version" recordings that were staged, generation that only varied parameters. We do the opposite on every point.

## What must be true at the end

1. **The station, in MuJoCo.** The real SO-101 model (official MJCF/URDF from TheRobotStudio/SO-ARM100 or LeRobot, keep the licence), a table, Duplo-shaped bricks with studs (2×2 and 2×4, several colours), two bowls, a top-view camera and an inspection camera. Defects a person could make with a marker, a sticker or a knife: scribble, dot, sticker, chipped stud, missing stud, discolouring. Defects are generated procedurally and seeded. A person can also draw their own defect in the web page and get it on a brick.
2. **Scenes by seed.** Brick count 3 to 12, loose to messy piles, lighting, camera jitter, table surface. Same seed gives the same scene.
3. **Vision only at test time.** The inspector and the policy see camera images and joint states. Never simulator ground truth. The scripted expert may use ground truth, because it is the demonstrator, and the page must say so.
4. **The inspector.** Anomalib trained on renders of good bricks only, exported to OpenVINO, running on CPU, with latency measured. Report false accepts and false rejects, including on defect types it never saw.
5. **Demonstrations.** A scripted expert (inverse kinematics plus a grasp state machine) records demos in LeRobot dataset format, so they load in LeRobot and Physical AI Studio unchanged. Thousands of them, in parallel on this box.
6. **A learned policy.** An imitation policy (ACT or similar from LeRobot) trained on the demos from images, compared with the expert on the same held-out scenes. This box has 80 cores and no GPU: size it to train on CPU in under two hours, and measure before you commit. If `modal` works with hsec `JUNCTION_MODAL_TOKEN_ID` / `JUNCTION_MODAL_TOKEN_SECRET`, use a GPU there; if the workspace is over its limit, stay on CPU and say so in `status/`.
7. **One honest benchmark.** A fixed, held-out set of at least 200 seeds that nothing trains on, plus held-out defect types. Score = bricks in the right bowl / all bricks; also cycle time, drops, false accepts, false rejects. One command re-runs it. Every number shown anywhere comes from a run of that command, with the commit it ran on. Nothing staged, no degraded "before" recordings: a "before" is a real earlier checkpoint.
8. **Nemotron as the failure analyst.** `nvidia/nemotron-3.5-super-vl-120b-a12b` at `https://integrate.api.nvidia.com/v1` (OpenAI-compatible; key: `hsec exec --only NVIDIA_API_KEY -- <cmd>`; about 40 requests a minute, so retry with backoff). It watches the frames of failed rollouts, says why each failed, and writes the next batch of scenes inside a strict schema, so training gets harder where the policy is weak. Measure it: success on the held-out set after Nemotron's batch against after a random batch of the same size. If it does not help, say so and show the numbers.
9. **Tiger Data holds every rollout.** 30 Hz joint telemetry, events and outcomes for every rollout in a hypertable, with continuous aggregates and compression; the dashboard reads only from those aggregates. Start on a local TimescaleDB container; keep the connection in one env var (`DRYRUN_DB_URL`) so it can move to Tiger Cloud when the supervisor gives a URL.
10. **A page a stranger gets in 15 seconds.** Live on a public URL (Fly: `hsec exec --only FLY_API_TOKEN -- fly ...`; pick a free local port first). First screen: the station sorting bricks in 3D with both camera views, and the score. Then: replay any rollout, the score climbing by checkpoint, the failures and what Nemotron said about them, draw-your-own-defect with a live rollout. UI quality bar is linear.app: type, spacing, hierarchy, motion, loading, empty and error states, phone and laptop. Nothing that looks generated.
11. **OpenVINO.** Both models exported and timed on CPU, because that is what the Intel laptop runs.

## Order of work

Write `status/M<n>.md` (what works, the numbers, the commit) and commit when each is done. The supervisor watches for these files.

- **M1** (aim: 2 h): station, scenes by seed, scripted expert sorting with ground-truth labels, benchmark command, one rendered video of a full run.
- **M2**: inspector trained and in the loop; expert + inspector on the benchmark without ground-truth labels.
- **M3**: telemetry in TimescaleDB, page live locally then on Fly.
- **M4**: demos in LeRobot format, learned policy trained and on the benchmark.
- **M5**: Nemotron loop measured.
- **M6**: draw-your-own-defect, OpenVINO timings, polish, docs.

If something blocks a milestone for more than 30 minutes, write the blocker to `status/BLOCKED.md`, go around it and carry on with the next one.

## How to use Codex (this is half the entry)

The Codex judges score how Codex was used. Use these for real and log each use in `CODEX.md` (time, feature, what it did, the session id), because the submission and the film quote that file:

- **Goal mode**: this goal, overnight.
- **Parallel sessions in worktrees**: split independent parts (simulator, inspector, page, data) across `codex exec --worktree` sessions or native subagents, and merge.
- **`codex review`** on every milestone before it is committed; log what it caught and what you fixed.
- **Fast mode** where a person would wait; measure the same task with and without it once and log both times.
- **Models by job**: `gpt-6.1-sol` for the build, `gpt-6.1-terra` to look at renders and screenshots, `gpt-6-luna` for repeatable chores. Log which did what.
- **Image generation** for defect stickers and table surfaces used in the scenes.
- **Browser use**: open the page yourself, at laptop and phone size, and fix what you see.
- **Hooks, skills, MCP, AGENTS.md**: write an `AGENTS.md` for this repo; add a hook or skill only if it earns its place (for example the benchmark smoke test after edits).

Do not claim a feature you did not use.

## Rules

- Never invent a number, a result or a quote. If something was not measured, it is not on the page.
- Say plainly what is simulated. No claim about the real arm unless a real test backs it. Put what Touko could test at the Intel station on Sunday morning (for example photos of real marked Duplo bricks through the inspector) in `MORNING.md`.
- Box hygiene: at most 32 worker processes; `/bulk2` has 18 GB free, so keep run output under 5 GB in `/bulk2/runs/dryrun/` and only text records in git (one `.jsonl` per run).
- Paid API keys are off limits except the ones named here. LLM calls other than Nemotron go through the seat pool (`http://127.0.0.1:8317`, key hsec `CLIPROXY_API_KEY`).
- Plain language everywhere a person reads: short sentences, no jargon without a gloss, no hype.
- Also write: `README.md`, `SUBMISSION.md` (one section per challenge: Intel, Codex, Nemotron, Tiger Data; the Nemotron section must name the model, what it does, why it matters and how its output was improved), `FILM_PLAN.md` (a 2-minute film Touko shoots himself, shot by shot), `MORNING.md` (only what Touko must do).
