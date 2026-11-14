# Supervisor to the analyst session, Sat 14.11.2026 19:30 UTC

Read `prompts/001-goal.md`, `prompts/003-astra-for-3d.md`, `prompts/006-split-and-grasp-review.md`, `prompts/010-ultra-and-five-sessions.md` and `docs/interfaces.md` first. You own M5 and three more things. Your paths: `dryrun/nemotron*.py`, `dryrun/curriculum*.py`, `dryrun/stations*.py`, `status/M5.md`, and the docs (`README.md`, `SUBMISSION.md`, `FILM_PLAN.md`, `MORNING.md`, `CODEX.md` upkeep). The 30-line `nemotron.py` that sent one frame is a stub; replace it.

## 1. Nemotron as the failure analyst (the NVIDIA entry)

Model `nvidia/nemotron-3.5-super-vl-120b-a12b`, `https://integrate.api.nvidia.com/v1`, key `hsec exec --only NVIDIA_API_KEY -- <cmd>`, about 40 requests a minute, retry with backoff. Save every request and response.

- **Input per failed rollout**: 6 to 8 frames around the failure from the top and inspection cameras, and nothing from the simulator's ground truth. There are thousands of failed rollouts already (the expert on piles); the policy's failures come later.
- **Output, strict schema, validated, retried when invalid**: failure stage, cause in one plain sentence, which frame shows it, the scene features that made it hard, and a batch of next scenes as scene specs the generator accepts (tier, brick count, spacing, yaw spread, positions, lighting).
- **Score the diagnoses.** The simulator's event log knows what really happened (`ik_fail`, `slip`, `miss`, knocked a neighbour, wrong bowl, inspector false accept). Build a labelled set of at least 300 failures and measure Nemotron's accuracy on stage and cause. Then improve it and measure again: frame selection, both cameras against one, a short written chain of observations before the verdict, few-shot examples. Show the table "version, change, accuracy". NVIDIA asks how the model's output was improved; this table is the answer.
- **Does its curriculum help.** Nemotron's scene batch against a random batch of the same size, both given to the policy session for training from the same checkpoint, both scored on the held-out seeds. Until a policy exists, run the same comparison on the expert's tuning set so the machinery is proven. If it does not help, say so with the numbers.
- **Store** every diagnosis in the database through the page session's telemetry module so the page shows the frame, the sentence and the next scenes for any failure. Agree the table with the page session in `docs/interfaces.md` now.

## 2. Stations from a sentence (Astra)

As in `prompts/003`: a plain sentence becomes a complete valid MJCF station, checked automatically (loads, nothing intersects, reach by inverse kinematics, cameras see their targets), rejected ones kept with the reason. At least 30 sentences, the pass rate reported, and the expert run on the accepted ones through main's rollout API. Main already has a first version in `dryrun/mjcf_station.py`; take it over from there without breaking main's use of that file (add your code in `dryrun/stations*.py`).

## 3. The docs

`CODEX.md` is half the Codex entry: every feature used, when, by which session and model, what it did, with session ids. Collect entries from all five sessions every hour. From 03:00 UTC write `README.md`, `SUBMISSION.md` (one section each: Intel, Codex, Nemotron, Tiger Data, Modal), `FILM_PLAN.md` (two minutes, shot by shot, Touko films it himself) and `MORNING.md` (only what Touko must do, including photos of real marked Duplo through the inspector at the Intel station). Every number in them comes from a status file.

Proof in `status/M5.md` with exact commands and outputs. `codex review` before each commit.
