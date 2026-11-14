# Supervisor to the policy session, Sat 14.11.2026 19:30 UTC

Read `prompts/001-goal.md`, `prompts/005-gpu-on-modal.md`, `prompts/006-split-and-grasp-review.md`, `prompts/010-ultra-and-five-sessions.md` and `docs/interfaces.md` first. You own M4: the demonstrations and the learned policy. You own `dryrun/demos*.py`, `dryrun/policy*.py`, `dryrun/train*.py`, `dryrun/dagger*.py`, `modal/` and `status/M4.md`. The four-episode dataset and the 36-line trainer in the repo are a stub; replace them.

## What must exist

1. **Demo recorder.** Hooks into `dryrun.rollout.run(..., record=...)`. Writes a real LeRobot dataset (current format, loads with `lerobot` unchanged, and so in Physical AI Studio): `observation.images.top`, `observation.images.inspection`, `observation.state` (six joints), `action` (six joint targets), 30 Hz, a task string, success flags per brick cycle. Only successful cycles go into training. Start now on the `loose` tier with whatever the expert can already do (single bricks lift 10/10), and regenerate as main improves. Until main's API lands, build against the existing `grasp_test` loop and switch over.
2. **How the policy learns which bowl.** The policy never sees a defect label. After the `present` step the inspector's verdict (from the camera image) is fed to the policy as one extra input (0 before inspection, +1 good, -1 defective), the same way the real station would pass Anomalib's output. Say so on the page and in M4.
3. **Training on Modal GPU.** ACT from LeRobot (action chunks, two cameras). Save a checkpoint every N steps to a Modal volume and copy them here; the page's "score by checkpoint" curve comes from those real checkpoints. Log loss to a `.jsonl`. Measure one short run first to size the rest inside the budget (15 dollars).
4. **The policy on the benchmark.** `uv run python -m dryrun.benchmark --controller policy --checkpoint PATH --seeds 200` through the same rollout code as the expert, images and joints only, closed loop. Per tier, next to the expert.
5. **DAgger, because a simulator can.** Run the policy, let the scripted expert label the right action in every state the policy reached, add those to the dataset, retrain. One real arm in one weekend cannot do this; ten thousand simulated ones can. Measure the held-out score after each round: demos only, +DAgger round 1, +round 2. This is the main "score climbing" story, so every point on it must be a real checkpoint scored on the held-out seeds.
6. **Scenes on request.** The analyst session will hand you scene batches (a `.jsonl` of scene specs) chosen by Nemotron and a random batch of the same size. Train one policy on each from the same starting checkpoint and report both held-out scores. Agree the file format with the analyst in `docs/interfaces.md` now.
7. **OpenVINO.** Export the final policy, time it on CPU at 1 and 4 threads, and run the benchmark once with the OpenVINO model to show the score holds.

## Order

19:30 to 21:00: recorder, 500 loose demos, first short training run, the policy stepping in the benchmark loop (any score). 21:00 to 23:00: scale demos (thousands, on Modal), first real policy scored on all 200 seeds. After that DAgger rounds and the Nemotron batch comparison until 03:00.

If ACT does not learn the full cycle in time, do not hide it: report what it does learn (reach, grasp, lift, present, drop rates per stage) and keep the full-cycle number next to it. A smaller honest result beats a claim.

Proof in `status/M4.md`: the exact commands, dataset size, training log, checkpoints, per-tier scores for each checkpoint with the commit. Log Codex features used in `CODEX.md`. `codex review` before each commit.
