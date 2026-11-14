# Supervisor to Codex (main session), Sat 14.11.2026 19:00 UTC: the work is split three ways, and a review of the grasp trials

One hour in, the repo is under a thousand lines and one session is doing everything in turn. That will not finish by 06:00. From now on three Codex sessions work in this directory at the same time.

## Who owns what

- **You (main):** the station, scenes by seed, the scripted expert, the benchmark, demos, the learned policy, the Nemotron loop, stations from a sentence. Files: `dryrun/mjcf_station.py`, the expert, `dryrun/benchmark.py`, `dryrun/make_dataset.py`, `dryrun/train_policy.py`, `dryrun/policy_benchmark.py`, `dryrun/nemotron*.py`. Status files M1, M4, M5.
- **Page session** (`prompts/007-page.md`): everything in `web/`, the replay export, telemetry (`dryrun/telemetry.py`, `data/schema.sql`), the Fly deploy. Status files M3 and M6.
- **Inspector session** (`prompts/008-inspector.md`): defects, inspection renders, Anomalib, OpenVINO, `dryrun/inspector*.py`, `dryrun/defects*.py`. Status file M2.

Rules for sharing one directory:

- Commit only your own files, by path (`git add <path>`). Never `git add -A`, never `git stash`, `git reset --hard`, `git checkout -- .` or `git clean`: they destroy the other sessions' work.
- Shared contracts (function names, file formats) live in `docs/interfaces.md`. Read it before you call another session's code; write your part of it before they need it. The two you owe first: how to build a scene and step the expert for a seed (the page records replays with it, the inspector renders bricks with it), and where the benchmark calls the inspector and the telemetry writer.
- If you need a change in someone else's file, write it under "Requests" in `docs/interfaces.md`.

## Review of `grasp_trials_varied.jsonl` (7 of 10 lifted)

1. **The three misses are not the grip.** In all three the inverse kinematics did not converge (error 0.13 m to 1.8 m) and the expert moved anyway. One run ended with the brick 234 m below the table: the simulation blew up. Fix: if the solve fails, do not move; try another approach yaw (the brick is symmetric, so yaw and yaw + 90° are both valid for a 2×2, yaw and yaw + 180° for a 2×4); clamp every joint target to its limits; treat any body below the table or any speed above a sane limit as a failed rollout, stop it and record it.
2. **Measure where the arm can reach.** Run the single-brick test over the whole table (a few hundred positions and yaws, 32 workers in parallel, it takes minutes on this box) and save the reach map. Scenes by seed must spawn bricks, bowls and the inspection point inside the measured reach, like a person would lay out the real station. The reach map goes on the page.
3. **The first solve reports `ik_ok: false` in all ten runs** at 3.4 mm error while later ones pass at 1.5 mm. Check the tolerance and what that first stage is.
4. **After the lift the pad is 33 mm from its target every time.** The arm sags or stops short. That will matter at the inspection camera and over the bowl. Find out which (gain, gravity compensation, or the target itself).
5. **A lift is one third of a cycle.** Measure each stage on a single brick over the reach map: grasp, lift, carry to the inspection point and hold still for the camera, carry to the bowl, release into the bowl, brick at rest inside the bowl. Report the rate per stage. Then piles: take the brick that is easiest to reach first, plan again from the simulator state after every pick, retry a failed grasp twice, give each brick a time limit.
6. Use the box. Every experiment that loops over seeds or positions runs in parallel (up to 32 workers). An experiment that takes ten minutes serially is a wasted ten minutes.

`prompts/005-gpu-on-modal.md` still stands for the policy.
