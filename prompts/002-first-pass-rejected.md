# Supervisor to Codex, Sat 14.11.2026 18:05 UTC: the first pass is rejected

Three minutes in, `status/M1.md`, `M2.md`, `M4.md`, `M5.md` and `M6.md` say milestones are complete. They are not. The repo is 245 lines. There is no MuJoCo in it (`import mujoco` fails on this box), no SO-101 model, no physics, no rendered image, no Anomalib, no trained policy, no Nemotron call, no database. A benchmark that scores 1.000000 on a "stand-in" is not a result. Status files that say "complete" for stubs break the first rule of this project: never claim what was not done.

Do this now:

1. Delete every status file. Fix `README.md`, `SUBMISSION.md`, `CODEX.md` and `FILM_PLAN.md` so they claim nothing that does not exist yet. Commit that as what it is.
2. Tell every subagent the same rule: a stand-in, placeholder, fallback or interface "ready for" something does not count. If the real thing is missing, the milestone is open.
3. Work the milestones in order. You have until 06:00 UTC, about twelve hours. Spend them. M1 alone is expected to take one to two hours of real work.

A milestone is done only when its proof exists. The supervisor checks each proof by running it.

**M1 proof**
- A `uv` environment in the repo with `mujoco` installed; `uv run python -c "import mujoco"` works.
- The real SO-101 model, downloaded from its official repo with meshes and licence, loads in MuJoCo and its joints move under actuator control. No placeholder base.
- Bricks are rigid bodies with stud geometry and collision. The gripper picks them up by friction contact in physics. No teleporting, no attaching bodies by weld, no setting object positions.
- `uv run python -m dryrun.render --seed 7 --out /bulk2/runs/dryrun/m1/seed7.mp4` writes a real video rendered by MuJoCo (EGL or OSMesa) with three views: free camera, top-view camera, inspection camera. The arm picks each brick, holds it in front of the inspection camera and drops it in a bowl.
- The benchmark runs the physics for every seed and reports the true score, whatever it is. A scripted expert under real physics will fail on some messy piles. Report those failures; do not hide them. A perfect score on the first try means the test is too easy or the physics is fake.
- Look at frames from the video yourself (use `gpt-6.1-terra` for it) and fix what looks wrong before writing the status file.

**M2 proof**: Anomalib installed and trained on rendered inspection-camera images of good bricks only; the model file exists; an OpenVINO export exists and its CPU latency is measured; a confusion table on at least 500 rendered bricks, with held-out defect types listed separately; the benchmark runs with the inspector deciding the bowl.

**M3 proof**: a TimescaleDB container running, a hypertable with real 30 Hz joint rows from real rollouts (give the row count), continuous aggregates and compression set up, the page reading from them; the page live locally, then on Fly, with a 3D replay of a real rollout.

**M4 proof**: a LeRobot-format dataset on disk that `lerobot` loads; a training log with a falling loss; a checkpoint file; the learned policy's score on the held-out seeds next to the expert's, from images and joint states only.

**M5 proof**: saved request and response pairs from the real Nemotron API on frames of real failed rollouts; the scenes it wrote; the measured comparison with a random batch.

**M6 proof**: the items in the goal, each one run.

Each status file lists the exact commands that prove it and their output. If a proof cannot be met, the status file says "open" and why.

On effort: this is hard engineering (contact physics, inverse kinematics, grasping). Use high reasoning effort for the simulator and the expert. Speed comes from parallel sessions, not from stubs.
