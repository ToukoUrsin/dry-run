# Supervisor to Codex (inspector session), Sat 14.11.2026 19:00 UTC: you own the defects and the inspector

Read `prompts/001-goal.md` to `prompts/006-split-and-grasp-review.md` and `status/` first. Three Codex sessions work in this directory at once; 006 says who owns what and the rules for sharing it (commit by path only, no `git add -A`, no stash, reset or clean; contracts in `docs/interfaces.md`).

You own: the defects, the inspection-camera renders, the Anomalib model, the OpenVINO export, `dryrun/inspector*.py`, `dryrun/defects*.py`, status file M2. Run output goes under `/bulk2/runs/dryrun/m2/` (keep it under 2 GB).

## Where it stands

`status/M2.md`: a PaDiM model trained on good bricks, an OpenVINO export, a 580-image evaluation. First find out what those 580 images really are and say so in the status file. If they are clean bricks alone in front of a camera, the numbers do not describe the station.

## What to build

1. **Defects a person could make at the real station** with a marker, a sticker or a knife: scribble, dot, sticker, chipped stud, missing stud, discolouring. Procedural and seeded, on Duplo 2×2 and 2×4 bricks in several colours. Marker and sticker defects are textures; chipped and missing studs are geometry. Use Codex image generation for sticker art and table surfaces and log it in `CODEX.md`. Agree with the main session in `docs/interfaces.md` how a scene asks for a defective brick.
2. **Inspection renders as the station sees them.** The brick is held in the SO-101 gripper in front of the inspection camera, as in a rollout: jaws partly covering it, the pose a little different every time, lighting and camera jitter by seed. Render with MuJoCo from the real station model. Good bricks for training; a separate set for choosing the threshold; a test set that nothing else touches.
3. **The model.** Anomalib, trained on good bricks only. Compare at least three (for example PaDiM, PatchCore, EfficientAD) on image AUROC and on CPU latency after OpenVINO export, and pick one for a stated reason. Choose the threshold on the validation set, never on the test set.
4. **Honest numbers.** A confusion table on at least 2,000 test renders; false accepts and false rejects; recall per defect type; and separately the defect types the model never saw in any form, not even when the threshold was chosen. Say which defects it misses and show examples with heat maps. If small dots on the hidden side of a brick are missed because the camera cannot see them, say that; then decide with the main session whether the arm should turn the brick.
5. **OpenVINO on CPU.** Latency at 1 and 4 threads, median and 95th percentile, on this box, with the exact OpenVINO version. This is the part an Intel laptop runs.
6. **One class to call.** `dryrun.inspector.Inspector` loads the OpenVINO model once and returns score, verdict and heat map for an RGB frame. The benchmark calls it to choose the bowl (no ground truth); the page calls it for "draw your own defect". Also a function that takes a person's drawing (PNG with transparency), puts it on a brick face, renders that brick at the inspection point and returns the frame. Write both in `docs/interfaces.md`.
7. **The real thing on Sunday morning.** `uv run python -m dryrun.inspect_photo PHOTO` runs the same model on a phone photo of a real Duplo brick. The model was trained on renders, so say plainly what to expect, and do what is cheap to narrow the gap (randomised lighting, backgrounds, blur and noise in the renders). Write the steps for Touko in `MORNING.md`: what to photograph, how, and the command.

## Proof

`status/M2.md` lists the commands and their output: dataset sizes, the model comparison, the confusion table, per-defect recall, held-out types, latency, and the path of a contact sheet with examples (good, each defect type, misses, heat maps). Look at the contact sheet yourself with `gpt-6.1-terra` before you write the status file: defects must be visible and look like something a person did. Run `codex review` before each commit of real work and log what it found.
