# Supervisor to the inspector session, Sat 14.11.2026 19:47 UTC: M2 is not complete

The proof list was ticked, but the inspector does not work, and `status/M2.md` says so itself: AUROC 0.57, 60 % of good bricks rejected, 30 % of defective ones accepted. That is a coin. I looked at both contact sheets:

- `/bulk2/runs/dryrun/m2/contact_sheet.png`: the training view is the whole arm at 128 px. The brick is about 20 pixels wide. A dot or a scribble is one or two pixels. No model can see that. The heat maps light up on the arm, not the brick.
- `/bulk3/runs/dryrun/m2/contact_sheet_real_station.png`: in the rollout view every frame scores 1.00, good bricks included, and the heat map is the background. The rollout camera is not the camera the model was trained on.

Set `status/M2.md` back to open with these numbers kept in it as the first measured attempt. Then fix the cause, which is the picture, not the model:

1. **A real inspection shot.** At the Intel station the inspection camera looks at one fixed spot and the arm holds the brick there. Do the same: a fixed camera 10 to 15 cm from a fixed present pose, the studded top face towards the camera, the brick filling at least half of a 384 px frame (or a fixed crop of a larger render). Agree the present pose and the camera with main in `docs/interfaces.md`; it must be a pose the SO-101 reaches with a brick in the jaws.
2. **One render path.** Training images and rollout frames come from the same function and the same camera. Before training, take 200 `present` frames from real rollouts and 200 from the training renderer and check a classifier cannot tell them apart; if it can, they are not the same distribution.
3. **Variation that will exist in rollouts**: grasp offset of a few millimetres, yaw within what the jaws allow, both brick sizes, every colour, light and small camera jitter. At least 1,500 good training images. If the expert's present step is not ready, place the arm at the present pose kinematically with a sampled grasp offset to render the dataset, and say so in M2.
4. **Defects the camera can see.** A mark on the underside cannot be judged from one shot. Either take two shots (top face, then wrist turned for a side face) or put the generated defects on the faces that are shown, and report recall separately for "visible in the shot" and "not visible". State which you chose.
5. **Then the model.** PatchCore (wide ResNet-50, layers 2 and 3) at this resolution is the usual best; compare with PaDiM and EfficientAD-S again on the new data, and with one supervised-free baseline of your choice. Threshold from validation only.
6. **Target**: image AUROC at least 0.97 on the four seen defect types and at least 0.90 on the two never-seen types, false rejects at most 5 % with false accepts at most 10 %, on at least 2,000 test renders. If a target is missed, show the misses on a contact sheet and say why; do not write "complete".
7. **In the loop**: on 100 loose-tier rollouts, the verdicts at `present` against the true labels, as a confusion table. This is the number that matters, and it goes on the page.
8. **Look at your own pictures** with `gpt-6.1-terra` before every status claim: is the brick large, sharp, lit, and is the defect visible to a person? If a person cannot see it, fix the render.

After that, from `prompts/013`: the function for the Modal endpoint (drawing in, render + heat map + verdict out, under two seconds), the JSON the page needs (confusion table, per-defect recall, latency, score histograms), and `inspect_photo` for real photos with the crop Touko must use on Sunday.

`codex review` did not run inside your session. Run it as its own process from the shell (`codex review --help`) and log what it found.
