# Sunday station checklist

1. Put the Intel station on a stable table and photograph one known-good red,
   blue, yellow and green Duplo brick.
2. Hand-mark one brick for each of scribble, dot, sticker, chipped stud,
   missing stud and discolouring. Photograph each mark clearly. Photograph the
   same bricks at the inspection point with the full brick in frame; use diffuse
   light and no flash.
3. Copy the photographs to the box and run the inspector on each one:

   ```bash
   # X Y W H is the tight crop around the visible studded top face.
   uv run python -m dryrun.inspect_photo PHOTO.jpg --crop X Y W H --out PHOTO.result.json
   ```

Replace `PHOTO.jpg` with the actual copied filename and replace `X Y W H` with the crop rectangle in source pixels. Crop tightly around the visible studded top face, keeping the whole brick and no arm or bowl. The command writes the
JSON result and a sibling `PHOTO.result_heatmap.png`; it can be repeated once
per photo, including photos in a new output directory.

   Save the verdict and heat map, and note which face was visible. Turn a
   brick before relying on a verdict about a hidden face.
4. Capture both station cameras and run one teleoperated demonstration. Record
   the real cycle time beside the simulator replay.
5. If a Tiger Cloud URL has been supplied, set `DRYRUN_DB_URL` and run the
   telemetry setup/sync command from `status/M3.md`. Record the resulting
   `/api/score` and `/api/benchmark-tiers` payloads with their source commit;
   keep those simulator measurements separate from the real-arm check.
6. Photograph the real arm sorting a held-out pile. Keep the photos and the
   inspector outputs with the run record.

Dry Run makes no real-arm performance claim until these checks have been run.

## Cold-open match cut

Use `/bulk3/runs/opens/final/dryrun/open.mp4` for the first 10 s of the film's cold open, then cut to the phone shot for its final 2 s. End the open on the arm and one lime brick alone on black, framed like `/bulk3/runs/opens/final/dryrun/last.jpg`.

**Shooting note**
1. Camera: phone on the table, lens about 30 cm above the table top and about 1.25 m from the arm's base, turned
   about 37 degrees from a pure side view toward the gripper end, 2x lens (50 mm equivalent), tilted about 8 degrees
   down. In the frame: the base a quarter in from the left, the brick low and just right of centre, as in `last.jpg`.
2. Object: upper arm straight up, forearm level, fixed jaw pointing forward, moving jaw open straight up. One brick
   on the table 30 cm in front of the centre of the base and 4 cm to the far side. If the arm cannot hold that pose,
   keep the camera spot and the brick: the cut still lands on the brick.
3. What moves first: nothing in the open's last half second, so hold the phone shot still for half a second, then the
   first move is the real one (the arm, or the hand with the marked box).

If Touko dislikes the sound, mute `open.mp4` while keeping its type; `open_clean.mp4` has no type and no audio.
