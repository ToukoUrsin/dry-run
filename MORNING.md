# Sunday station checklist

1. Put the Intel station on a stable table and photograph one known-good red,
   blue, yellow and green Duplo brick.
2. Hand-mark one brick for each of scribble, dot, sticker, chipped stud,
   missing stud and discolouring. Photograph each mark clearly. Photograph the
   same bricks at the inspection point with the full brick in frame; use diffuse
   light and no flash.
3. Copy the photographs to the box and run the inspector on each one:

   ```bash
   uv run python -m dryrun.inspect_photo PHOTO.jpg --out PHOTO.result.json
   ```

Replace `PHOTO.jpg` with the actual copied filename. The command writes the
JSON result and a sibling `PHOTO.result_heatmap.png`; it can be repeated once
per photo, including photos in a new output directory.

   Save the verdict and heat map, and note which face was visible. Turn a
   brick before relying on a verdict about a hidden face.
4. Capture both station cameras and run one teleoperated demonstration. Record
   the real cycle time beside the simulator replay.
5. If a Tiger Cloud URL has been supplied, set `DRYRUN_DB_URL`, run the
   telemetry setup/sync command from `status/M3.md`, then rerun the benchmark
   command shown there and save its commit.
6. Photograph the real arm sorting a held-out pile. Keep the photos and the
   inspector outputs with the run record.

Dry Run makes no real-arm performance claim until these checks have been run.
