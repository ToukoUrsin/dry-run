# Supervisor to inspector, Sun 15.11.2026 04:21 UTC: the camera-only sorter loses in its loop, not in its eyes. Three things I read in your sweep and in `dryrun/camera_expert.py`

`camera_stage_sweep_20_focus.json`, 20 loose scenes, 100 bricks: 5 in the right bowl. Per seed:

- **11 of 20 scenes: seen 0, events 0, `observations: 3`.** The sorter looked three times, saw nothing, and declared itself done. Your static test detects 88 % of bricks, so in the loop the detector is getting something else. `_refresh` has `except Exception: return []` twice over: any error (image size against calibration, `joints` shape for the arm mask, a missing key) looks like an empty table. Remove the silent excepts, count and log what is raised, and save the first `obs["top"]` the loop sees for seed 0 next to the frame your static test uses for the same seed. Find the difference.
- **Seed 14: 1,110 events in 1,110 steps, seen 1,110, reached 0.** Every step it picks the detection nearest the origin, IK fails, and it picks the same one again. Try the detections in order until one is reachable; remember tokens that failed and skip them; an IK failure must cost time (a short hold), not spin.
- **Scenes that sort one brick then stop (seeds 3, 8, 10, 15, 19; seed 7 managed two).** After the drop the arm returns to `qn` = (0.22, 0, 0.14), which is over the middle of the table, and looks from there. The arm and its mask cover the bricks; three misses and it is done. Look from a pose that is out of the top camera's view of the table (where the reference sorter parks), and do not give up after three looks while the last look was taken with the arm over the table.

Also: grip misses in seeds 2, 4, 18 (reached, not gripped) → print the camera estimate against the true pose for those three in the debug output only (truth never enters the sorter) and say whether it is position, yaw or height.

Fix in that order, one commit each, re-run the same 20 seeds after each fix so the page ladder shows which fix bought what. Report: seen / reached / gripped / presented / right bowl of 100, per fix. The false rejects (2) come after. By **05:10 UTC**; `inspector.json` updated each time. Do not touch `web/`.
