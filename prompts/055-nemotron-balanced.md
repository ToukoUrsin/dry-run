# Supervisor to analyst, Sun 15.11.2026 04:33 UTC: 049 is done and checked. Last job: give Nemotron a test it can pass or fail honestly

Checked: `codex.json` has `session_ids`, `review`, `fast_mode`, `authorship`; `SUBMISSION.md` is 413 words; prompts are committed. Good.

The Nemotron number is the weakest thing on the page: 43.7 % where always answering "no clear grasp" scores 98.3 %. That test cannot tell a good reader from a bad one because 295 of 300 failures are one kind. The honest fix is a test set with several kinds of failure in it, not a kinder score.

1. **Balanced set from failures we already have.** Sources: the first sorter's touching and pile runs (44 % and 8 %: missed grasps, pushes, drops), the camera-only sorter's 20-scene sweep (`/bulk3/runs/dryrun/m2_rebuild/camera_stage_sweep_20_focus.json`: never saw a brick, reached but did not grip, wrong bowl), the learned policy's rollouts (never reached), and the current sorter's pile failures. Use four stages a person can tell apart from four frames: **never touched a brick · touched but did not lift · lifted then lost it · put it in the wrong bowl**. Labels come from simulator events only. Up to 25 per stage, the same number in each stage as far as the data allows; say the count per stage.
2. Same 448 px four-panel sheets, same model, strict JSON, one call per failure, cached. Report: accuracy, per-stage table, confusion, and the two baselines (always the largest class; chance = 1 over the number of stages). Write the sentence that is true, whichever way it falls.
3. `analyst.json`: add `balanced` beside `final` (do not delete the 300-failure result; it stays as "what the station's own failures look like"). `headline` = the object the page should show first, with `n`, `accuracy`, `baseline`, `sentence`. Four examples with their sheets, one per stage, at least one wrong.
4. If by **05:20 UTC** the balanced run is not finished, stop, keep the current headline and say so in `status/M5.md`. Do not edit `web/`.

Then, until 05:50: keep `night_data.py` fresh every 10 minutes, and re-read `CODEX.md` top to bottom once as a judge with six minutes: the first screen of the file must be the five sessions, the timeline, the seven rejections and where to open each piece of evidence. Move anything else below.
