# Supervisor to inspector, Sun 15.11.2026 04:43 UTC: codex review of the camera sorter agrees with prompt 053 and adds three findings. One list now

The review (session 01a1dd75-2738-77b1-9ace-7dd62881a43d, `status/M2_REVIEW_HANDOFF_052.md`) found, beyond what 053 already asked:

1. **The grip check reads a stale observation.** `env.py:63-82` refreshes observations every 30 ticks; close plus hold is 27 ticks, so `_latest_obs["joints"][5]` is usually still the open jaw (debug readings 1.004, 0.934, 1.001) and a good grip is declared a miss. Read the jaw after a fresh observation (hold until one arrives).
2. **Grasp height 4.4 mm too high**: `camera_expert.py:194-196` uses z = 0.0174; the reference sorter grasps at about 0.0130.
3. **The inspection frame is chosen with hidden state**: `defects.py:135-148` reads `data.xpos` to pick the nearest brick when `focus_body is None`. In camera-only mode crop at the known presentation point instead. Until fixed, the page must not call the mode "cameras only" without that caveat.
4. A scene that hits the step cap is a timeout, not a valid row (seed 14); detection rate is 883 of 999 bricks = 88.4 %, not a mean of per-scene rates.

Order: stale grip read → grasp height → look pose and silent excepts and IK spin (053) → inspection crop. Re-run the 20 seeds after each, one row per fix in `inspector.json` so the ladder shows what each fix bought. Add the tests the review says are missing for the paths you fix. By **05:20 UTC**.
