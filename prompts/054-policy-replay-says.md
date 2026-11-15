# Supervisor to policy, Sun 15.11.2026 04:24 UTC: I ran your replay script. The loop is fine; the policy is fed something the demos never showed

`MUJOCO_GL=egl PYTHONPATH=. python scripts/replay_recorded_actions.py --dataset /bulk3/runs/dryrun/m4/onebrick-overfit-replay-dataset/onebrick-overfit-20-dataset-20261115-1 --out /tmp/drx/replay_check.json` → **20 of 20 succeed** when the recorded targets go through `rollout.run` with the 0.06 rad clamp. So the action definition, the clamp and the loop are not the bug. Commit the script.

That leaves what the policy sees. Three differences I can read in the code; check them in this order, each is minutes:

1. **The demos start at tick 37.** Your own comment: "The recorder's first frame is after the expert's first 36 approach ticks". The replay only works because it plays 36 expert ticks first. Does the policy test do the same? If the policy starts at tick 0 from the home pose it is in a state that is in none of its 6,020 frames. Either run the same 36-tick prefix before the policy takes over, or record from tick 0. Try the prefix first: it costs nothing.
2. **The inspection picture.** `demos.py:215` records a black 64×64 image whenever no inspection frame is passed. `policy_benchmark.py:28` feeds the live `obs["inspection"]` on every tick. If the demos have black where the test has a picture, the network has never seen its test input. Same question for `verdict`: recorded as −1/0/+1 from the recorder's argument, at test from `obs.get("verdict", [0])`.
3. **Teacher-forced comparison, the decisive one.** During one replay rollout of seed 10020, at every tick: (a) compare the live `obs["top"]`, `obs["inspection"]`, joints and verdict with the recorded frame of the same index (max and mean absolute difference per field; save the first pair that differs as PNGs), and (b) ask the policy for its action on the live observation and on the recorded one, and compare both with the recorded action. If (b)-recorded is 0.02 rad and (b)-live is large, the field that differs in (a) is the bug.

Also odd, worth one line: 4,067 of 6,020 recorded targets are more than 0.06 rad from the current joints, and the common maximum is exactly 1.586. Is that the gripper channel, action in one unit and `joints[5]` in another? If the state and action disagree on the jaw, say so.

Fix what it shows, re-run the 20 one-brick scenes closed loop, then the 600. Stage counts and the sentence into `policy.json` either way, by **05:15 UTC**. If it sorts even one brick, export that replay for the page.
