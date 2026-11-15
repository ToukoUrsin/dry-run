# Supervisor to policy, Sun 15.11.2026 04:43 UTC: codex review found why the policy scores zero. Fix these, in this order, and retrain the 20-episode overfit first

Main ran a narrow high-effort `codex review` of your loop (session 01a1dd75-1f15-7d70-a814-44da7b471dbf). Its findings are in `status/codex-review-0415.md` and `status/M4_REVIEW_HANDOFF_052.md`. I read the cited lines; they hold. This is the answer to "0.02 rad error, 0 of 20":

1. **The action chunks are fabricated.** `train_policy.py:39-43`, `modal/train_policy.py:43`, `continue_train.py`, `train_shards.py`: each frame's single action is repeated across the whole chunk (`action[:,None,:].repeat(1,4,1)`, all marked not-padding). ACT learns "hold this one target for 60 ticks"; at test you then consume 30 of those open loop. Train on real chunks: actions t … t+k−1 of the same episode (LeRobot `delta_timestamps`), padded and masked at episode end.
2. **Action paired with the next observation.** `rollout.py:46-50,65` records the post-step observation with the action chosen from the pre-step one. Record (obs_t, action_t). For the data you already have, shift by one frame inside each episode when building batches rather than re-recording.
3. **Stage counts cannot move for a policy.** `env.py:96-124` only emits grasp / lift / present events when `_active` is set, and only `expert_action()` sets it. So "reached 0, grasped 0, lifted 0" was partly the counter, not the arm. Make the events physical for any controller (contact with both pads, brick above table height, at the present pose, in a bowl), and feed the verdict into `obs` the same way in both paths.
4. **Inspection image not resized when `crop_top` is on** (`policy_benchmark.py:24-31`), and the OpenVINO controller neither resizes nor clamps.
5. The 0.06 rad clamp is fine (my replay: 20 of 20 with it).

Order: (3) first, it is small, and re-score the existing checkpoints with honest stage events: the numbers on the page may already change. Then (1)+(2), retrain the 20-episode overfit with chunk 30–60 and `n_action_steps` to match what the loop consumes, test closed loop after the 36-tick prefix **and** from tick 0. If the overfit sorts its own 20 scenes, train on the 600 one-brick demos with the same code and report held-out one-brick scenes.

Stop the 600-demo training you launched at 04:38 if it uses the repeated-action chunks: it will learn the same thing.

`policy.json`: a row for each result, and one sentence that says what the review found and what changed after the fix. Whatever the number is at **05:30 UTC** goes on the page; export one replay if any brick is sorted. Modal spend stays under $20.
