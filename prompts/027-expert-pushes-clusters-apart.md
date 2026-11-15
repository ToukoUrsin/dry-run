# Supervisor to main, Sun 15.11.2026 00:52 UTC: second reference expert. Touching bricks 44 % → 94 %, pile 8 % → 59 %

The page's story is the tier gradient, and "messier piles" is the Intel stretch goal. The current expert gives up as soon as no brick has a clear grasp (almost every failure in the touching and pile tiers is `no_clear_grasp`). I extended the reference outside the repo, same way as prompt 021, and ran it through your `StationEnv` at `ad4d006` on seeds 2000–2023:

| Tier | Current expert (your held-out run) | `expert_v2` (my seeds 2000–2023) |
|---|---|---|
| loose | 382/382 | 126/127 (one brick flung out at release, the known 1-in-900 case) |
| touching (messy) | 217/494 = 43.9 % | 153/162 = 94.4 % |
| pile | 31/386 = 8.0 % | 116/196 = 59.2 % |

No `unstable` event in any of the 72 episodes. Different seeds, so your held-out run is the number that counts.

`prompts/027-reference/expert_v2.py` is a drop-in for `dryrun/expert.py` (class `Expert`, same constructor, `action()`, `events()`, `phase`, `current`). `prompts/027-reference/run_reference.py` is the loop that scored it (`EXPERT=expert_v2 python run_reference.py REPO SEED0 SEED1 TIER`). What changed against prompt 021:

1. **Pushes.** When no brick has a clear grasp, the closed gripper pushes a blocked brick in a straight line at TCP height 16 mm, then replans. At most 12 pushes per episode. Event `push` (brick, reason `blocked`); phases `push_reach`, `push`, `push_abort`. Pushes that go sideways or towards the base are tried before pushes that go outward, because outward pushes shove bricks out of reach. Start point needs 32 mm of free table.
2. **Never squash a brick.** While the hand comes down (to push or to grasp), it stops and backs off if the hand-to-brick force passes 1 N (push) or 3 N (grasp). That is what removed every `unstable` event. A stopped grasp is a `miss` with reason `blocked_descent`.
3. **Bricks that are not flat.** A brick resting on any face can be gripped if a horizontal axis is 40 mm or less wide and it lies no higher than 35 mm.
4. **Softer release.** The jaw opens from where it actually is, over 0.5 s, so the grip force fades before the pad moves. `Q_CLOSE` is 0.2 (the 1 N·m limit still sets the force). Three tries per brick instead of two.

Do this, in this order:

1. Port it as it is (rename the class as before). Run your own tests. Do not re-derive or retune it.
2. Re-run the held-out tiers with ground-truth labels: 80 loose, 70 touching, 50 pile, the same seeds as the table in `status/M1.md`. Put old and new side by side in `status/M1.md` with both commits. The old row stays: it is the honest "before" and the page shows both.
3. If loose drops below 99 %, say so with the seeds and keep the old expert for loose only; tell me in `status/EXPERT_NOTES.md`.
4. `/api/benchmark-tiers` serves the new numbers and the old ones (the page session reads it; tell them the field names in `docs/interfaces.md`).
5. Export one touching-tier replay and one pile replay with pushes in them for the page (seeds where the score is high but not perfect are the honest pick), and re-export the M1 video with a push in it.
6. Add `push` to the event list in `docs/interfaces.md`. The policy and analyst sessions need to know about it: a `push` is not a failure.
7. Known limits, write them down as they are: about 40 % of pile bricks still fail, mostly because a dense pile moves as one lump when pushed and bricks end beyond 0.26 m from the pan axis where a top-down grasp cannot reach. Do not hide it and do not move the pile to make it easier.

`CODEX.md` and `README.md` already say the grasp cycle was prototyped by the supervisor outside Codex. Add this file to that line.

Also still open from prompt 022 and your own list: the inspection camera post is drawn in the middle of the work area (move it to where it does not stand between the arm and the bricks, the camera pose itself must not change), `codex review` on the M1 diff (if the Blender MCP approval blocks it, start `codex review` with MCP servers disabled), and the fling at release if it still shows up after change 4.

Target: step 2 numbers committed by 01:40 UTC.
