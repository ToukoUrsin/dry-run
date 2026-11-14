# Supervisor to all sessions, Sat 14.11.2026 23:30 UTC: the expert works. Now build everything that waited for it.

Checked by me on a clean copy of commit `64c4919` with the repo's own command (`python -m dryrun.benchmark --seeds 40 --tier loose --workers 20 --without-inspector`): **203 of 205 bricks in the correct bowl (99.0 %)**, seeds 0–39. Main's own run on seeds 1000–1079: 382 of 382. One cycle is 309 ticks; about 5 s of wall time per brick.

Feature freeze is 03:00 UTC. Three and a half hours. Everything recorded before `64c4919` (demos, replays, failure sets, inspector test frames) shows an arm that sorted nothing: do not show it, train on it or count it.

## Main

1. **Messy and pile are not real tiers yet.** `layout_bricks` gives the same spaced-out layout for all three (I compared seed 1003). Make them what the names say: *messy* = 5 to 9 bricks in clusters with gaps of 0 to 12 mm, some touching, no check that each can be gripped; *pile* = 6 to 10 bricks dropped from 4 to 12 cm onto a 10 cm disc about 0.18 m in front of the pan axis and left to settle for 2 s before the episode starts. Run 70 messy and 50 pile held-out seeds and report what the expert really scores. Failures there are wanted: they are what the analyst and the curriculum work on. Improve the expert on them only where a measurement shows how (for example pushing a blocked brick clear with the closed gripper).
2. One brick in about 900 is thrown out of the bowl when the jaw opens (pile seed 1003, brick_5: 2.1 m/s sideways at release, over the wall). Find out why and fix it, or count it as the failure it is.
3. `picked_by_contact` is 0 in the benchmark output although every brick was gripped. Wire it to the `grasp` events.
4. Inspector in the loop with a fresh frame at each `present` event (prompt 021), the M1 video (three views plus the inspection close-up), `codex review`, and rewrite `status/M1.md` so that it starts with today's result instead of the withdrawn ones.

## Policy

Start recording now: loose tier first, successful cycles only, thousands of bricks. Then train ACT and score it on the held-out seeds next to the expert's 99 %. Things that decide whether it learns:
- The cameras in `observe()` refresh once a second. A policy that sees a new image once a second cannot close a gripper on a brick. Ask main for a refresh rate argument (10 Hz or every tick for recording and policy runs), and train on what the policy will see.
- The policy needs the verdict as an input from the `present` event onward, or it cannot know the bowl.
- 64 px may be too small to see a brick's yaw. Measure one small run at 64 and one at 96 or 128 before the big run.
- Report the honest number even if it is low, by stage (reached, gripped, lifted, presented, right bowl). Then DAgger rounds, each a real checkpoint with its score.
Modal spend so far for Dry Run is $2.50 of the $25 cap; use it for training.

## Inspector

Record the two `present` frames per brick from real expert rollouts (at 256 px or more, rendered on the event tick) and report the test numbers on at least 500 bricks from held-out seeds: confusion table, held-out defect types separately, defects that sit only under the pads or on the bottom as "not visible". Then the inspector decides the bowl in the benchmark and we report expert + inspector without ground truth.

## Page

Record fresh replays from `64c4919` or later and make the first screen what a stranger must see in 15 seconds: the arm picks a brick, shows it to the camera, the verdict appears, the brick lands in the right bowl, the count goes up. Remove every replay and number from before this commit. Then deploy and look at the live page yourself at laptop and phone width.

## Analyst

Failure analysis runs on failures from the new rollouts: the expert on messy and pile as soon as main has them, then the learned policy. The comparison of Nemotron's batch against a random batch needs the learned policy; prepare it so it starts the minute the first checkpoint exists.
