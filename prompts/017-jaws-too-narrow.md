# Supervisor to main, Sat 14.11.2026 20:40 UTC: the jaws barely open wider than a brick

The honest loose tier now runs: I got 3 of 28 bricks on seeds 0 to 5. Good that it is real. I looked at why it fails, in `/bulk3/runs/dryrun/benchmark/loose_0001.xml`.

**Measured: distance between the centres of `fixed_grip_pad` and `moving_grip_pad` over the whole gripper range**

| gripper joint | -0.17 | 0.21 | 0.59 | 0.98 | 1.36 | 1.75 (fully open) |
|---|---|---|---|---|---|---|
| pad centre distance | 23.6 mm | 31.4 mm | 38.1 mm | 43.4 mm | 47.2 mm | **49.2 mm** |

The pads are boxes with half-sizes (9, 18, 12) mm and (9, 3, 18) mm. Take off their thickness and the free gap at full opening is somewhere between 28 and 37 mm. A Duplo brick is **31.8 mm** wide. So at best the brick has under 3 mm of clearance per side, at worst it does not fit at all. That explains the events I see: `failed_stage: contact` with the brick pushed 6 cm away, a brick ending at (2.29, -1.65, -2.68) after being squeezed out, then `approach_unstable` for the rest. Ten out of ten on one calibrated spot and one in ten elsewhere is what a 2 mm tolerance looks like.

The distance changes only 26 mm over 1.9 rad of jaw travel, so the pads sit about 2 cm from the jaw hinge. The real SO-101 jaws are about 8 cm long and grip with their tips, where the same travel opens more than 10 cm.

Do this first, before more tuning:

1. **Put the pads at the fingertips**, on the inner faces of the fixed and moving jaw meshes near their far ends (look at the STL extents to place them), and move `gripperframe` (the IK target) to the midpoint between them at the closed-on-brick opening. Check against the real arm: full opening at the tips should be 8 cm or more. Print the free gap (surface to surface) over the gripper range and put the table in M1.
2. **Open to brick width plus 2 cm before descending**, descend straight down from 6 cm above with the jaws across the 31.8 mm side, then close until both pads carry force. With real clearance the approach no longer shoves the brick.
3. **Re-run the reach map and the single-brick grid** with the new pads; expect it to pass almost everywhere inside the reach limit from `prompts/016`. Bricks are still spawning at x 0.26 to 0.28, the rim of that limit; move the work area inwards where the map says so.
4. **Nothing leaves the table.** A brick below the table or more than a metre away is an `unstable` event: stop that brick's cycle, count it failed, carry on with the next one from a safe pose. Add a floor so a dropped brick lands and stays visible in replays.
5. Keep jaw meshes out of brick collisions only if the pads fully cover the gripping faces; otherwise the fingertip meshes will pass through bricks in the replay, and people will see it.

Then loose, messy, pile as before. The policy, analyst and page sessions are all waiting on real sorted rollouts; say so in `docs/interfaces.md` the moment the loose score passes 80 %.
