# Supervisor to policy, Sun 15.11.2026 02:10 UTC: the arm stops 56 mm short. Find out why in 15 minutes, then spend the GPU on the cause

21,588 steps took 16 minutes on the A100, loss 0.99 → 0.33 and still falling, closest approach 110 → 87 → 61 → 56 mm, zero bricks reached. The zero stays on the page as measured. You have until 04:30 UTC for one more real result. Do not train blind again. Three checks first, all with the final checkpoint, run in parallel on Modal:

1. **Training scenes, closed loop.** 20 seeds the policy trained on. If it fails there too, it is underfitting or the test loop differs from the recording (image crop, colour order, normalisation, camera, joint order, action scale, the 0.06 rad step limit cutting the motion short, re-planning before the descent finishes). Compare one training episode tick by tick: recorded action against the policy's action from the recorded observation, per joint, and the image the policy gets in the test loop against the stored frame (mean absolute pixel difference). Print the numbers.
2. **One brick alone.** 20 new scenes with a single brick. Each demo picks the nearest brick of several; a policy that averages over them moves to a point between bricks, and 56 mm looks like that. If it reaches a lone brick, the miss is the choice of brick, not the motion.
3. **Where it stops.** For 20 held-out scenes, the gripper's closest point relative to the nearest brick and to the centroid of all bricks (mm, and height above the table). Say in one sentence where the arm goes.

Then act on what you measured, and start the GPU work at once where it does not depend on the checks:

- A brick is about 7 pixels wide in the 224-pixel top view. Re-record with the top view **cropped to the work area before resizing** (the area the bricks and bowls occupy, same crop in the test loop, 224 or 320 px), and add the inspection or a wrist view only if it is already in the shards. Start this recording now, in parallel shards, while the checks run.
- Train longer: 80,000–100,000 steps (about an hour on an A100 at this speed), checkpoints every 10,000 scored by stage on 20 seeds as they land. Run up to three variants side by side on separate GPUs (for example chunk 30 against 60; KL weight; lone-brick curriculum mixed in), each named and logged. Modal for Dry Run must stay under 20 USD in total; print the spend before and after.
- If check 2 says the choice of brick is the miss, add single-brick demos and record which brick the demo took as a visible cue only if the cue comes from the camera image (never from simulator state at test time).

Page: keep `policy.json` honest and current: the zeros, the closest-approach curve by step (that is a real learning curve; give the page the points), the three check results in plain words, and any stage count above zero the moment it is measured. Last numbers at 04:30 UTC.
