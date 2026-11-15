# Supervisor to inspector, Sun 15.11.2026 03:00 UTC: the camera-only sorter fails on yaw. Fix that one thing

Your own numbers (`perception_eval_200.json`): position median 1.7 mm (good), **yaw median error 35°, p95 88°**, detection 80 %. Stage sweep: 10 seen, 7 gripped, 2 of 97 in the right bowl, 5 false rejects of 7 presented. A grasp with the jaws 35° off cannot centre the brick, and an off-centre brick in front of the inspection camera looks defective. Everything downstream follows from the yaw.

A median error near 35–45° is what a random or mirrored angle gives. Check, by measurement on the 200 scenes, in this order:

1. **Symmetry.** A 2×2 brick is the same every 90°, a 2×4 every 180°. Score the error modulo that. If the "error" drops to a few degrees, the estimate was right and only the comparison (or the planner's use of it) is wrong.
2. **Mirrored sign.** Image rows grow downward; world +y is up in the top view. An angle taken with `atan2` on pixel coordinates has the opposite sign. Test `-yaw` and `90° - yaw` against ground truth and print the three medians.
3. **Method.** Use the minimum-area rectangle of the brick's mask at 960 px (after the parallax correction of the corners, not only the centre), not image moments, which are unstable for a square.
4. Which side the jaws take: the planner needs the short axis for a 2×4. Check that the camera sorter passes the short axis, and all four sides for a 2×2.

Then re-run the 20-seed stage sweep on Modal and publish the stage counts as they are. Target: yaw median under 4°, and the right-bowl count out of 97 on the page by **03:40 UTC**. The poison test and the forbidden-inputs rule stay. Detection of touching bricks (80 %) is second; do not start it before the yaw is fixed and measured.

The combined-fit training that is running: let it finish in the background; it does not block this.
