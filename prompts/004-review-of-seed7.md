# Supervisor to Codex, Sat 14.11.2026 18:32 UTC: review of `seed7_real2.mp4`

The status files are honest now. Keep them that way. What I saw in the video, in order of importance:

1. **Zero pickups is the one thing blocking everything.** Put your best effort here before anything else, and split it: one session on the grasp, others carry on with M2 to M5 wiring. Things that usually decide a friction grasp in MuJoCo, check each one by measurement, not by guess:
   - Contact model: `cone="elliptic"`, `impratio` around 10, `noslip_iterations` 2 to 5, timestep 2 ms or less.
   - Jaw contact: the SO-101 jaw meshes collide as convex hulls. Add thin box pads on the inside of both jaws as the only gripper collision surfaces, friction about 1.5 to 2, `condim="4"`. Same `condim` and high friction on the brick sides.
   - Actuators: position control on the gripper needs enough gain and force range to squeeze, and the arm needs enough gain to hold the brick's weight without sagging. Print the contact force on the pads while closing.
   - Approach: come straight down with the jaws across the brick's short side (31.8 mm), stop a few millimetres above the table, close slowly, wait until both pads report contact force, then lift slowly.
   - Inverse kinematics: solve for the pad midpoint, not the wrist. Check the solved pose by forward kinematics before moving. The arm has 5 joints plus the gripper, so position plus a top-down approach direction is the most you can ask for.
   - Debug on one brick alone on an empty table until it lifts 10 times out of 10 from different positions and yaws. Only then the pile.
2. **The two bowls intersect each other** in seed 7 (red and green overlap). The "nothing intersects at rest" check missed it. Fix the check, then the layout.
3. **The arm never gets near the bricks** in this video. Whatever the benchmark says, look at the frames: if the gripper is not at the brick, the IK target frame or units are wrong.
4. **Scale.** Check real sizes: a 2×2 Duplo is 31.8 × 31.8 × 19.2 mm plus studs; the SO-101 reaches about 40 cm. The bowls look larger than the arm's shoulder. Use a bowl about 14 to 16 cm across.
5. **Looks, later.** The render is flat and dark. This video is only a proof. The page's own 3D replay is what people will see, so do not spend time on MuJoCo lighting now; do make the three-view layout readable.

Do not answer this with a summary. Fix 1 first.
