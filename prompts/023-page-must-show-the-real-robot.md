# Supervisor to page, Sat 14.11.2026 23:58 UTC: the first screen is rejected again. Show the real robot sorting.

I opened the live page at 23:50 UTC (screenshot: `/bulk3/runs/dryrun/review_live_2350.png`). What a stranger sees: a robot made of five yellow boxes, two domes for bowls, three grey posts in front of everything, nothing moving, and the headline "validation · Pre-handoff validation trace · not a benchmark". Nobody understands that in 15 seconds, and it looks like a mock-up. Meanwhile the simulator now really sorts bricks (commit `64c4919` and later). The page must show exactly that.

## 1. The 3D view shows what MuJoCo simulates, not a stand-in

- **Geometry comes from the compiled MuJoCo model.** Write one export step (Python, run at build time) that walks every geom of a station model and writes a viewer scene: mesh geoms with their real vertices and faces (the SO-101 parts; simplify to at most 200 000 triangles in total so the file stays under about 4 MB), boxes, cylinders (studs, bowl floors), each with its body, local pose and colour. One binary file (glTF `.glb` with one node per body is fine). The bricks have studs, the bowls are rings with a floor, the gripper has its two black pads. No hand-built shapes in JavaScript. Leave the camera posts out of the view, or draw them as thin rods.
- **Motion comes from recorded body poses.** Each replay frame stores world position and quaternion of every moving body (arm links, jaw, bricks), 30 Hz or 15 Hz with interpolation, quantised. The viewer sets the node transforms. No kinematics in JavaScript, so the replay cannot disagree with the physics.
- **Framing.** Three-quarter view from the front right, close: arm, bricks and both bowls fill the frame. Drag to orbit, scroll to zoom.
- **Look.** Background the same colour as the page, matte light table, soft shadows, a key light and sky light, filmic tone mapping, sRGB output, antialiasing, device pixel ratio up to 2. The arm is yellow printed plastic with black motors (colours from the model).

## 2. The first 15 seconds

- The replay **plays by itself on load**, loops, at 1.5×, starting with a loose scene of 5 or 6 bricks. No "Waiting for a recorded brick".
- At each `present` event the **real inspection-camera frame from that tick** slides in as a card with the verdict (GOOD or DEFECTIVE) and the score; then the brick goes to that bowl and the bowl's counter goes up. At the end: "6 of 6 sorted", then the next seed.
- Small labels in the view: tier, seed, "brick 3 of 6", simulated time.
- The two side videos either play in sync with the replay or go; a paused video with a black top half is worse than none.

## 3. The headline is the benchmark, in plain words

Replace the "validation" block with the measured result by tier, read from the benchmark files of the current commit:

- Loose bricks: 382 of 382 sorted (80 held-out scenes)
- Touching bricks: 217 of 494
- Dropped pile: 31 of 386

One line under it: "Scripted expert with ground-truth labels, simulated in MuJoCo, commit …. Nothing on this page is a real arm." When the inspector runs in the loop and when the learned policy has a score, they get their own rows. The low numbers stay: they are the point of the tool.

## 4. Proof

Deploy, then open the live URL yourself at 1440 px and at phone width, wait 20 s, take screenshots and look at them. Fix what you see. Write the two screenshot paths in `status/M6.md`. Target: live by 00:45 UTC. I will open it then.
