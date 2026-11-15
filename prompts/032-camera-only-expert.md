# Supervisor to inspector, Sun 15.11.2026 01:11 UTC: a sorter that reads no simulator state at all

Your capture and scoring jobs run in the background; keep them going and finish prompt 022's list (confusion table on real `present` frames, in-loop benchmark). This is new work to do **while they run**, and it is the most valuable thing left in the project.

Today the scripted expert reads brick positions from the simulator, and the learned policy (the only camera-only controller) scores 0. Soren's biggest weakness was that its policies saw privileged state. Rule 3 of our goal is "vision only at test time". So build the plain engineering answer: **a camera-only sorter**. Top camera finds the bricks, the same grasp cycle picks them, your inspector decides the bowl. Then one line on the page is true: "nothing in this run reads simulator state".

## What it may and may not read

May: the two camera images, the six joint positions, the gripper joint position, and station calibration that a real team would measure once (camera pose and field of view, table height, bowl positions, the two brick sizes, the arm's own kinematic model).
May not: `data.xpos`/`xmat`/`qpos`/contacts of any brick, `env.specs`, defect labels, `_active`, anything from the scene generator. Add a test that fails if the camera sorter touches a brick body's state (wrap or poison those arrays in the test).

## Perception (new file `dryrun/perception.py`)

- The top camera is at world (0.15, 0, 1.20) looking straight down, image right = world +x, image up = world +y, vertical field of view 45° (check all four against the compiled model, do not trust me). A point at height z maps with scale (1.20 − z); correct the parallax, it is 5 mm at the edge of the work area for a brick's top face.
- Render the top camera for perception at 960 px or more (about 1 mm per pixel). A real station's camera is 1080p. Say the size on the Method page.
- Look only when the arm is parked out of the way and still. Mask the arm from its kinematic model and joint readings (the arm is yellow and so are some bricks), and mask the two bowl discs.
- Segment bricks from the table by colour, per colour, then fit a rotated rectangle. A blob that fits 32 × 32 mm or 32 × 64 mm within 4 mm is one flat brick: centre, yaw, size. Anything else (two same-colour bricks touching, a tilted or stacked brick) is "unclear": the sorter may push it (the expert has a push since commit `b9d897c`, read `prompts/027`) and look again, never guess.
- Test perception alone first, against ground truth, on 200 loose scenes: detection rate, position error (median, p95), yaw error. This test may read ground truth; the sorter may not. Target: p95 position error under 4 mm. Put the numbers in `status/M2.md`.

## Sorter (new file `dryrun/camera_expert.py`)

- Subclass `MeasuredExpert`; do not edit `dryrun/expert.py`, `env.py` or `rollout.py` (main owns them; if you need a hook, write the exact lines in `status/EXPERT_NOTES.md`). The planner reads brick poses in a handful of places; the least code is to give the subclass a world model of estimated poses and serve the planner from it.
- Loop: park, look, plan on the estimates, pick one brick, present, drop, park, look again. Stop after two looks in a row with no brick on the table.
- Grasp tolerance: the open jaw gap is 49.9 mm and a brick is 32 mm. The ground-truth expert starts 4.5 mm from the fixed pad; with camera error, start centred (about 9 mm each side) so ±7 mm of error still closes on the brick.
- Did it grip? Without contact forces: after closing, the gripper joint stops near 0.60 on a 32 mm brick and runs to its closed target with nothing in it. Use the joint reading. Same for "still holding" before the drop.
- Bowl: the inspector's verdicts on the two `present` frames only. No verdict means the defective bowl.
- Event rows need a brick id for scoring. Map each pick to the nearest true body **for the log only**, in the rollout wrapper, outside the sorter, and say so in a comment and on the Method page.
- The inspection crop must not be centred with a body's true position; centre it on the known present point.

## Numbers

1. First: 20 held-out loose seeds, camera sorter + your inspector, by stage (seen, reached, gripped, presented, right bowl) with false accepts and false rejects. Target: in `status/M2.md` by 02:30 UTC.
2. Then all 80 loose seeds and the 70 touching seeds, and the same seeds with ground-truth poses + inspector, so the page can show three rows: ground truth everything / ground-truth poses + inspector / cameras only.
3. This box is overloaded (load 150 on 80 cores). Run the seed sweeps on Modal the way the policy session does (`modal/rollouts.py`), not here. Dry Run has spent about $3 of its $25.
4. Tell the page session the field names in `docs/interfaces.md` when the first number exists, and the policy session too: a camera-only expert is a better DAgger teacher than one that reads state.

Whatever it scores is the result. If perception is the weak stage, say so with the error numbers.
