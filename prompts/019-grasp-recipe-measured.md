# Supervisor to main, Sat 14.11.2026 22:10 UTC: stop tuning. Here is a measured recipe that sorts 96.7 %

Four hours on the grasp and the honest loose score is 0 of 50. Stop the current line of work now. I ran my own throwaway experiment outside the repo on your station model (same MuJoCo 4.0.1, same SO-101 file, same bricks) and changed one thing at a time until it worked. Result: **64 random scenes, 332 bricks, 321 sorted (96.7 %), 0 slips, 0 missed grasps, 0 missed drops.** All 11 failures were bricks my random layout put right beside a bowl, where no clear grasp exists. Every grasp that was attempted held through lift, two presentation poses and the drop. Pad force was 14 to 16 N on each pad every time.

Build exactly this. Do not re-derive it, do not "calibrate" it, do not keep the old expert next to it. You write the code; the numbers below are measured.

## Why the old attempts failed (each one measured)

1. **The grip was a 6 mm deep collision, not a grip.** Gripper position control has kp 998 and a 12 N·m limit, and contacts are soft by default. The jaw reached its closed target by sinking 6 mm into the brick. A 6 mm thick pad that is 6 mm inside a box flips its contact normals. That is what threw bricks to (2.29, −1.65, −2.68).
2. **The moving pad was a flat plate on a swinging jaw.** At a 32 mm opening it was tilted 36°, so it touched the brick with one edge and pushed it down into the table.
3. **Friction locked the brick to the table.** MuJoCo uses the larger friction of the two geoms. Brick friction 1.8 made brick-on-table 1.8 too. The closing pad drags the brick down, the table then holds it, and it never slides the last half millimetre to the fixed pad (I measured moving pad 11.7 N, fixed pad 0 N).
4. **No room on the fixed side.** With the fixed pad 0.5 mm from the brick, any tracking error lands the pad on top of the brick and kicks it 3 to 4 cm.
5. **The bowls stood in the workspace with 8 cm walls**, and bricks were laid out to 0.27–0.29 m from the pan axis. A top-down hover is only solvable to 0.24 m.

## 1. Model changes

In `so101_dryrun.xml` (gripper body frame; the jaw frame is the `moving_jaw_so101_v1` body):

| What | Value |
|---|---|
| `fixed_grip_pad` | box, half-size `0.005 0.018 0.012`, pos `0.001 0 -0.085` |
| `moving_grip_pad` | box, half-size `0.005 0.012 0.018`, pos `-0.01596 -0.06371 0.0188`, quat `0.955336 0 0 -0.295520` |
| both pads | `friction="1.5 0.01 0.001" condim="4" priority="1" solref="0.005 1" solimp="0.95 0.995 0.001"` |
| gripper actuator | `forcerange="-1.0 1.0"` (about 15 N at the pads) |

The quat pre-rotates the moving pad by −0.6 rad about the jaw hinge. Check it by measurement: at gripper joint 0.60 the two inner faces are parallel and 32.0 mm apart; inner-face gap is 24.6 mm at 0.50, 41.5 mm at 0.80, 49.9 mm at 1.00. `priority="1"` makes the pad's friction and stiffness win in pad-on-brick contact, so the brick itself stays soft and slippery everywhere else. Do **not** stiffen the brick geoms: with `solref 0.004` on the bricks they bounced out of the bowl at 3.9 m/s.

In `mjcf_station.py`:

- Brick body and stud friction `0.5` (first value). Table friction `0.5`.
- Bowls: inner radius 0.095 m, floor cylinder half-height 0.004, 24 wall boxes of half-size `0.005 0.0135 0.030` (6 cm wall) on that radius, friction `0.6 0.05 0.01`. "Bowls on the sides" puts them at `(-0.0093, ±0.2657)`: 0.27 m from the pan axis at ±1.75 rad, out of the brick area. Good bowl at −y.
- Never change geom sizes or body positions on a loaded `MjModel`. The bounding volumes are not rebuilt and bricks fall through. Change the XML.

## 2. Frames and inverse kinematics

- Pan axis is at world x = 0.0388, y = 0. "Radius" and "azimuth" below are measured from it. Pan joint ≈ −azimuth.
- **TCP** = the point `(0.0225, 0, -0.086)` in the gripper body frame. It is the centre of a 32 mm brick that touches the fixed pad.
- Finger direction = −z of the gripper body. Jaw axis = +x of the gripper body, pointing from the fixed pad to the moving pad.
- "Pitch p" means fingers point down, tilted outward by p radians along the radial direction: `f = sin(p)·r̂ − cos(p)·ẑ`.
- IK I used (any solver is fine if it meets the same acceptance, checked by MuJoCo forward kinematics, not by its own model): damped least squares on `mj_jac` at the TCP. Task = position error (3 rows) and 0.05 × rotation error (3 rows), rotation error = `cross(f, f_des) + cross(x_jaw, jaw_des)`, damping 1e-6, step clipped to 0.3 rad, joints kept 0.02 rad inside limits, 120 iterations. Seeds: pan = −azimuth; (lift, elbow, wrist_flex) from `(-0.3, 0.6, 1.2)`, `(0.2, 0.2, 1.2)`, `(-0.8, 1.2, 1.1)`; wrist_roll from `0, ±1.4, ±2.5`; plus the previous solution. Accept: position < 1 mm, finger direction < 2°, jaw axis < 3°. The jaw axis is **signed**: it decides which side the fixed pad is on.
- Measured top-down reach (all five azimuths in ±1.2 rad): TCP z 0.013 → radius 0.08–0.28 m; z 0.08 → 0.12–0.24 m; z 0.14 → none. With pitch 0.6–0.9 every transport pose below solves.

## 3. The cycle (30 Hz ticks, minimum-jerk interpolation in joint space, one tick = 17 physics steps)

Gripper targets: open **1.0**, closed **0.5** (contact happens at 0.60; the 1 N·m limit sets the force).

Fixed poses, solved once per station:
- **Neutral**: TCP (0.22, 0, 0.14), pitch 0.6.
- **Present A**: TCP (0.20, −0.10, 0.16), pitch 0.9, then set wrist_roll = −2.3. My solution: pan 0.5527, lift −1.2042, elbow 0.8575, wrist_flex 1.0175, roll −2.3. The brick centre is then at world (0.1715, −0.1009, 0.130).
- **Present B**: same brick position and finger direction, jaw axis reversed (wrist roll + π), solved by IK so the brick stays at the same point.
- **Drop** for each bowl: TCP at bowl centre + one of `(±0.03, ±0.03)` or `(0, 0)`, z 0.115, pitch 0.6. Cycle through the five offsets so bricks do not stack on one spot.

Per brick:
1. Pick the nearest brick that lies flat and has a plan. Candidate jaw axes: ± the brick's short axis; for a 2×2 all four sides. A candidate is **clear** if the gripper footprint (along the jaw axis from −27 mm to +52 mm around the brick centre without the slot −17.5…+18.5 mm, and ±19 mm across) touches no other brick rectangle grown by 3 mm and stays 15 mm outside every bowl. It needs IK at z 0.013, 0.055 and 0.07.
2. **Approach offset**: shift all three targets 4.5 mm towards the fixed-pad side (minus the signed jaw axis). The closing jaw pushes the brick those 4.5 mm onto the fixed pad.
3. Neutral → hover z 0.055, gripper open: 1.0 s, hold 0.15 s.
4. Down to z 0.035: 0.35 s. Down to z 0.013: 0.35 s. Hold 0.2 s. (IK at each height, straight down.)
5. Close to 0.5 over 0.6 s, hold 0.3 s. If either pad carries less than 0.2 N on this brick: `miss`, open, back up to hover, neutral, next brick.
6. Up to z 0.04: 0.4 s. Up to z 0.07: 0.4 s.
7. Present A: 1.2 s, hold 0.4 s. Present B: 1.0 s, hold 0.4 s. The inspector looks during the two holds. If the brick is more than 20 mm from the TCP: `slip`.
8. Drop pose: 1.2 s, hold 0.2 s. Open over 0.4 s, hold 0.5 s. Neutral: 1.0 s.

One cycle is 10.2 s of simulated time. Sorted = brick centre inside the right bowl's radius, 0 < z < 0.11, at rest at the end of the episode.

## 4. Loose tier layout

3 to 7 bricks, radius 0.12–0.24 m, azimuth within ±0.85 rad, random yaw, centre distance at least 78 mm between two 2×2 and 95 mm if either is a 2×4. Reject and resample a scene if the planner in step 1 (geometry and IK only, no physics) cannot find an order in which every brick has a clear grasp. Say this rule on the page: loose means every brick can be gripped on its own. Messy (touching) and pile (dropped) do not get this check; the expert will fail on some of those and that is the honest result.

## 5. What to build, in this order

1. Apply section 1. One new module, `dryrun/expert.py`, holding the IK, the planner and the cycle as a state machine that returns six joint targets per 30 Hz tick. **Delete** the old expert in `render.py`, `grasp_test.py`, `grasp_grid.py` and the three older IK versions. `StationEnv.expert_action()` calls the new expert. The expert rollout, the policy rollout and the demo recorder must all go through `StationEnv.step`, one path. Right now `rollout.run` sends the expert down a different code path from the policy; remove that.
2. Events come from physics: `grasp` when both pads carry force, `lift`, `present`, `verdict`, `drop`, `miss`, `slip`, `unstable`.
3. Tests: one brick alone on a grid (radius 0.12–0.24 × azimuth ±0.8 × six yaws, both sizes) lifts and holds ≥ 98 %; 20 loose seeds sort ≥ 90 %; `tests/test_benchmark_honest.py` still passes.
4. Run the 80 held-out loose seeds with ground-truth labels, write the number and the failure stages in `status/M1.md`, **commit and push that at once**, and add three lines to `docs/interfaces.md` (poses, event names, what changed). Four sessions are waiting on this commit. Then messy and pile, then the inspector in the loop, the video and `codex review`.
5. Tell me in `status/M1.md` if any number here does not reproduce, with your measurement next to mine. Do not quietly change the recipe.

Target: step 4 committed by 23:15 UTC.
