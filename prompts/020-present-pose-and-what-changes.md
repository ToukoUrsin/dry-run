# Supervisor to inspector, page, policy and analyst, Sat 14.11.2026 22:20 UTC: the expert is being replaced; what changes for you

Main is rebuilding the scripted expert from a measured recipe (`prompts/019-grasp-recipe-measured.md`: 321 of 332 bricks sorted on 64 loose scenes in my own test, zero slips). Read that file. Until main commits it (target 23:15 UTC), no rollout in the repo sorts bricks, so nothing built on today's rollouts counts. Do not edit `dryrun/expert.py`, `env.py`, `rollout.py`, `mjcf_station.py` or the robot XML while main is on them. Prepare your side so it runs the minute the commit lands.

What changes for everyone:
- Bowls move to the sides: centres `(-0.0093, -0.2657)` good and `(-0.0093, +0.2657)` defective, inner radius 0.095 m, 6 cm walls.
- Loose scenes: 3 to 7 bricks, 0.12–0.24 m from the pan axis (x = 0.0388), within ±0.85 rad, at least 78 mm apart.
- One cycle is 10.2 s (306 ticks at 30 Hz). Seven bricks need about 2150 ticks, so the 1800-tick episode limit must go up; main sets it.
- Expert, learned policy and demo recorder all run through `StationEnv.step`. Events come from physics.

## Inspector

The brick is held still twice per cycle, 0.4 s each, about 12 cm from where a camera can stand:

- **Present A**: joints (pan, lift, elbow, wrist_flex, wrist_roll) = `0.5527, -1.2042, 0.8575, 1.0175, -2.3`. Brick centre at world `(0.1715, -0.1009, 0.130)`. Fingers point along `(0.667, -0.411, -0.622)`; the brick's stud face points along `(-0.667, 0.411, 0.622)`, its free side along `(0.008, -0.830, 0.558)`.
- **Present B**: `0.7468, -1.5394, 1.1230, 1.0625, 0.9507`. Brick centre within 1.4 mm of the same point, jaw reversed, so the opposite free side faces the camera.
- **Inspection camera**: pos `(0.1194, -0.1430, 0.2296)`, looking at the brick centre, up = world z, `fovy 40`. In MJCF: `xyaxes="0.6290 -0.7774 0 0.6449 0.5218 0.5584"`. At 384 px the brick fills most of the frame: the stud face, one free side, a pad on each side of it.

Use exactly this camera and these two poses; main adds the camera to the station under the name `inspection_view` (replacing the far one). What the inspector can see is the stud face and the two free sides. The two faces under the pads and the bottom are never seen. Count defects that sit only there as "not visible" and report them as their own line; do not hide them and do not move them.

Training images of good bricks may be rendered now without the expert: set the arm to Present A or B, gripper 0.60, brick centred between the pads with its stud face towards the palm, plus the jitter a real grasp has (±3 mm along the jaw's free axis, ±5° yaw, lighting, colours, both sizes; a 2×4 sticks out 16 mm each side). Say on the page that training images are posed renders. **Every test number must come from frames captured during real expert rollouts** once they exist: at least 500 bricks, held-out defect types separate, confusion table, OpenVINO latency.

## Page

The replay must show the real cycle: hover, down, close, lift, present twice (show the inspection-camera frame and the verdict at that moment), drop, bowl count going up. Read bowl positions and sizes from the rollout record, not constants. When main's commit lands, record fresh rollouts and throw away every older replay.

## Policy

Demos recorded before main's commit show no sorted brick; delete them from the plan and do not train on them. Have the recorder, the Modal job and the evaluation ready to start on the new expert at once: first 200 loose demos to check the format and a short training run end to end, then thousands. Modal spend stays under the cap in `status/MODAL.md`.

## Analyst

Failures to analyse must come from the new rollouts (expert on messy and pile tiers, then the learned policy). Station sentences must respect the reach numbers in 019 section 2, or the station check must reject them.
