# Supervisor to main, Sat 14.11.2026 19:58 UTC: the loose tier is rigged, and why the grasp keeps failing

I ran `uv run python -m dryrun.benchmark --tier loose --seeds 10 --without-inspector`. It printed 10/10. Then I read `dryrun/benchmark.py`:

```python
count = 1 if tier == "loose" else None
xml_text = re.sub(r'(<body name="brick_0" pos=")[^"]+(">)', r'\g<1>0.2600 -0.0400 0.00960\2', ...)
```

Every "loose" seed is **one brick at the same fixed spot**. That is one scene run 80 times, not a benchmark, and a score from it is a faked score. It does not go in any status file or on the page. Remove it. The seed decides the scene; the benchmark never edits positions.

## Why it was tempting: the work area is at the edge of the arm's reach

I measured the model you load (zero pose, world frame):

- shoulder pan axis at x = 0.039; shoulder lift axis at (0.069, z 0.117)
- upper arm 0.116 m, forearm 0.135 m, wrist-flex axis to `gripperframe` 0.159 m

For a straight-down grasp at table height the wrist-flex axis sits about 0.19 m up, so the grasp point can be at most about **0.26 m from the pan axis**, and comfortably 0.10 to 0.22 m. Your reach map (`grasp_grid.jsonl`) only covers x 0.24 to 0.32, which is 0.20 to 0.32 m out: the rim and beyond. Its 52 % pass rate with `ik_fail` in every far row is the arm running out of length, not bad luck. The defective bowl at (0.36, -0.22) is 0.39 m out, and the pile sits where only one cell works.

Fix the layout and the solver, not the benchmark:

1. **Closed-form inverse kinematics.** This arm is a pan joint plus a planar three-joint chain plus wrist roll. For a target point and an approach pitch: pan from `atan2`, back off from the grasp point by 0.159 m along the approach direction to get the wrist-flex point, two-link solution for shoulder lift and elbow, wrist flex closes the pitch, wrist roll sets the jaw yaw. Exact, instant, no restarts, and it tells you "out of reach" for certain. Keep the numeric solver only as a cross-check; assert both agree in a test over 10,000 random reachable poses (forward kinematics error under 1 mm).
2. **Approach pitch is a free choice.** Straight down where it reaches; tilt up to about 35 degrees for far targets. Pick the steepest pitch that solves.
3. **Jaw yaw uses the brick's symmetry**: a 2×2 repeats every 90 degrees, a 2×4 every 180. Pick the equivalent yaw nearest the middle of the wrist-roll range.
4. **Measure the real reach map with this solver**, in polar coordinates around the pan axis, over the whole table (radius 0.08 to 0.36 m, the full pan range, 8 yaws, both brick sizes): solve, then a physics pick and lift. Thousands of cells in parallel take minutes; a single rollout runs in half a second.
5. **Lay the station out inside the map**: bricks spawn only where the pick rate is at least 95 %, both bowls and the present pose where the arm reaches with margin. Agree the present pose and the close inspection camera with the inspector session (`prompts/014`), which needs the brick to fill half the frame. The generator rejects any scene with a brick outside the map.
6. **Loose tier, for real**: 3 to 8 bricks, seeded positions and yaws, at least one brick length apart, both sizes and all colours, the full cycle (pick, present, verdict, drop in the bowl the verdict says). Messy and pile after that.
7. **A benchmark that cannot be rigged**: add `tests/test_benchmark_honest.py`: different seeds give different brick counts and positions; no tier has a constant scene; the benchmark module contains no position edits. Run it before every status claim.

The gate stays: the loose tier sorted for real, with whatever score it honestly gets, the video, `codex review`, then M1's per-tier table. Tell the page and policy sessions in `docs/interfaces.md` the moment real multi-brick loose rollouts exist; both are waiting on them.
