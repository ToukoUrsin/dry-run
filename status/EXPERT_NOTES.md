2026-11-14 post-commit validation at 2e99a05: seed 1 loose (4 bricks) reached grasp/lift then physics reported unstable at t=3.634 s before any present/drop; seed 0 page capture reproduced MeasuredExpert.action current=None during phase transition.
2026-11-14 controlled MuJoCo check: MeasuredExpert._solve((0.20,-0.10,0.16), pitch=0.9, jaw=-2.3) returned None in 0.465 s although prompt-020 Present A joints forward to TCP (0.171495,-0.100870,0.130010), so the numeric presentation solve cannot be used as evidence.
2026-11-14 replay-export check at 64c4919: the stale manifests counted `good_bowl` instead of `result.correct` (so defective bricks in the defective bowl were mis-scored); exporter fix is in the page worktree, and those manifests are diagnostic only.

2026-11-14 Prompt 022 pile release audit at commit 64c4919: seed 1003 through
the public `StationEnv` produced 4/7 correct and 5 physical `grasp` events.
The trace is `/bulk3/runs/dryrun/audit/pile1003-trace.jsonl`. Brick body 17
(scene index 5) was stable at the drop pose until the gripper opened: at
49.400 s it had eight pad contacts and q=0.594; at 49.434 s contacts fell to
one and the brick acquired lateral velocity; by 49.634 s it was moving at
2.00 m/s and the environment reported `unstable` at 49.934 s after it left
the bowl area. This is an honest pile failure from release contact dynamics,
not a successful drop. The benchmark's `picked_by_contact` field now counts
unique brick ids from physical `grasp` events (with a legacy flag fallback).

2026-11-15 Prompt 027 pushing-expert held-out loose rerun at `b9d897c`: 80 seeds,
382 bricks, 380 correct (99.48%), zero pushes, zero slips, and one unstable
release. Both lost bricks are seed 1079 (7 bricks, 5 correct); the trace has
an `unstable` event for brick body 14 at 49.600 s after release. The result is
above the 99% gate, so the pushing expert remains the loose controller. This is
the known roughly 1-in-900 release-fling case, not a hidden success.
2026-11-15 camera-only audit at 2b1024a: CameraExpert emits opaque tokens without a physical focus body, so env._active stays None and inspection crops retain other bricks; seed-7 token-2 scores 0.789/0.780 versus expert body-13 0.409/0.417, with paired frames under /bulk3/runs/dryrun/m2_rebuild/camera_trace_seed7_present and real_inspector_seed7_expert.
