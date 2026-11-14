# Supervisor to every Dry Run session, Sat 14.11.2026 19:30 UTC: Touko's order

Touko, 19:25 UTC: "Get it all done, whatever it takes. Use ultra. Don't give up. Take heavy inspiration from how the other guys (Soren) did it, but make it better and cooler. I don't understand why this would be so hard or take so long."

He is right. After 90 minutes the repo is about 1,400 lines of our own code, the benchmark scores 0 and six milestones are open. One session doing the hardest part in series is the reason. This changes now.

## 1. Effort is `ultra`

Every session now runs at reasoning effort `ultra` (maximum reasoning with automatic task delegation). Use the delegation: hand independent pieces to subagents at once, keep working while they run, check what they return by running it. Never wait on one experiment when three could run. The weekly limit is not a concern; the plans end on 17.11 and what is left is lost anyway.

## 2. Five sessions, one owner per part

| Session | Owns | Milestone |
|---|---|---|
| main (`gpt-6-astra`) | the simulator core, the rollout API, the scripted expert, the benchmark, the rendered video | M1 |
| inspector | defects, inspection renders, Anomalib, OpenVINO inspector (`prompts/008`) | M2 |
| page | `web/`, replay export, telemetry, Tiger Cloud, Fly (`prompts/007`, `009`) | M3, M6 |
| policy (new) | demo recorder, LeRobot dataset, ACT training on Modal, DAgger, the policy on the benchmark (`prompts/011`) | M4 |
| analyst (new) | the Nemotron loop, stations from a sentence, the docs (`prompts/012`) | M5 |

Main no longer does M4, M5 or stations. The sharing rules in `prompts/006` hold: commit only your own paths, never `git add -A`, `git stash`, `git reset --hard`, `git checkout -- .` or `git clean`. Requests to another session go in `docs/interfaces.md` under that session's name; read that file every time you finish a piece.

## 3. Main: publish the rollout API first, then sort loose bricks, then piles

Everything else waits on two things from main, so they come before any more grasp tuning:

1. **By 19:50 UTC, the rollout API in `docs/interfaces.md` and in code**, even if the expert behind it is still weak:
   - `dryrun.env.StationEnv(seed, tier=None)` with `reset()`, `step(action)` (six joint position targets, 30 Hz), `observe()` returning `{"top": uint8[H,W,3], "inspection": uint8[H,W,3], "joints": float[6]}` and nothing from the simulator's ground truth, and `result()` (bricks in the right bowl, in the wrong bowl, dropped, cycle times).
   - `env.expert_action()` for the scripted expert, which may use ground truth.
   - `dryrun.rollout.run(seed, controller, inspector=None, record=None)`, where `controller` is the expert or a learned policy and `record` receives every step (for telemetry, replay export and the demo recorder). One code path for expert and policy.
   - The event list both sides log: `reach`, `grasp`, `lift`, `present` (brick held still in front of the inspection camera), `verdict`, `drop`, `miss`, `slip`, `ik_fail`, `unstable`, with brick id and time.
2. **Three difficulty tiers in the held-out benchmark**, fixed by seed and reported separately and together: `loose` (bricks apart, flat; this is the Intel base task), `messy` (touching, any yaw, some on their side), `pile` (dropped heap; the Intel stretch goal). 200 held-out seeds: 80 loose, 70 messy, 50 pile.
3. **By 20:15 UTC the expert sorts the `loose` tier for real**: pick, hold at the inspection camera, drop in the bowl the label says. You have 10/10 single-brick lifts; a loose scene is that, repeated. Then render the video, run `codex review`, commit, write M1 with the per-tier numbers. Then `messy`, then `pile`. A good loose score with an honest low pile score is a result. Zero everywhere is not.

## 4. The box is overloaded, use Modal for anything big

Other projects hold this box at load 100 to 200 on 80 cores. So:
- At most 12 local worker processes per session, started with `nice -n 10`.
- Any job that takes over 5 minutes locally (benchmark sweeps, demo generation, dataset renders, training, policy evaluation) goes to Modal with `.map()` over seeds; command in `prompts/005`. CPU containers with `MUJOCO_GL=osmesa` or GPU containers with EGL, whichever is faster per dollar; measure once.
- Modal money is shared with other entries: Dry Run may spend **25 dollars in total**. Every session adds a line to `status/MODAL.md` for each run (what, containers, minutes, estimated cost). Policy session gets 15 of it, the rest 10. Set timeouts on every function.
- `/bulk2` is nearly full (12 GB). New large output goes to `/bulk3/runs/dryrun/<task>/`. Delete your own scratch files in `/bulk2/runs/dryrun/` when done (literal paths only).

## 5. Checkpoints (UTC). If you are behind one, cut scope, not honesty

- 20:15 loose tier sorted by the expert; rollout API in use by the other sessions.
- 21:00 inspector in the loop; first 500 demos; real replay on the public page from Tiger Cloud.
- 23:00 first learned policy scored on the held-out seeds; Nemotron diagnoses scored against the simulator's event log.
- 01:00 second policy after DAgger and Nemotron's batch against a random batch; page complete.
- 03:00 feature freeze. Judges, fixes, polish.
- 05:00 docs, film plan, final deploy. 06:00 done.

A note on Soren and "cooler" follows as `prompts/013` once the supervisor has read their repo.
