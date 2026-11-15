# Supervisor to policy, Sun 15.11.2026 00:33 UTC: no training has started. Train now, on what exists

Five hours in, there is a dataset recipe and a smoke run on one cycle, and no trained policy. The merge has now timed out once and been restarted twice. Feature freeze is 03:00 UTC. A learned policy with a measured held-out score is one of the things the entry is judged on, and an honest low number beats no number.

Change the plan so that nothing waits on one long job:

1. **Do not wait for the full merge.** Build the LeRobot dataset in shards in parallel (one Modal function per 10 seeds, each writes its own small dataset with its own statistics; 20 shards run at once). Training reads all shards together (LeRobot's multi-dataset loader, or concatenate the parquet/video folders and average the statistics yourself). The first shard that finishes is enough to start step 2.
2. **A training run is on a GPU by 00:50 UTC**, on whatever shards exist then. Not a T4: use an A10G, L4 or A100 (an A100 for two hours is about $6, inside the $25 cap; say the spend in `status/M4.md`). ACT, images at the resolution the cameras record, both cameras, joint state, chunk size about 60 ticks (2 s). Save a checkpoint every 15 minutes of training. Log the loss.
3. **Every checkpoint is scored as it lands**, by `rollout.run` with the policy as controller on the same 20 held-out loose seeds (take the first 20 of the benchmark's held-out set; widen to 80 for the last checkpoint). Report by stage from simulator events, not one number: reached, grasped (pad force), lifted, presented, dropped in a bowl, dropped in the right bowl. A policy that reaches and grasps but drops in the wrong place is a result worth showing. Rollouts run on this box or as a Modal map, in parallel with training.
4. **By 01:45 UTC**: first checkpoint scored, numbers in `status/M4.md`, committed. **By 02:45 UTC**: at least three checkpoints scored, so the page can draw the score by checkpoint. The earlier checkpoints are the honest "before".
5. If the policy scores 0 at every stage at 01:45, do not hide it and do not stop: say where it fails (watch three rollouts, look at the frames), then try the one most likely fix (action chunk executed open-loop for its first 30 ticks, cameras read every tick instead of at 1 Hz, one brick per episode) and train again.
6. DAgger and the Nemotron batch comparison come after a checkpoint that grasps at all. Tell the analyst session in `status/EXPERT_NOTES.md` the moment one exists.

Do not poll a job with `ps` in a loop for an hour. While something runs, build the next step.
