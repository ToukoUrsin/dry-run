# Supervisor to main, Sun 15.11.2026 01:06 UTC: Tiger Cloud holds one old rollout. It must hold every benchmark rollout

The live page's `/api/telemetry` reads Tiger Cloud and returns 1,128 raw rows from one rollout: seed 7 at 19:25 UTC, the broken expert. Goal item 9 is "Tiger Data holds every rollout", and Tiger Data is one of the four prizes. `dryrun/benchmark.py` already writes 30 Hz telemetry when `DRYRUN_DB_URL` is set, so this is mostly a matter of setting it.

1. Run the held-out re-runs of prompt 027 (80 loose, 70 touching, 50 pile, new expert) with the database set to Tiger Cloud: `hsec exec --only DRYRUN_TIGER_DB_URL -- bash -c 'DRYRUN_DB_URL="$DRYRUN_TIGER_DB_URL" ...'`. If a re-run is already going without it, let it finish for the numbers and run it again with the database; the benchmark is deterministic by seed, and the two must agree (say so if they do not).
2. Every row carries: rollout id, seed, tier, expert version (commit), tick time, six joint positions, six targets, gripper effort, and the events (reach, grasp, lift, present, verdict, drop, push, miss, slip) in their own table. Outcome per rollout: bricks, correct, false accepts, false rejects, pushes, cycle time.
3. Also load the "before" run (old expert, commit `64c4919`, same seeds). It is 200 episodes; run it again with the database set. Then the before → after comparison on the page is a query, not a constant.
4. Write in batches (COPY or `executemany` of a whole episode), not row by row; 20 workers will otherwise crawl. Never print the URL.
5. Continuous aggregates the page needs: score by tier and expert version; median and p90 cycle time per brick by tier; pushes per episode; failures by event and tier; rows and rollouts in total. Compression on the raw hypertable, and report the measured ratio and the query times.
6. `/api/benchmark-tiers` must be answered from those aggregates on the live app (with the file as the fallback only when the database is down, and the response says which source it used). Tell the page session the field names in `docs/interfaces.md`.
7. Proof in `status/M3.md`: `remote-stats` output (raw rows, rollouts, aggregate rows, compression ratio), and the live URL's JSON for the tiers with `source: timescale`.

Do this alongside prompt 027 step 2, not after everything else. Target: rows in Tiger by 01:50 UTC.
