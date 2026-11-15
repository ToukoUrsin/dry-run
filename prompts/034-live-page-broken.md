# Supervisor to page, Sun 15.11.2026 01:41 UTC: the live first screen is broken right now. Fix before anything else

Deploy `9b9c934`, opened at 01:39 and again at 01:41 UTC (`/bulk3/runs/dryrun/review_live_0140_full.png`):

- Chrome reports one page error: `Cannot read properties of undefined (reading 'toLowerCase')`. After it the viewer stays dark with "loading trace", the top bar says "loading evidence", the three tier cards are gone, and the first screen is a headline over a black box.
- Likely cause: `/api/benchmark-tiers` changed shape at 01:25 when main moved it to Tiger (rows are now `{"tier": "loose", "expert_version": "b9d897c", ...}`, no `label`). Read every field defensively; one missing field must never stop the viewer.
- Both camera panels say "No camera video is attached to this deployment." They had video before `9b9c934` ("exclude generated datasets from deploy context"). The deploy context dropped them. Put them back.
- The eleven evidence pictures in the Inspector section are grey boxes in a full-page capture. Check that they are deployed and load (or are lazy and do load on scroll).

Rules from now to the end:

1. **Never deploy without opening the deployed page in a real browser and reading the console.** A page error on load means roll back at once (`fly releases`, redeploy the previous image), then fix.
2. The viewer and the first screen load from files baked into the deploy and must work with every API down. APIs only add to it.
3. The arm file is 11 MB. Show the scene at once with a light stand-in (the bowls, table and bricks are tiny) and swap the arm in when it arrives, or split the arm so the first paint is under 2 s on a phone connection. Gzip or meshopt it.

Then continue with prompt 033. Section notes from the same capture:
- "Cameras only" and "Learned policy" tables show raw ids (`ground_truth_poses_plus_inspector`, `act-shards-three-v2-eval600-cpu-20261115-1 · reached`). Write them as words; draw the policy stages as one small funnel per checkpoint, not 24 rows of zeros.
- Numbers: `471.75459324999997 ms` → `472 ms`. Round everywhere.
- "Messier piles": the bars carry tiny grey text at the far right. Put before → after as two bars per tier with the per cent in normal type.
- The headline still says 382 of 382 with commit `64c4919`. The story now is the three tiers after the pushing expert; take the headline numbers from `tiers.json`.

Target: first screen working again within 10 minutes.
