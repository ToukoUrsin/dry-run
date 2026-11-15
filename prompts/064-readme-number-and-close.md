# 064 · README number, then close (06:11 UTC, supervisor)

To: analyst.

1. `README.md` line 17 still says the reference expert is correct on 203/205 bricks (99.0%) across 40 loose scenes. Every other place (page, CODEX.md, SUBMISSION.md, the Junction description) says 380 of 382 loose (99.5%), 451 of 494 touching (91.3%), 259 of 386 pile (67.1%). Make the README say the same three numbers, taken from `data/page/tiers.json`, not typed by hand. Read the rest of the README once for any other number that disagrees with `data/page/*.json` and fix those the same way.
2. Run `night_data.py` one last time and check CODEX.md, SUBMISSION.md and README agree with the page data. The policy line everywhere is: after the fix it reaches 5 of 20 one-brick scenes and still sorts 0; earlier result 0 of 1,309 bricks on 200 held-out scenes.
3. Commit. Do not deploy. Finish by 06:30 UTC and write the final commit hash in your status file.
