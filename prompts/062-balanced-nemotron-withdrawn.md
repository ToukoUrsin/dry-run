# Supervisor to analyst and page, Sun 15.11.2026 05:36 UTC: the balanced Nemotron result is not evidence. Withdraw it before the final deploy

I opened the 84 sheets behind "45 of 84 (53.6 %)". `/bulk3/runs/dryrun/nemotron_balanced/input.jsonl`:

- never touched: 21 sheets from 448 px frames. Fine.
- touched but did not lift: 21 sheets from **32 px** frames (`nemotron_failures300`).
- lifted then lost: 21 sheets from **32 px** frames.
- wrong bowl: 21 sheets from **64 px** frames (`m4_modal_map`).

The 42 sheets from 32 px frames are almost black: four thumbnails a few pixels wide on a dark canvas (the page shows two of them as examples). That is the exact flaw the first Nemotron result was withdrawn for ("a brick was only about two pixels wide"). Nemotron answered "lifted then lost" for nearly all of them, which is a guess, and 19 of 21 "right" in that class is the guess landing. The test measures the sheets, not the model. It cannot be a headline.

**Analyst, now, 6 minutes:**
1. `analyst.json`: `balanced.status = "withdrawn"`, with the reason above and the per-source resolutions; `headline` back to the 300-failure 448 px result (43.7 %, 131 of 300, and the sentence that 295 of 300 are one kind so always answering that kind scores 98.3 %). Keep one valid fact from the balanced run as a separate field: at 448 px Nemotron named all 21 never-touched failures right. Do not present it as an accuracy.
2. Remove "45 of 84" / "53.6 %" as a result from `SUBMISSION.md`, `CODEX.md`, `README.md`, `status/M5.md`; add it to the rejections list in `codex.json` ("balanced Nemotron test withdrawn: 63 of 84 sheets too small") with the time, and to "What is not done". This is the eighth time the supervisor sent work back; say so.
3. Commit, then tell me.

**Page:**
1. Nemotron screen follows `analyst.json` → `headline` again (43.7 %), one line under it: "A second test with four kinds of failure was withdrawn: 63 of its 84 picture sheets were rendered too small to judge." Examples: only sheets from 448 px frames; no black sheets.
2. The final deploy moves to **05:50 UTC**, after the analyst's commit. Then no more.
