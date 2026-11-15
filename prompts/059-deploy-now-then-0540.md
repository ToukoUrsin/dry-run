# Supervisor to page, Sun 15.11.2026 04:51 UTC: the film is cut, the freeze is over. Deploy the local build now; final deploy at 05:40

I checked local 8891 at 1440×900: Codex screen (362 commits, score lines to the right edge, lanes with model ids, four tiles, authorship sentence, "Read 58 prompts · Read the Codex log · Read the submission"), `/codex`, `/prompts`, `/submission` and a prompt page all answer 200. Good work.

1. **Deploy now** through the normal pipeline. After it: `/api/health` shows the deployed commit; `/prompts`, `/codex`, `/submission`, `/prompts/037-redesign-around-the-3d.md` answer 200 on live; a prompt diamond opens its text; no console errors at 1440×900 and 390×844; the first screen and the three tier videos still play.
2. Between now and 05:40, locally:
   - The **codex review tile**: `codex.json` → `review` is being updated by the analyst to five real review runs with findings and fix commits (`status/codex-review-0415.md`). Show it as the strongest tile: runs, findings, fixed, and the one line "review found why the learned policy scored zero". Link it to `/status/codex-review-0415.md` served like the other documents.
   - **Nemotron**: the sentence prints "43.6667%" and "98.3333%": one decimal everywhere, and when `analyst.json` → `headline` / `balanced` exists use that as the number and sentence, with the 300-failure result below it.
   - **Inspector heat maps**: the first passed brick is drawn in a pale palette and the rest in the rainbow one. One palette for all eight.
   - **Policy** and **Cameras only**: the policy and inspector sessions are fixing bugs the review found; their JSON will change before 05:30. Draw whatever the JSON says at deploy time; no hard-coded numbers ("21,588 steps" is hard-coded text today).
   - The parallel-sessions tile has a blank line between every row: tighten it.
   - `/codex` shows raw Markdown in a monospace block: render headings, tables and links as HTML so the session table reads as a table.
3. **Final deploy at 05:40 UTC** with whatever is committed then, same checks, then stop deploying. Tell me the commit.
