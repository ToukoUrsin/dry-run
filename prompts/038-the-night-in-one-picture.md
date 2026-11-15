# Supervisor to analyst, Sun 15.11.2026 02:02 UTC: data for "the night in one picture" (page section 2, the Codex screen)

Touko's main goal for Dry Run is Best Use of Codex. The page is being redesigned (prompt 037); its second screen is one picture of the whole night. You own its data. Add a `night` object to `data/page/codex.json`, and keep it fresh with a script (`scripts/night_data.py`, one command) so the page session can re-run it before each deploy.

- `night.start_utc` = `2026-11-14T18:00:00Z`, `night.end_utc` = time of the run.
- `night.sessions`: the five sessions with a short name, model, effort, what each owns (six words at most).
- `night.commits`: every commit from `git log`: `t` (UTC), `sha` (7), `subject`, `session`. Attribute by the files a commit touches (`web/` → page; inspector files and `data/page/inspector.json` → inspector; `modal/`, policy files → policy; analyst, stations, Nemotron, docs → analyst; the rest → main). Say in one field how the attribution was done. Commits made by other tools (the two film-plan commits and their reverts at 01:43–01:51) go in a sixth group `other`, not in a Codex lane.
- `night.prompts`: every file in `prompts/` (`t` from its first line, `file`, `title` in plain words of at most 8 words, `to` which session).
- `night.rejections`: the moments work was sent back, from the prompts and status files, each with `t`, 6-word label, and the prompt file: the rejected first pass (18:05), zero pickups (18:32), "0 of 50 after four hours" (22:10), the rigged one-brick tier, the withdrawn inspector (rejected every good brick on real frames), the first policy run (201 steps, 0 at every stage), the broken live page (01:41). Check each against the repo; drop any you cannot back with a file.
- `night.scores`: measured benchmark points over time for loose, touching and pile (`t`, `tier`, `sorted`, `bricks`, `commit`), only from recorded benchmark runs in `status/` and `data/`. Zero is a real point: include the 0 of 50.
- Totals: commits, prompts, sessions, lines of code written (command in the JSON), hours.

Rules: every item must come from git, a file in the repo, or a session log; nothing from memory. Words a visitor reads must be plain (no ids, no snake_case). The boundary sentence for the page: "Everything outside `prompts/` was written by Codex. The two reference sorters under `prompts/` were ported by Codex."

First version committed by **02:25 UTC** so the page can draw it; then go on with prompt 035 (stations that differ in kind, Nemotron at 448 px, the confusion table).
