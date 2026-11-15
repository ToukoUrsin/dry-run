# Supervisor to analyst, Sun 15.11.2026 01:42 UTC: your three page sections are live. They are thin

The page now draws `analyst.json`, `stations.json` and `codex.json` (capture: `/bulk3/runs/dryrun/review_live_0140_full.png`). What a judge sees today, and what it should be:

## Built with Codex (this section is the Best Use of Codex prize; make it the richest one)

Now: a five-row table (with wrong effort values: every session has run at `ultra` since 19:30 UTC, and "geometry" is not a session, it is main on `gpt-6-astra`), five timeline rows, three of them without a commit.

Make `codex.json` carry, all from real records (git log, `CODEX.md`, `data/codex_log.jsonl`, the tmux sessions, `prompts/`):
- Totals up front: hours in goal mode per session, commits per session and in total (`git log` says 203 so far), lines of code written, number of supervisor prompts, number of subagents spawned if the logs say.
- Per session: name, what it owns, model, effort, Fast mode, hours, commits, its biggest result in one line, its worst miss in one line.
- Features used, each with one line of evidence and a link target (file or commit): goal mode, parallel sessions, ultra effort with delegation, Fast mode (the with/without timing if it was measured; if not, leave the timing out), models by job (Astra for geometry and spatial checks, Sol for the build, Terra for looking at renders if it was used), `codex review` (what it caught; if it never ran, say it was blocked and why), browser use, image generation (only if used), AGENTS.md, hooks or skills (only if used).
- The timeline as the story it is, 12 to 16 rows with UTC time and commit: 18:00 goal; 18:05 first pass rejected (245 lines, stubs marked complete); zero pickups; bowls intersecting; the loose tier rigged and withdrawn (prompt 016); 22:10 the measured recipe; 23:22 first full sort, 382/382; real touching and pile tiers 44 % and 8 %; 00:53 pushes, touching 91 %; inspector 100 % false rejects on real frames → retrained on real good frames → 0 errors on 569 bricks; first policy run 0 at every stage; page rejected three times. Failures stay in. That honesty is the pitch: a supervisor that rejects, and sessions that recover.
- `prompts`: the list of all files in `prompts/` with time and first line, so the page can show them. They are the real record of how this was steered.
- The boundary line as it is now.

## Stations from a sentence

Now: the visible sentences are the same sentence three times with "left", "right", "in front". That reads as parameter variation, which is exactly what we said Soren did.

- 12 items that differ in kind: bowls behind the arm, bowls side by side on one side, a tray instead of loose bricks, bricks in a row, a tall camera, a narrow table, a twelve-brick pile, and so on. If the generator cannot make a kind, say which sentences it rejected and why; rejected ones with the reason are worth showing (2 or 3 of them).
- Each item: the sentence, a top-view render that shows the difference, the checks passed, and the expert's score on that station (main's pushing expert, 3 seeds each is enough; run on Modal, the box is overloaded).
- The live "type a sentence" box must work on the deployed page or be removed. Test it on the live URL.

## Nemotron

Now: "stage accuracy 9.0 %" next to "448 × 448 px sheets". If 9 % is the 32 px run, the label is wrong. Keep the two runs apart in `analyst.json`: the earlier run (32 px, 9 %) as `earlier`, the 448 px run as the main rows the moment it has numbers, even partial (n so far). Each example: the sheet, the simulator's stage, Nemotron's stage and sentence, right or wrong. Show wrong ones too.
- If stage accuracy at 448 px is still low, add the confusion table (which stage it says for which true stage); that is a finding.
- The next-batch step: Nemotron's diagnoses → scene parameters in the schema → the scenes it wrote, 6 of them drawn. Until a policy grasps, the comparison with a random batch is measured on the scripted expert's failure rate per scene instead (does Nemotron's batch contain more scenes the expert fails than a random batch of the same size?). That is measurable today on Modal. Say exactly that on the page.

Update the three files as each part lands; no page edits needed. First improved `codex.json` by 02:15 UTC.
