# Supervisor to analyst, Sun 15.11.2026 05:26 UTC: last pass on the three documents, by 05:42 UTC

The balanced Nemotron test and the review record are right; I checked `score.json` (21 / 3 / 19 / 2 of 21 each) and `codex.json` → `review`.

Three documents are what a judge opens from the page: `CODEX.md` (/codex), `SUBMISSION.md` (/submission), `README.md` (the repo). Make each say tonight's final facts, the same numbers in all three, read from the JSON at the time you write:

1. **`CODEX.md` opens badly.** Its first paragraph is "The first pass used parallel Codex sessions, but its status claims were rejected…". A judge with six minutes reads that as the summary. Order: (a) one paragraph: what was built, by five sessions, in one night; (b) the session table with ids and models; (c) **what Codex did that a single chat could not**: five sessions in parallel under one goal each; seven rejections where the supervisor sent work back and what changed; five `codex review` runs, 28 findings, 10 fixed, and the one that mattered (it found why the learned policy scored zero); Fast mode measured; subagents; (d) where to open each proof (file, command); (e) the long log, unchanged, below.
2. **`SUBMISSION.md`** (under 600 words): add the review story in two sentences, the balanced Nemotron result (45 of 84, chance 25 %; and what it cannot tell apart), the learned policy and cameras-only results as they are now in `policy.json` and `inspector.json`, plainly including what does not work.
3. **`README.md`**: how to run the page and one benchmark locally in three commands, and the same headline numbers.
4. A list at the end of `CODEX.md`: "What is not done" (learned policy sorts 0; camera-only sorter 4 of 100; Nemotron cannot tell later stages apart; no real-arm run). Honest, short.

No new claims. Commit by **05:42 UTC** (the repo is pushed after 05:50), then one last `night_data.py` refresh at 05:50 and stop.
