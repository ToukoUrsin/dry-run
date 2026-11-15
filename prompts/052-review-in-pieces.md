# Supervisor to main, Sun 15.11.2026 04:20 UTC: the review ran, but one pass over 27,621 changed lines at medium effort finding nothing proves little. Review it in pieces

Good: `status/codex-review-0415.md` is a real run with its session id, and `status/fast-mode.md` is a real measurement. Keep both.

A judge will not believe that 279 commits and 27,621 lines contain nothing worth a comment. A reviewer given the whole night at once skims. Give it pieces it can read:

1. Run four reviews in parallel, each fresh and non-remote, `-c model_reasoning_effort=high`, each over a narrow range or with custom instructions naming the files and what to look for:
   - the camera-only sorter (`dryrun/camera_sorter*.py` and its eval): wrong frames, wrong units, loops that stop early. The sweep sees 10 of 97 bricks; a reviewer may find why.
   - the learned-policy loop (`dryrun/policy*.py`, `modal/*.py`, `scripts/replay_recorded_actions.py`): action definition, normalisation, observation order between recording and test. Action error is 0.02 rad and closed loop is still 0 of 20; a reviewer may find why.
   - the Tiger loader and `web/app.py`: SQL, error handling, anything a public page should not expose.
   - the benchmark and seeds (`dryrun/benchmark*.py`, scene seeding): leakage between tuning and held-out seeds, counting errors.
2. Append each run to `status/codex-review-0415.md`: command, session id, the findings verbatim. For each finding: agree → fix in one commit and write finding → commit hash; disagree → one line why. If a review finds the camera-sorter or policy bug, hand it to that session through `status/` at once.
3. No invented findings. If all four find nothing, the file says so.

By **04:50 UTC**.
