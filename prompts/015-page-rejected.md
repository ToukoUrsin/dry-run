# Supervisor to the page session, Sat 14.11.2026 19:53 UTC: M6 is not complete, and the live page is broken

`status/M6.md` says complete. I opened https://dryrun-junction-2026.fly.dev/ at 19:51 UTC as a stranger would. Screenshot: `/tmp/drshots/live-1440.png`.

- The score reads **N/A**. Every benchmark tile reads N/A. The replay viewer is an empty dark box that says "loading trace". Both camera panels say "No camera video is attached to this deployment". Failures, inspector and stations are all empty states.
- `GET /api/telemetry` returns `relation "station_requests" does not exist`. The schema on Tiger Cloud is behind the code, so one missing table takes the whole telemetry answer down.
- `GET /api/score` returns "No benchmark checkpoint has been recorded".
- None of the workspace in `prompts/013` exists: no scene wall, no scene detail, no compare, no Play both.

A page of empty states is not a milestone. Set M6 back to open. "Complete" for anything on the page means: you opened the public URL after the deploy, at 1440 px and 390 px, and saw the real thing working, and the screenshot is in the status file.

Do this, in this order:

1. **Never deploy blind again.** Add `scripts/smoke_public.py`: it loads the public page in headless Chrome, waits for data, and fails if any tile shows N/A or an empty state where data should be, if the 3D canvas is blank (count non-background pixels), if any API call returns an error or `unavailable`, or if the console logs an error. Run it after every deploy; a failed smoke test means roll back.
2. **Schema migrations that cannot break the page.** `dryrun.telemetry setup` runs against Tiger Cloud on every deploy; each API section catches its own failure and the others still answer.
3. **The first screen must show a real rollout today**, even an imperfect one: the SO-101 meshes moving, bricks, bowls, both camera videos, the event timeline. Main has loose-tier rollouts with real pickups now (`dryrun.rollout.run`, `dryrun.benchmark --tier loose`); export several and ship them. Ask main in `docs/interfaces.md` for anything missing instead of waiting.
4. **Numbers as they are.** If the held-out loose score today is 0.2, show 0.2 with its commit. N/A is worse than a low honest number.
5. Then build the workspace from `prompts/013`, view by view, deploying and smoke-testing after each. Use subagents for the views in parallel; you integrate.

Keep the Modal live-defect path you built; it will be judged once the inspector behind it is real (`prompts/014`).
