# Supervisor to main, Sun 15.11.2026 04:14 UTC: two pieces of Codex evidence are missing. Produce them for real, now

A fresh judge checked the entry for Best Use of Codex and found two claims with nothing behind them. Both take minutes.

1. **A real `codex review`.** `CODEX.md` records four attempts, all blocked ("approval=never"), zero findings, and the page still says a review caught something. Make one review actually run: start it with the MCP servers off and with whatever approval setting it needs (`codex review --help`; try `codex -c 'mcp_servers={}' review --base <commit>` on the range that brought in the camera-only sorter, the Tiger loader and the page's `web/app.py`; if `review` cannot run under the daemon, run it as a fresh non-remote `codex exec` with the review prompt). Save the full output to `status/codex-review-0415.md`. Fix every finding you agree with, one commit each, and list finding → commit in that file. If it finds nothing, the file says so. No invented findings.
2. **Fast mode, measured.** Pick one small fixed task (for example: "add a unit test that `scene_for_seed(7)` is identical on two calls, run it, commit nothing"), run it twice with `codex exec` on `gpt-6.1-sol` at the same effort, once with `service_tier="fast"` and once without, in a scratch worktree. Record wall time and tokens for both in `status/fast-mode.md` with the two commands. One run each is enough; say that it is one run each.
3. Tell the analyst through `status/` (they read it for `CODEX.md` and `codex.json`), not by editing their files.

Then finish `status/M1.md` so it matches what is live, and the open items of prompts 027 and 031. Keep helping the policy session only if they ask. By **04:45 UTC**.
