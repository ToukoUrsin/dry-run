# Fast-mode comparison — 2026-11-15 04:17 UTC

This is one successful run in each mode, using the same `gpt-6.1-sol` model,
`medium` reasoning effort, prompt, and isolated scratch worktree
`/tmp/dryrun-fast.gXpkl2`. The task was read-only: inspect
`dryrun/mjcf_station.py`, call `layout_bricks(7, "loose bricks")` twice, and
compare the serialized results. No files were changed and no commit was made.

The successful commands were:

```text
codex --no-daemon --dangerously-bypass-approvals-and-sandbox \
  -m gpt-6.1-sol -c model_reasoning_effort=medium -c 'mcp_servers={}' \
  exec --ephemeral -C /tmp/dryrun-fast.gXpkl2 --json - \
  < /bulk3/runs/dryrun/codex-fast/task.txt

codex --no-daemon --dangerously-bypass-approvals-and-sandbox \
  -m gpt-6.1-sol -c model_reasoning_effort=medium \
  -c service_tier='"fast"' -c 'mcp_servers={}' \
  exec --ephemeral -C /tmp/dryrun-fast.gXpkl2 --json - \
  < /bulk3/runs/dryrun/codex-fast/task.txt
```

| mode | wall time | input tokens | output tokens | reasoning tokens | result |
|---|---:|---:|---:|---:|---|
| default (no service tier flag) | 28.53 s | 77,078 | 1,115 | 415 | `identical: True`, 5 bricks |
| `service_tier="fast"` | 22.42 s | 56,076 | 827 | 172 | `identical: True`, 5 bricks |

Raw JSONL, stderr, exit markers, and `/usr/bin/time` records are under
`/bulk3/runs/dryrun/codex-fast/`. A preliminary normal-mode attempt using the
read-only sandbox was excluded from the table because the sandbox failed before
the task (`synthetic mount registry must not be a symlink`); it is retained as
`normal.jsonl` and `normal.stderr`.

This is a one-run comparison, not a statistical benchmark. On this pair, Fast
completed 6.11 seconds sooner (21.4% lower wall time), with fewer billed input
and output tokens; the task result was the same.
