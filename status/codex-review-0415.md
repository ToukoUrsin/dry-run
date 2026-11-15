# Real Codex review — 2026-11-15 04:15 UTC

Command:

```text
codex -c 'mcp_servers={}' review --base 40d23d4
```

The review ran as a fresh non-remote process with MCP servers disabled. Codex
reported:

```text
OpenAI Codex v0.169.0
workdir: /home/ubuntu/junction/dryrun
model: gpt-6.1-sol
provider: cliproxy
approval: never
sandbox: workspace-write
reasoning effort: medium
session id: 01a1dd6a-74f3-7be2-b5de-008b0a2789ec
user: changes against '40d23d4'
2026-11-15T04:15:37.972719Z ERROR codex_core::tools::router:
  error=apply_patch verification failed: Failed to read
  /tmp/codex-bwrap-synthetic-mount-targets-1000: path
  `/tmp/codex-bwrap-synthetic-mount-targets-1000` is not a file
codex
No actionable findings identified.
No actionable findings identified.
```

The process exited successfully. The sandbox mount warning produced no review
finding, and there are no finding-to-fix commits.


## Prompt 052 benchmark/seeds review

Command:

```text
codex --no-daemon --dangerously-bypass-approvals-and-sandbox -m gpt-6.1-sol -c model_reasoning_effort=high -c 'mcp_servers={}' exec --ephemeral [review prompt from /bulk3/runs/dryrun/codex-review-052/benchmark-task.txt]
```

Session id: `01a1dd71-53f5-7721-943f-b47972b1ebaf`
Scratch worktree: `/tmp/dryrun-review-benchmark-0422`; exit 0; wall time 261.88 s.

Verbatim findings:

- **High — `dryrun/benchmark.py:250-257`**: `--max-steps` is ignored for expert runs. Thread it through `run()`/`_one()`/`rollout_run()` and record it in results.

- **High — `dryrun/benchmark.py:251-255`**: policy path ignores `--tier`; `--tier pile` still evaluates canonical aggregate groups. Pass the tier through and emit/validate the selected partition.

- **High — `dryrun/benchmark.py:228-231`**: held-out partitioning is positional and unchecked. `run(range(200), tier="pile")` selects `0..49` instead of `150..199`; reject duplicates and validate fixed tier partitions.

- **Medium — `dryrun/benchmark.py:193-196`**: legacy `picked_by_contact` fallback sums truthy rows and can inflate contact counts. Deduplicate by physical brick/event.

## Prompt 052 camera-only sorter review

Command:

```text
codex --no-daemon --dangerously-bypass-approvals-and-sandbox -m gpt-6.1-sol -c model_reasoning_effort=high -c 'mcp_servers={}' exec --ephemeral -o /bulk3/runs/dryrun/reviews/camera-sorter-review-final.txt
```

Session id: `01a1dd75-2738-77b1-9ace-7dd62881a43d`

Verbatim findings:

Read-only review complete. No `dryrun/camera_sorter*.py` files exist; the sorter is `dryrun/camera_expert.py`. No edits, commits, or expensive rollouts were made; the worktree is clean.

### Critical

- **Grasp detection is effectively using a stale, inverted gripper test.** `Q_OPEN=1.0` and `Q_CLOSE=0.2`, but `camera_expert.py:206-210` declares a miss when the reading is `> .88`. `env.py:63-82` refreshes observations only every 30 ticks; the close plus hold lasts 27 ticks, so `_latest_obs` is commonly still an open-jaw reading. The debug artifact confirms readings `1.004`, `0.934`, and `1.001` with accurate XY/yaw estimates: `/bulk3/runs/dryrun/m2_rebuild/camera_grip_miss_debug_053.json:36-39,70-73,102-105`. This explains false “miss” events and invalid `gripped` totals.

- **Inspection control is conditioned on hidden simulator brick positions.** `defects.py:135-148` scans brick bodies and reads `data.xpos` to select the nearest brick when `focus_body is None`; `env.py:79-80` supplies that frame, and `rollout.py:57-62` feeds its inspector verdict back into bowl control. This violates the prompt-032 requirement to use the known presentation point without selecting the crop from a true body pose. It also undermines the status claim that the camera-only run is state-free (`status/M2.md:51-56`).

### High

- **The camera grasp height is about 4.4 mm too high.** Perception reports the top surface at `z=.0224` (`perception.py:275,329`), while the camera sorter uses a hard-coded grasp target `z=.0174` (`camera_expert.py:194-196`). The reference expert targets about `.0130` (`expert.py:20,190-197`), and the debug artifact explicitly identifies the discrepancy as a height issue (`camera_grip_miss_debug_053.json:5,39,73,105`).

- **Post-drop looks are taken with the arm over the table, and the first frame/joint mask can be mismatched.** Both the pre-cycle and post-drop “neutral” motions target `(0.22, 0.0, 0.14)` (`camera_expert.py:193,232`), directly over the work area. The perception mask projects the current arm joints over the image (`perception.py:222-265`), while the top frame is cached independently at 1 Hz (`env.py:63-82`). This can mask bricks or apply a current-arm mask to an older frame, producing false empty looks and premature termination.

- **Perception failures are still treated as empty tables.** `_refresh` catches every exception and returns `[]` (`camera_expert.py:163-177`); two such misses terminate the sorter (`camera_expert.py:252-255`). Logging the exception does not prevent detector/calibration errors from being counted as “no brick.”

- **Immediate IK failure has no in-controller recovery and the evaluation wrapper spins.** `_cycle` returns before yielding on IK failure (`camera_expert.py:197-200`), while `act` assumes `next(self._motion)` yields an action (`camera_expert.py:248-251`). The Modal wrapper catches `StopIteration`, resets to `look`, and immediately retries (`/bulk3/runs/dryrun/m2_rebuild/camera_sweep_modal.py:41-46`). Seed 14 consequently produced 1,110 `ik_fail` records, zero reaches, and ended at 1,110 steps (`camera_stage_sweep_20_focus.json:7654-7679`). There is also no failed-token set; the nearest detection is selected again on every look (`camera_expert.py:248-250`).

- **World/image conversion assumes square frames.** Horizontal projection and back-projection use `height_px` as the x scale (`perception.py:102-105,112-118`), and bowl radius uses the same scalar for both axes (`perception.py:211-218`). This is correct for the square 720/960 evaluation images but wrong for a real rectangular 1080p frame.

### Medium

- **The focus sweep counts an environment cap as a valid completed row.** `StationEnv` forces `done` at `_max_ticks` (`env.py:44,127`). Seed 14 hit exactly 1,110 steps with repeated IK failures, yet the focus artifact reports it among 20 rows and the status says the run covered 100 valid scene bricks (`status/M2.md:185-193`). The earlier merged yaw artifact correctly classified seed 14 as a timeout and used a 97-brick completed denominator.

- **Perception metrics are aggregated at the scene-statistic level, not the brick level.** `/bulk3/runs/dryrun/m2_rebuild/eval_perception_200.py:11-12` averages per-scene detection rates and takes percentiles of per-scene medians/p95s. The artifact reports 883 matched bricks but a detection rate of `0.880333` (`perception_eval_200.json:7-11`); 883 matches out of 999 truth bricks is `0.883884`. The status wording “883 matched bricks ... detection rate 88.03%” (`status/M2.md:90-93`) is therefore not a single consistent metric.

- **The tests do not exercise these paths.** `tests/test_camera_only.py:52-68` performs one action on blank images and only guards forbidden qpos/body-array access. It does not test rectangular calibration, stale gripper readings, inspection-frame selection, repeated IK failures, look termination, or stage accounting.

The corrected yaw implementation itself has no remaining symmetry/sign finding: `perception.py:192-208,324-328` and `evaluate_perception()`’s size-aware periods (`perception.py:351-365`) agree with the yaw audit’s raw transform; the reported `0.192°` median is supported. Overall findings remain.

Nothing needs Touko; this is the completed review record.

## Prompt 052 learned-policy loop review

Command:

```text
codex --no-daemon --dangerously-bypass-approvals-and-sandbox -m gpt-6.1-sol -c model_reasoning_effort=high -c 'mcp_servers={}' exec --ephemeral [learned-policy review prompt]
```

Session id: `01a1dd75-1f15-7d70-a814-44da7b471dbf`
Scratch worktree: `/tmp/dryrun-policy-review-clone.QuVa6Y`.

Verbatim findings:

The two strongest confirmed causes are the temporal/action-contract bugs:

Confirmed bugs
- Recorded action is paired with the wrong observation. dryrun/rollout.py:46-50 chooses action from pre-step obs, then replaces obs with env.step(action) output; dryrun/rollout.py:65 records post-step observation alongside old action. Same pairing persists modal/rollouts.py:197-215 and dryrun/demos.py:172-180. Training learns action_t conditioned on approximately obs_{t+1}, evaluation predicts from obs_t. modal/eval_action_mae.py:25-35 measures misleading same-row pairing; scripts/replay_recorded_actions.py:84-85 implicitly uses correct previous-state comparison. Could explain zero success: yes, likely.
- Action chunks are fabricated, not temporal. Every frame's single action is repeated across the chunk in dryrun/train_policy.py:39-43, modal/train_policy.py:43-44, modal/continue_train.py:38-41, modal/train_shards.py:82-88; all repeated elements marked non-padding. Evaluation consumes 30 actions open-loop from a 60-step prediction in dryrun/policy_benchmark.py:44-54. Recorded runs used open_loop_ticks=30 (status/M4.md:208-216, 242-249). Model trained to emit constant hold while execution assumes meaningful trajectory. Could explain zero success: yes; strongest explanation for 30-tick maps.
- Cropped evaluation feeds inspection camera at wrong size. crop_top=True makes top exact model size (dryrun/policy_benchmark.py:24-26), then resize condition :29-31 skips inspection resize, leaving raw render size. Crop run trained at 224 but evaluated render_size=320,crop_top=true (data/page/policy.json:435-438,460-463; dataset 224 at status/M4.md:232). Could explain cropped zeros, not earlier full-224 zeros.
- Policy rollouts never drive active-brick/verdict path. dryrun/env.py:96-124 emits grasp/lift physics events only when _active set; _active/present/inspection forcing set only by expert_action() at env.py:130-141. Learned calls use dryrun/rollout.py:9-15 and never expert_action(). rollout.py:57-65 keeps verdict local; ACTController reads obs.get(verdict) at policy_benchmark.py:32-37, so zero forever. This makes policy stage telemetry and defect routing invalid; can cause wrong-bowl failures, but cannot alone explain zero physical reach/successes because StationEnv.result computes placement separately.
- 0.06-radian clamp only in learned/replay control (policy_benchmark.py:41-61); expert recordings store unclamped target (rollout.py:65, modal/rollouts.py:211). StationEnv.step only actuator-range clipping (env.py:89-94). replay_recorded_actions.py:40-49 confirms separate target-vs-sent transformation. Could explain zero if many transitions exceed 0.06; clipping frequency unavailable, so unverified.
- OpenVINO is not behaviorally equivalent: OpenVINOController neither resizes inspection nor applies per-tick clamp (policy_benchmark.py:70-78). Could explain OpenVINO-only failure, not ACT maps.
Hypotheses: camera refresh every 30 ticks shared train/eval (env.py:63-82), JPEG quality 78 vs raw eval likely small; no joint-order bug (ARM_NAMES/expert/env/XML agree); no scale mismatch (images /255 eval and dataset [0,1], raw radians both), though LeRobot processor/statistics bypassed in both.
Full-resolution run was also zero, so crop-size issue is not sole cause. Temporal state/action pairing plus fabricated chunks are most capable causes.

## Prompt 052 Tiger/web review

Command:

```text
codex --no-daemon --dangerously-bypass-approvals-and-sandbox -m gpt-6.1-sol -c model_reasoning_effort=high -c 'mcp_servers={}' exec --ephemeral [Tiger/web review prompt]
```

Session id: `01a1dd74-09d4-7913-8260-ad233e0760bf` (exec session `60698`)
Scratch worktree: `/home/ubuntu/junction/tiger-codex-review.zVCm4E`.

Verbatim findings:

Verdict: changes requested. The live API suppresses slow query timings, but the review found one high severity data integrity issue and several medium severity correctness/disclosure issues.

High: telemetry.py:82-84 precheck is racy; schema.sql:61-77 only PK(finished_at,rollout_id), so concurrent retries can duplicate COPY rows/results and multiply failure counts. Medium: schema.sql:82-89 groups benchmark_tier by day but telemetry.py:199-216 omits bucket/no reaggregation, yielding duplicate/non-total tier rows across days; telemetry.py:289-295 sync() defaults source_url to DEFAULT_URL, bypassing DRYRUN_DB_URL; schema policy add_* failures are swallowed at schema.sql:30-33,56-59,90-93,159-162; web/app.py:261-271 catches any dashboard exception and serves stale snapshot HTTP 200; raw exception text and absolute filesystem paths leak at web/app.py:173,229-238,271,593,614,635-637,217,258,409-417 and telemetry.py:245-254; artifact fallback can expose >300ms persisted query_ms (web/app.py:267-269, score_payload:372); telemetry.py:89 writes outcome=None so rollout_1s.status is always NULL. No files changed/committed/deployed/heavy work.

## Prompt 052 decisions and handoff

- Benchmark/seeds: agree with all four findings. `--max-steps` now reaches expert rollouts and is recorded; policy tier selection is explicit; duplicate and noncanonical partitions are rejected; legacy contact rows are deduplicated. Fixed in commit `6471dbe`.
- Tiger/web: agree with all listed findings. Same-rollout writes are serialized and protected by a unique rollout-id index; tier rows are reaggregated across daily buckets; `sync()` honors `DRYRUN_DB_URL`; aggregate policy failures no longer hide non-duplicate errors; stale/error responses are marked and surfaced without HTTP 200; local paths and exception text are removed from public payloads; persisted query timings are capped at 300 ms; 30 Hz rows carry a non-null status. Fixed in commit `6471dbe`.
- Camera sorter: agree with the actionable findings and handed them to the camera session in `status/M2_REVIEW_HANDOFF_052.md`. No camera source was changed in this session because that session owns the sorter and its evaluation artifacts.
- Learned policy: agree with the confirmed action/observation, chunk, resize, and active-brick findings and handed them to the policy session in `status/M4_REVIEW_HANDOFF_052.md`. No policy training-loop source was changed in this session because that session owns the policy experiments.

### Raw Tiger/web review output (verbatim)

```text
codex
Verdict: **changes requested**. The live API suppresses slow query timings, but the review found one high severity data integrity issue and several medium severity correctness/disclosure issues.

| Severity | Finding |
|---|
| **High** | **Rollout deduplication is racy and unenforced.** [`telemetry.py:82-84`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/dryrun/telemetry.py:82) checks `rollout_id` before inserting, but [`schema.sql:61-77`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:61) only keys results by `(finished_at, rollout_id)`. Concurrent retries can both COPY telemetry/events and insert separate results; [`schema.sql:106-111`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:106) then multiplies failure counts. |
| **Medium** | **Tier API returns daily rows as separate tiers.** [`schema.sql:82-89`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:82) groups `benchmark_tier` by day, but [`telemetry.py:199-216`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/dryrun/telemetry.py:199) does not aggregate or include `bucket`. Multiple days produce duplicate tier/version entries and non-total metrics. |
| **Medium** | **`sync()` ignores the configured source database.** [`telemetry.py:289-295`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/dryrun/telemetry.py:289) defaults `source_url` to `DEFAULT_URL`, bypassing `DRYRUN_DB_URL`; CLI sync can therefore copy from the wrong database. |
| **Medium** | **Continuous-aggregate policy failures are silently discarded.** All `add_*_policy` calls are wrapped in `EXCEPTION WHEN OTHERS THEN NULL` ([`schema.sql:30-33`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:30), [`56-59`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:56), [`90-93`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:90), [`159-162`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:159)). A permission/version failure leaves aggregates stale with no setup error; only an explicit [`refresh()`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/dryrun/telemetry.py:184) call repairs them. |
| **Medium** | **Telemetry fallback masks unrelated failures with stale data.** [`web/app.py:261-271`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:261) catches any dashboard exception and serves `snapshot.json` with HTTP 200. A broken compression query, diagnosis query, or station query can make `/api/telemetry`, `/api/score`, and `/api/failures` report old data rather than an isolated error. |
| **Medium** | **Public responses expose raw internal errors and filesystem metadata.** Examples include raw exception text in [`web/app.py:173`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:173), [`229-238`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:229), [`271`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:271), and POST handlers [`593`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:593), [`614`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:614), [`635-637`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:635). Public payloads also return absolute paths via `cast_path`, artifact `source`, and diagnosis `frame_paths` ([`217`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:217), [`258`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:258), [`409-417`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:409), [`245-254`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/dryrun/telemetry.py:245)). |
| **Low** | **Slow timing suppression is incomplete on the artifact path.** Live timings are gated at 300 ms ([`telemetry.py:264-270`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/dryrun/telemetry.py:264)), but [`web/app.py:267-269`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:267) returns persisted `query_ms` unchanged when falling back to `snapshot.json`; [`score_payload`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/web/app.py:372) forwards it publicly. |
| **Low** | **The loader never populates aggregate status.** [`telemetry.py:89`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/dryrun/telemetry.py:89) writes `outcome=None` for every telemetry sample, while [`schema.sql:48-53`](/home/ubuntu/junction/tiger-codex-review.zVCm4E/data/schema.sql:48) derives `status` from `outcome->>'status'`; `rollout_1s.status` is therefore always NULL. |

No files were changed, committed, deployed, or subjected to heavy jobs. Nothing needs Touko.
```
