{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/}

# Resuming Long Runs

A full benchmark domain contains 50–114 scenarios that run end to end over live WebSocket audio. An
out-of-memory error, interrupted process, or unresponsive large language model (LLM) server can stop a run
before completion.
`run_evaluation.py --resume` continues the existing session instead of starting over.
Resumed sessions preserve existing database (DB)-state evidence for completed scenarios.

## Prerequisites

Keep the original session directory under `eval_results/`, and restart the simulated-user and agent bot
servers before reconnecting the bridge.

## Resume a Run

Both bot servers must be up again before you resume. The bridge reconnects to them per scenario.

```bash
# Terminal 1 + 2: restart the two bots (SERVER_CONFIG_PATH resolves against the CWD, so cd first)
cd evaluation && WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml  python bot_server.py
cd evaluation && WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py

# Terminal 3: resume the session created as eval_results/eval_20260618_072325/
cd evaluation && python run_evaluation.py \
    --domain tau2_retail \
    --resume 20260618_072325
```

`--resume` takes only the timestamp, not a path. The session directory is resolved as
`<--output-dir>/eval_<TIMESTAMP>` (default `--output-dir` is `./eval_results`), and the run aborts with an
error if that directory does not exist. Pass the same scenario selection (`--domain` / `--scenarios`) you used
originally — resume does not remember which scenarios were queued.

## The Three Per-Scenario States

Before running anything, the runner classifies every queued scenario's subdirectory using
`classify_scenario_resume_state` in `nemo_voice_agent/evaluation/resume.py`. The same pass runs on a fresh run
too, where it is a no-op because no subdirectory exists yet.

| State | Detected When | What Happens |
| --- | --- | --- |
| **completed** | `metrics.json` exists, parses, has `total_turns > 0`, and clears the agent-turn threshold | Skipped. Its `metrics.json` is loaded from disk and folded into the run-level aggregate, so the final numbers are identical to a live run. Existing artifacts are untouched. |
| **in-flight** | Subdir exists but `metrics.json` is missing or unreadable, or `total_turns == 0`, or agent turns are below `--min-agent-turns` | Subdir is renamed to `<scenario>.killed.<resume_ts>/`, a `__KILLED__` marker file is dropped inside it, and the scenario is re-run from scratch. |
| **fresh** | No subdir at all | Runs normally. |

The runner does not delete killed backups. They accumulate under the session directory across repeated
resumes and automatic retries, so delete them after you no longer need the partial logs.

At the end of a resumed session, `all_metrics.json`, `all_summary.txt`, and `all_latencies.csv` are rewritten
from scratch covering every scenario, both freshly run and loaded from disk. `evaluation_log.txt` is opened in
append mode and gets a `=== RESUMING run eval_<TIMESTAMP> ===` banner rather than being truncated.

## run_args.json and the Consistency Soft-Check

Every invocation appends a record to `run_args.json` at `<session_dir>/run_args.json`, shaped as
`{"invocations": [...]}`. Each record holds the wall-clock start time, raw `argv`, the fully parsed argument
namespace, and the resolved scenario count and names. `judge_api_key` is replaced with `"<redacted>"` before
the file is written. Resume records also carry `resumed_from_invocation`, the index of the invocation they
continued.

On resume, the runner diffs the new invocation against the most recent prior one across the scoring-relevant
fields below and logs a warning listing every mismatch. It is a **soft** check — it never blocks the run.

| Checked Field | Why It Matters |
| --- | --- |
| `domain`, `scenarios` | Changes which scenarios the aggregate covers |
| `duration` | Different wall-clock budget per scenario |
| `judge_url`, `judge_model`, `judge_max_tokens`, `judge_temperature`, `judge_top_p`, `judge_seed` | Different judge, different verdicts |
| `judge_threshold` | Changes the `JUDGE_PASSED` signal |
| `strict_match` | Changes the action-list comparator |

Output directory and WebSocket URLs are recorded but not diffed. If you do change a scored field mid-session,
the aggregate mixes scenarios graded under different settings. Re-run from scratch (omit `--resume`) when you
need a clean comparison.

## --min-agent-turns

**Default: 2.** Fewer than N LLM responses usually indicate stalled infrastructure rather than a task
failure. For example, the agent can greet the user before the LLM server stops answering. The runner handles
these scenarios separately to keep them from distorting the per-signal rates.

The floor has to sit below the shortest legitimate conversation. Across 500 measured scenarios, the agent
LLM-call distribution bottoms out at exactly 3 calls, where 21 scenarios sit. All 21 are clean exits
(`stop_reason` of `[EXIT]`) and 11 of them succeeded, so a floor of `3` lands inside the legitimate cluster
and fails scenarios that did the work. A floor of `2` sits below it.

The turn count comes from `count_agent_responses`. It first uses the bridge's live
`token_usage.agent.n_calls` from `metrics.json`. For older runs without that field, it counts assistant
messages in `bot_logs_agent/llm_context.json`. When neither signal is available, the check is skipped.

Effects of falling below the threshold:

| Where | Effect |
| --- | --- |
| Composite `is_successful` and `is_task_successful` | Forced to `False` — counted as **failures** |
| Per-signal rates (action-match, DB-state, DB-state-assertion, NL-assertion, judge, clean-exit) | **Skipped**, not counted either way |
| `metrics.json` | Gains `"insufficient_agent_turns": true` |
| `all_summary.txt` and the run log | A warning block naming every stalled scenario, plus the exact `--resume` command to retry them |
| Token usage rollup | Still counted, so cost accounting stays accurate |
| `--resume` classification | Treated as in-flight: moved aside and re-run |

That last row is the point — one resume command both retries the stalled scenarios and cleans the aggregate.
Pass `--min-agent-turns 0` to disable the filter entirely and score every scenario on its own merits.

```bash
# Retry everything that stalled, keeping the default threshold
cd evaluation && python run_evaluation.py --domain tau2_telecom --resume 20260618_072325

# Score every scenario as-is, no stall detection
cd evaluation && python run_evaluation.py --domain tau2_telecom --resume 20260618_072325 --min-agent-turns 0
```

## Automatic In-Run Retry

`--resume` is the manual path you take after a run ends. The runner also retries a scenario on its own,
immediately, when the finished attempt looks like an infrastructure failure instead of a task failure. The
two mechanisms compose: automatic retry catches transient failures during the run, and `--resume` still
handles everything the run never reached.

| Flag | Default | Effect |
| --- | --- | --- |
| `--no-auto-resume-on-stale` | retry is on | Turns off the automatic re-run of a scenario that recorded zero conversation turns. |
| `--auto-resume-on-insufficient-turns` | off | Also re-runs scenarios that merely fall below `--min-agent-turns`. |
| `--max-auto-retries N` | `1` | Caps the automatic re-runs per scenario, so a persistently dead backend cannot loop the run. |
| `--auto-retry-backoff-secs SECS` | `5.0` | Waits this long before the re-run, because an immediate retry against a hung backend is the least likely moment to succeed. |

Zero conversation turns means the two bots never exchanged audio. That attempt contains no measurement, so
re-running it cannot bias the score, which is why this trigger ships enabled.

Falling below `--min-agent-turns` is a different case, and the trigger for it ships disabled. The floor is a
heuristic, and legitimately short scenarios sit on it. Because such a scenario is forced to
`is_successful: false`, a retry converts a guaranteed failure into a fresh draw and pushes the success rate
up. Run-to-run churn within a single arm of these benchmarks measures 12% to 32%, so the bias is
measurable, not theoretical. Enable `--auto-resume-on-insufficient-turns` only when you accept that trade.

The zero-turn trigger reads the bridge's own conversation record rather than `token_usage.agent.n_calls`.
Token usage increments only when the agent bot emits an RTVI token-usage message, so an agent that does not
report usage looks like zero calls on a healthy run. Gating a default-on retry on that counter would
silently re-run every scenario.

Each discarded attempt is kept at `<scenario>.killed.autoretry<N>.<timestamp>/` with a `__KILLED__` marker
file inside, so its logs stay available. `check_resume.py` and a later `--resume` both skip those
directories. The accepted attempt's `metrics.json` records `auto_retry_count`, plus `auto_retry_reasons`
when a retry fired. Read those fields when you compare runs; refer to the
[Metrics Dictionary](../../reference/evaluation/metrics.md) for the field definitions.

```bash
# Turn the automatic re-run off entirely
cd evaluation && python run_evaluation.py --domain tau2_telecom --no-auto-resume-on-stale

# Allow two automatic re-runs per scenario, with a 30-second wait between them
cd evaluation && python run_evaluation.py --domain tau2_telecom \
    --max-auto-retries 2 --auto-retry-backoff-secs 30
```

## Preview with check_resume.py

`check_resume.py` at `evaluation/check_resume.py` reports what a resume *would* do without renaming or writing
anything. It takes a
path to the session directory, not a timestamp.

```bash
cd evaluation && python check_resume.py eval_results/eval_20260618_072325 --min-agent-turns 2
```

It prints counts for completed, would-rerun, and fresh scenarios. It then lists each scenario that would be
rerun and its reason. Reasons include `no metrics.json (in-flight)`, `0 turns (bot crashed before audio)`,
and `1 agent LLM response(s) < 2 (TIMEOUT)`.

A `Completed after an automatic in-run retry (N):` section follows the re-run list whenever the session
contains finished scenarios that the runner retried during the run. Each line reads
`<scenario>  [auto_retry_count=<n>]`. Those scenarios are complete and a resume leaves them alone, so the
section is informational: it tells you which results survived an infrastructure failure on the first
attempt. It does not change the completed, would-rerun, or fresh counts.

Scenarios that were retried and still classify as re-runnable stay in the `Would re-run` list, where their
reason carries the suffix `; already auto-retried <n>x in-run`. Read those first. They are the cases where
the automatic retry ran and did not fix the problem, which usually points at the backend rather than at a
transient failure.

```text
Would re-run (1):
  dom__stuck  [1 agent LLM response(s) < 2 ([TIMEOUT]); already auto-retried 1x in-run]

Completed after an automatic in-run retry (1):
  dom__ok  [auto_retry_count=1]
```

The suffix comes from the same `auto_retry_count` field, so it appears only when the scenario has a readable
`metrics.json`. A scenario that never produced one is listed with its classification reason alone.

Three things to know:

- **Its `--min-agent-turns` defaults to `2`, matching `run_evaluation.py`.** Pass the value you intend to
  resume with whenever you override the default, or the preview misreports the re-run set. Raising the floor
  above `2` is rarely useful, because legitimately short scenarios bottom out at 3 agent responses and a
  floor of `3` or more re-runs real results instead of stalls.
- **Detect retried scenarios through `auto_retry_count` in `metrics.json`**, which is the field this script
  reads. Do not glob for `<scenario>.killed.autoretry*/` directories, because that layout is an
  implementation detail of how failed attempts are preserved.
- It walks only the subdirectories already on disk, skipping `.killed.` backups and anything without a
  `metrics.json`, `bridge_log.txt`, or `scenario_config/`. Queued scenarios that were never started have no
  subdirectory at all, so they do not appear in the preview — the fresh bucket stays empty in practice.

## Tips for Long Runs

Use these practices to control scenario duration, preserve evidence, and avoid unnecessary reruns.

- `--duration` is unset by default, in which case each scenario's own `max_duration` applies. Setting it
  overrides every scenario and is one of the diffed consistency fields.
- Resume in the same shell environment. `--output-dir` is relative to the CWD, so running from a different
  directory silently creates a new session instead of finding the old one.
- Check `all_summary.txt` for the stalled-scenario warning after every long run — that block tells you whether
  the numbers are final or whether a resume is still owed.
- If multiple scenarios stall together, fix the backend first by using [Troubleshooting](../../troubleshooting/index.md).
  Resume after the backend is healthy to avoid repeated failures.

## Next Steps

Use the quickstart for a fresh run, the results guide for artifact triage, and the command-line interface
(CLI) reference for every
resume option.

- [Evaluation Quickstart](quickstart.md) — first end-to-end run
- [Scoring](../understand-scoring/scoring.md) — the six success signals and how the composite is computed
- [Reading Results](results.md) — the artifacts inside each scenario directory
- [Evaluation CLI](../../reference/evaluation/eval-cli.md) — every flag, with defaults
