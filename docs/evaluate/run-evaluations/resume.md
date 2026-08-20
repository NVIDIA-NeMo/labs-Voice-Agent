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

# Resuming & Long Runs

A full benchmark domain is 50–114 scenarios run end to end over live WebSocket audio, so a run can take hours
and can die halfway through — a hung LLM server, an OOM, or a deliberate Ctrl+C. `run_evaluation.py --resume`
picks the session back up instead of starting over.

## Resume a run

Both bot servers must be up again before you resume; the bridge reconnects to them per scenario.

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

## The three per-scenario states

Before running anything, the runner classifies every queued scenario's subdirectory using
`classify_scenario_resume_state` in `nemo_voice_agent/evaluation/resume.py`. The same pass runs on a fresh run
too, where it is a no-op because no subdirectory exists yet.

| State | Detected when | What happens |
| --- | --- | --- |
| **completed** | `metrics.json` exists, parses, has `total_turns > 0`, and clears the agent-turn threshold | Skipped. Its `metrics.json` is loaded from disk and folded into the run-level aggregate, so the final numbers are identical to a live run. Existing artifacts are untouched. |
| **in-flight** | Subdir exists but `metrics.json` is missing or unreadable, or `total_turns == 0`, or agent turns are below `--min-agent-turns` | Subdir is renamed to `<scenario>.killed.<resume_ts>/`, a `__KILLED__` marker file is dropped inside it, and the scenario is re-run from scratch. |
| **fresh** | No subdir at all | Runs normally. |

The killed backups are never deleted for you — they accumulate under the session directory across repeated
resumes. Delete them yourself once you no longer need the partial logs.

At the end of a resumed session, `all_metrics.json`, `all_summary.txt`, and `all_latencies.csv` are rewritten
from scratch covering every scenario, both freshly run and loaded from disk. `evaluation_log.txt` is opened in
append mode and gets a `=== RESUMING run eval_<TIMESTAMP> ===` banner rather than being truncated.

## `run_args.json` and the consistency soft-check

Every invocation appends a record to `<session_dir>/run_args.json`, shaped as
`{"invocations": [...]}`. Each record holds the wall-clock start time, raw `argv`, the fully parsed argument
namespace, and the resolved scenario count and names. `judge_api_key` is replaced with `"<redacted>"` before
the file is written. Resume records also carry `resumed_from_invocation`, the index of the invocation they
continued.

On resume, the runner diffs the new invocation against the most recent prior one across the scoring-relevant
fields below and logs a warning listing every mismatch. It is a **soft** check — it never blocks the run.

| Checked field | Why it matters |
| --- | --- |
| `domain`, `scenarios` | Changes which scenarios the aggregate covers |
| `duration` | Different wall-clock budget per scenario |
| `judge_url`, `judge_model`, `judge_max_tokens`, `judge_temperature`, `judge_top_p`, `judge_seed` | Different judge, different verdicts |
| `judge_threshold` | Changes the `JUDGE_PASSED` signal |
| `strict_match` | Changes the action-list comparator |

Output directory and WebSocket URLs are recorded but not diffed. If you do change a scored field mid-session,
the aggregate mixes scenarios graded under different settings; re-run from scratch (omit `--resume`) when you
need a clean comparison.

## `--min-agent-turns`

**Default: 3.** A scenario in which the agent produced fewer than N LLM responses almost always means the
infrastructure stalled — the agent greeted the user and the LLM server stopped answering — not that the agent
failed the task. Scoring those on their merits pollutes the per-signal rates, so they are handled separately.

The turn count comes from `count_agent_responses`, which prefers the bridge's live-accumulated
`token_usage.agent.n_calls` in `metrics.json` and falls back to counting assistant messages in
`bot_logs_agent/llm_context.json` for older runs that predate that field. When neither signal is available the
check is skipped.

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

## Preview with `check_resume.py`

`evaluation/check_resume.py` reports what a resume *would* do without renaming or writing anything. It takes a
path to the session directory, not a timestamp.

```bash
cd evaluation && python check_resume.py eval_results/eval_20260618_072325 --min-agent-turns 3
```

It prints a completed / would-re-run / fresh tally and then lists each scenario that would be re-run together
with the reason (`no metrics.json (in-flight)`, `0 turns (bot crashed before audio)`,
`1 agent LLM response(s) < 3 (TIMEOUT)`, and so on).

Two things to know:

- **Its `--min-agent-turns` defaults to `0`, unlike `run_evaluation.py`'s `3`.** Always pass the value you
  intend to resume with, or the preview under-reports the re-run set.
- It walks only the subdirectories already on disk, skipping `.killed.` backups and anything without a
  `metrics.json`, `bridge_log.txt`, or `scenario_config/`. Queued scenarios that were never started have no
  subdirectory at all, so they do not appear in the preview — the fresh bucket stays empty in practice.

## Tips for long runs

- `--duration` is unset by default, in which case each scenario's own `max_duration` applies. Setting it
  overrides every scenario and is one of the diffed consistency fields.
- Resume in the same shell environment. `--output-dir` is relative to the CWD, so running from a different
  directory silently creates a new session instead of finding the old one.
- Check `all_summary.txt` for the stalled-scenario warning after every long run — that block tells you whether
  the numbers are final or whether a resume is still owed.
- If many scenarios stall at once, fix the backend first (see [Troubleshooting](../../troubleshooting/index.md))
  and only then resume; resuming into a still-broken LLM server just re-kills the same scenarios.

## Related pages

- [Evaluation Quickstart](quickstart.md) — first end-to-end run
- [Scoring](../understand-scoring/scoring.md) — the six success signals and how the composite is computed
- [Reading Results](results.md) — the artifacts inside each scenario directory
- [Evaluation CLI](../../reference/evaluation/eval-cli.md) — every flag, with defaults
