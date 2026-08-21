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

# Reading Results

Every `run_evaluation.py` invocation writes a timestamped session directory under `--output-dir`
(default `./eval_results`). Session-level aggregates sit at the top; each scenario gets its own
subdirectory named after the scenario. This page maps each artifact to the question it answers.

## Directory layout

Each run directory combines session-level aggregates with one evidence directory per scenario.

```text
eval_results/eval_YYYYMMDD_HHMMSS/
├── evaluation_log.txt            # runner log for the whole session
├── run_args.json                 # CLI invocation history (appended on --resume)
├── all_metrics.json              # JSON array: every scenario's metrics.json, concatenated
├── all_latencies.csv             # one CSV row per latency measurement
├── all_summary.txt               # human-readable per-scenario + run-level summary
└── <scenario_name>/
    ├── metrics.json              # per-scenario scores, turns, latency, token usage
    ├── judge_result.json         # LLM judge output (written whenever the judge ran)
    ├── final_agent_response.json # actions the bots actually performed
    ├── final_scenario_db_hash.txt# post-run DB hash(es)
    ├── conversation_log.txt      # timestamped transcript with latency annotations
    ├── conversation_log.seglst.json  # segLST speaker segments
    ├── conversation_log.wav      # stereo audio, L = user→agent, R = agent→user
    ├── bridge_log.txt            # bridge debug log for this scenario
    ├── bot_logs_agent/llm_context.json  # agent bot's full LLM context
    ├── bot_logs_user/llm_context.json   # user-sim bot's full LLM context
    └── scenario_config/          # snapshot of the scenario as it was run
        ├── metadata.json
        ├── reference_answer.json
        ├── agent_prompt.txt
        ├── agent_tools.json
        ├── user_prompt.txt
        └── user_tools.json
```

## Which file answers which question

Start with the artifact that most directly answers your investigation question.

| Question | File |
|---|---|
| What was the headline success rate? | `all_summary.txt` |
| Which signal dragged a scenario down? | `<scenario>/metrics.json` → `success_breakdown` |
| Why did the judge score it that way? | `<scenario>/judge_result.json` |
| What tool calls did the agent actually make? | `<scenario>/final_agent_response.json`, `bot_logs_agent/llm_context.json` |
| Why did `db_state_match` fail? | `final_scenario_db_hash.txt` vs `scenario_config/metadata.json` → `expected_db_hash` |
| What did the two bots say to each other? | `conversation_log.txt`, `conversation_log.wav` |
| What was recognized versus what was spoken? | `conversation_log.seglst.json`, both `llm_context.json` files |
| Did the bridge time out, disconnect, or exit cleanly? | `metrics.json` → `stop_reason`; `bridge_log.txt` |
| Exactly which prompt and tools were used? | `scenario_config/` |
| What flags produced this run? | `run_args.json` |

## Session-level files

**`all_summary.txt`** is the first thing to read. It contains, in order: total scenarios / duration /
turns; a per-scenario block (composite `Is successful`, task-success excluding clean exit, each
applicable signal, turn count, duration, latency stats); overall latency statistics; the stalled-run
warning; the composite success rate; a per-signal pass-rate table; a token-usage rollup; and, when the
run spans more than one domain, per-domain breakdowns.

Two lines deserve special attention:

- `WARNING: N scenario(s) had fewer than 3 agent turn(s) ... and were counted as failures.` — the
  `--min-agent-turns` filter fired. Those scenarios are hard failures in the composite rate and are
  skipped in the per-signal rates, so they depress the headline number while shrinking every
  per-signal denominator. Re-run them with `--resume`; see [Resuming a Run](resume.md).
- `Overall Success Rate: N/A (no scenarios had any applicable signal)` — nothing was scoreable at all.

**`all_metrics.json`** is the JSON array of every scenario's `metrics.json`, convenient for scripted
aggregation. **`all_latencies.csv`** has the header
`Scenario,User_Transcript,Agent_Transcript,Latency_ms` and one row per measured user-to-agent
response. **`evaluation_log.txt`** is the runner's own log (appended, not truncated, on resume).
**`run_args.json`** holds an `invocations` list — each entry records `started_at`, `argv`,
`parsed_args`, and the resolved scenario list, with the judge API key redacted.

## `metrics.json`

The per-scenario source of truth. Fields always present:

| Field | Meaning |
|---|---|
| `scenario_name`, `scenario_directory` | Identity and on-disk location |
| `total_turns`, `turns` | Turn count and the list of `timestamp` / `role` / `text` entries |
| `duration_seconds`, `scenario_duration` | Bridge-measured and runner-measured wall clock |
| `latency_stats` | `count`, `mean_ms`, `p50_ms`, `p95_ms`, `min_ms`, `max_ms` |
| `latencies` | Per-measurement `user_transcript` / `agent_transcript` / `latency_ms` |
| `stop_reason` | `[EXIT]` (agent ended the call) or `[TIMEOUT]` |
| `clean_exit` | `true` iff `stop_reason` is `[EXIT]` |
| `is_action_match` | Action-list signal; `"N/A"` when the scenario ships no reference answer |
| `token_usage` | Per side: `n_calls`, `prompt`, `completion` |
| `success_breakdown` | Signal names bucketed into `passed` / `failed` / `not_applicable` / `excluded` |
| `is_successful` | Strict conjunction over the scenario's whitelisted signals |
| `is_task_successful` | Same conjunction ignoring `clean_exit` |

Conditional fields appear only when the corresponding signal applies:
`db_state_match` plus `db_state_expected_hash` / `db_state_actual_hash`;
`db_state_assertion_pass_rate` plus per-predicate `db_state_assertion_verdicts`;
`nl_assertion_pass_rate`; `judge_score` and `judge_passed`; and
`insufficient_agent_turns` (only when the stall filter fired). A `trace_metrics` block is folded in
when a `trace_metrics.json` file exists in the scenario directory or in `bot_logs_agent/`.

The names inside `success_breakdown` are the `SuccessSignal` string values —
`is_action_match`, `db_state_match`, `db_state_assertion`, `nl_assertion`, `judge_passed`,
`clean_exit`. `excluded` lists signals that were computed but are not in the scenario's whitelist, so
they are informational only. See [Scoring](../understand-scoring/scoring.md) for how each signal is derived.

## `judge_result.json`

Written whenever the LLM judge ran. `--judge-url` and `--judge-model` both carry defaults, so the
judge is constructed on every run unless you blank one of them. Always carries `score`, `reason`,
and `judge_input` (the verbatim `system_prompt` and `user_content` the judge saw — use it to triage
a surprising score without re-deriving the prompt). Scenarios with natural-language assertions additionally get
`nl_assertion_verdicts` (one entry per assertion) and `nl_assertion_pass_rate`.

## Action and DB artifacts

`final_agent_response.json` is always list-wrapped as `[{"actions": [...]}]`. Each action record
carries the upstream tool `name`, its `arguments`, a `result`, and a `side` tag stamped by the
bridge according to which bot produced it (`"agent"` or `"user"`). For
single-side domains everything is tagged `"agent"`; for the dual-side telecom domain, user-sim tool
calls arrive tagged `"user"` — see [tau2-telecom](../domain-guides/tau2-telecom.md).

`final_scenario_db_hash.txt` records one or two lines, `db_hash: <sha256>` and, when the scenario has
user-side state, `user_db_hash: <sha256>`. The full database never crosses the WebSocket, so a
`db_state_match` failure is diagnosed by comparing this file against `expected_db_hash` /
`expected_user_db_hash` in `scenario_config/metadata.json`. When the hashes differ, inspect
`bot_logs_agent/llm_context.json` to find the tool call that diverged from the reference.

## Transcript and audio

`conversation_log.txt` opens with the scenario start time, then one block per turn:

```text
[ 21.624s -  45.425s] (23.801s) AGENT: Here are the pizza options we have: ...
  → Response latency: 7931.2ms
```

It ends with the end time and `Stop reason`. `conversation_log.seglst.json` is the same content in
segLST form (`session_id`, `words`, `speaker`, `start_time`, `end_time`) for diarization and ASR
tooling. `conversation_log.wav` is stereo at `--output-sample-rate`: left channel is the audio sent
to the agent, right channel is the audio sent to the user-sim.

## LLM context and bridge log

`bot_logs_agent/llm_context.json` and `bot_logs_user/llm_context.json` are lists of
`role` / `content` messages including tool calls and tool results — the ground truth for what each
bot actually did, as opposed to what it said. Compare the user-sim's intended utterance against the
agent's recognized transcript to separate ASR errors from reasoning errors. For dual-side domains,
read both files: the user-sim's phone-tool calls exist only on the user side.

`bridge_log.txt` covers audio routing, RTVI events, scenario initialization, cross-side sync
dispatches, and the end-of-scenario summary pull. Start here when a scenario produced no
`metrics.json` at all — the bridge log records the connection or initialization failure.

## Triage recipes

Use these commands to identify failed or stalled scenarios and inspect their latency and database-state
evidence.

```bash
cd eval_results/eval_20260730_075304

# Which scenarios failed, and on which signal?
jq -r 'select(.is_successful != true)
       | "\(.scenario_name)\tfailed=\(.success_breakdown.failed | join(","))"' all_metrics.json

# Scenarios the stall filter counted as failures.
jq -r 'select(.insufficient_agent_turns == true) | .scenario_name' all_metrics.json

# P95 latency per scenario.
jq -r '"\(.scenario_name)\t\(.latency_stats.p95_ms)"' all_metrics.json

# Expected vs actual DB hash for one scenario.
jq -r .expected_db_hash tau2_retail__1/scenario_config/metadata.json
cat tau2_retail__1/final_scenario_db_hash.txt
```

## Next Steps

Continue with the scoring definitions or operational guides that match what you found in the artifacts.

- [Scoring](../understand-scoring/scoring.md) — how each of the six signals is computed and combined.
- [Resuming a Run](resume.md) — re-running stalled or in-flight scenarios into the same session directory.
- [Metrics Reference](../../reference/evaluation/metrics.md) — field-by-field reference.
- [Eval CLI Reference](../../reference/evaluation/eval-cli.md) — every `run_evaluation.py` flag.
