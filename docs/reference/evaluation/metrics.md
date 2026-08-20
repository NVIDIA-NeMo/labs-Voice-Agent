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

# Metrics Dictionary

Field-by-field reference for the numbers the NeMo Labs Voice Agent evaluation harness writes. All of
it is produced by `nemo_voice_agent/evaluation/runner.py` (per-scenario scoring and run aggregation),
`nemo_voice_agent/evaluation/bridge.py` (turns, latency, token counters), and
`nemo_voice_agent/evaluation/utils.py` (the LLM judge). For what the numbers mean conceptually, see
[Scoring](../../evaluate/understand-scoring/scoring.md); for where each file lives, see [Reading Results](../../evaluate/run-evaluations/results.md).

## Value conventions

| Convention | Meaning |
|---|---|
| Key absent | The signal did not apply to this scenario at all (for example, no `expected_scenario_db`, or no judge configured). |
| `"N/A"` (string) | The signal applies in principle but had no evidence. `is_action_match` is `"N/A"` when the scenario ships no reference answer; `is_successful` is `"N/A"` when no whitelisted signal was applicable. |
| Pass rate as a gate | A `*_pass_rate` float only counts as a *pass* for the composite when it is exactly `1.0` — every assertion must hold. |
| Rate is `None` | A run-level rate is omitted from `all_summary.txt` entirely when its bucket is empty. |

## `metrics.json` — always present

One file per scenario directory. These keys are written for every scenario that completed a bridge run.

| Field | Type | Meaning |
|---|---|---|
| `scenario_name` | string | Scenario class `name`. The substring before the first `__` is the domain bucket used for per-domain rollups. |
| `scenario_directory` | string | Path to this scenario's output directory. Refreshed on resume if the directory moved. |
| `total_turns` | int | Number of entries in `turns`. |
| `turns` | list | Each entry is `timestamp` (ISO 8601), `role` (`"user"` or `"agent"`), `text` (finalized transcript for that speaker segment). |
| `duration_seconds` | float | Bridge-measured elapsed time between its own start and end timestamps. |
| `scenario_duration` | float | Runner-measured wall clock around `bridge.run_scenario`. Slightly larger than `duration_seconds`. |
| `latency_stats` | object | Aggregate over this scenario's `latencies` — see below. |
| `latencies` | list | One entry per measurement: `user_transcript`, `agent_transcript`, `latency_ms`. |
| `stop_reason` | string | `"[EXIT]"` when the agent called `EndConversationTool`, `"[TIMEOUT]"` otherwise. |
| `clean_exit` | bool | `true` iff `stop_reason` is `"[EXIT]"`. Backs the `clean_exit` success signal. |
| `is_action_match` | bool or `"N/A"` | Deterministic reference-vs-prediction action-list comparison. `false` when the prediction file is missing; `"N/A"` when the reference file is missing. |
| `token_usage` | object | `agent` and `user` sub-objects, each with `n_calls`, `prompt`, `completion`. Accumulated from RTVI `metrics` events during the run. |
| `success_breakdown` | object | Signal names bucketed into `passed` / `failed` / `not_applicable` / `excluded`. |
| `is_successful` | bool or `"N/A"` | Composite verdict: strict AND over the applicable signals in the scenario's `success_signals` whitelist. |
| `is_task_successful` | bool | Same conjunction with `clean_exit` removed from the failure list. Written whenever `is_successful` is a bool. |

`latency_stats` fields: `count` (int), `mean_ms`, `p50_ms`, `p95_ms`, `min_ms`, `max_ms` (floats; all
zero when `count` is 0). One latency is recorded per user-to-agent handoff: the wall-clock gap from the
user bot's last audio frame to the agent's first audio frame after it, in milliseconds. It therefore
includes ASR, LLM, and TTS time-to-first-byte, not just LLM latency.

## `metrics.json` — conditional

| Field | Type | Written when |
|---|---|---|
| `db_state_expected_hash` | string | The scenario exposes `expected_scenario_db`. SHA-256 of the runner's in-process gold replay. |
| `db_state_actual_hash` | string or `null` | Same condition. The hash the bot returned from `get_scenario_summary`; `null` if it returned none. |
| `db_state_match` | bool | Same condition. `true` iff the two hashes are equal. A missing bot hash is a `false`, not `"N/A"`. |
| `db_state_assertion_verdicts` | list | The scenario exposes `db_state_assertions` (tau2-telecom today). One verdict per predicate. |
| `db_state_assertion_pass_rate` | float | Same condition. Passing predicates divided by total predicates. |
| `nl_assertion_pass_rate` | float | The scenario carries `nl_assertions` **and** a judge ran. Passing verdicts divided by total verdicts. |
| `judge_score` | float | A judge was configured (`--judge-url`). Raw score in the range 0 to 1. |
| `judge_passed` | bool | A judge was configured **and** `--judge-threshold` was set. `judge_score >= judge_threshold`. |
| `insufficient_agent_turns` | bool | `--min-agent-turns` (default 3) fired: the agent produced fewer LLM responses than the floor. Always `true` when present. |
| `trace_metrics` | object | A `trace_metrics.json` exists in the scenario directory or in `bot_logs_agent/`. Copied verbatim; the runner never interprets it. |

Each entry of `db_state_assertion_verdicts` has: `func_name` (registered predicate name), `side`
(`"agent"` or `"user"` — selects which pulled DB the predicate reads), `passed` (bool), `expected`
(the assertion's `assert_value`), `actual` (what the predicate returned, `null` on error), `message`
(optional upstream label), `error` (`null`, or the failure reason when the predicate is missing,
raises, or the required DB was not pulled — `passed` is then `false`).

### `success_breakdown` buckets

Names inside each bucket are the `SuccessSignal` string values: `is_action_match`, `db_state_match`,
`db_state_assertion`, `nl_assertion`, `judge_passed`, `clean_exit`.

| Bucket | Contents |
|---|---|
| `passed` | Whitelisted signals that evaluated to `true`. |
| `failed` | Whitelisted signals that evaluated to `false`. Non-empty means `is_successful` is `false`. |
| `not_applicable` | Whitelisted signals with no verdict for this scenario. If this covers the whole whitelist, `is_successful` is `"N/A"`. |
| `excluded` | Signals computed but **not** in the whitelist. Informational only; they never affect `is_successful`. |

`is_task_successful` is derived from this object: `true` iff `failed` contains nothing other than
`clean_exit`.

### Stalled scenarios

When `--min-agent-turns N` is set (default 3), the runner counts the agent's LLM calls — preferring
`token_usage.agent.n_calls`, falling back to the message count in `bot_logs_agent/llm_context.json`.
Below the floor, the scenario is forced to `is_successful: false` and `is_task_successful: false`,
gets `insufficient_agent_turns: true`, and is **skipped** in every per-signal bucket. It is a counted
failure in the composite rate, not an exclusion. Token usage still rolls up so cost accounting stays
correct. See [Resuming a Run](../../evaluate/run-evaluations/resume.md) to retry them.

## `judge_result.json`

Written once per scenario, only when a judge is configured.

| Field | Type | Meaning |
|---|---|---|
| `score` | float | 0 to 1. Copied into `metrics.json` as `judge_score`. `0.0` on any request or parse error. |
| `reason` | string | The judge's explanation, with quoted evidence. On error, `"Error: ..."`. |
| `judge_input` | object | `system_prompt` and `user_content` — the verbatim text the judge saw, for triage without re-deriving the prompt. |
| `nl_assertion_verdicts` | list | Present only when the scenario carries `nl_assertions`. One entry per assertion. |
| `nl_assertion_pass_count` | int | Same condition. Number of verdicts with `passed: true`. |
| `nl_assertion_total` | int | Same condition. Equals the number of assertions declared by the scenario. |
| `nl_assertion_pass_rate` | float | Same condition. `nl_assertion_pass_count / nl_assertion_total`. |

Each `nl_assertion_verdicts` entry has `index` (1-based, matching the numbered assertion the judge was
shown), `assertion` (the assertion text, inlined so the file is self-describing), `passed` (bool), and
`reason` (string). The list is normalized to exactly one entry per assertion: missing, out-of-range,
or non-boolean verdicts are filled as `passed: false` with an explanatory reason.

## `all_summary.txt` run-level aggregates

Rendered as text only — these rates are not serialized to JSON. Names in the first column are the
runner's internal variable names; the second column is the label as it appears in the file. A rate is
omitted entirely when its bucket is empty.

| Runner name | Label in `all_summary.txt` | Denominator |
|---|---|---|
| `success_rate` | `Overall Success Rate` | Scenarios with a boolean `is_successful` (the `"N/A"` ones are excluded). |
| `task_success_rate` | `Task Success Rate (excl. clean_exit)` | Same set of scenarios. |
| `action_match_rate` | `Action-list match` | Scenarios where `is_action_match` was a bool. |
| `db_state_success_rate` | `DB-State match` | Scenarios that declared `expected_scenario_db`. |
| `db_state_assertion_success_rate` | `DB-State-Assertion pass` | **Predicates** emitted in the run, not scenarios. |
| `nl_assertion_success_rate` | `NL-Assertion pass` | **Assertions** emitted in the run, not scenarios. |
| `judge_score_mean` | `Judge score mean` | Mean of `judge_score` over scenarios that were judged. |
| `judge_pass_rate` | `Judge passed (>= threshold)` | Scenarios with a `judge_passed` bool. |
| `clean_exit_rate` | `Clean exit` | Scenarios that completed a bridge run. |

The two assertion rates are denominated in assertions rather than scenarios so that they stay
comparable across domains where scenarios carry different assertion counts.

Also in the file:

- **`Overall Latency Statistics`** — `overall_latency_stats`, pooled over every latency measurement in
  the run: `count`, `mean_ms`, `p50_ms`, `p95_ms`, `min_ms`, `max_ms`. Each is `-1` when nothing was measured.
- **`Token Usage`** — per side, `n_calls`, `prompt`, `completion`, and their sum, plus a `Run total`
  across both sides. The whole block is suppressed when no token events were received.
- **Stall warning** — `WARNING: N scenario(s) had fewer than N agent turn(s) ... and were counted as
  failures.` followed by the scenario names and the suggested resume command.

### Per-domain breakdowns

Printed only when the run spans more than one domain bucket, keyed by the scenario-name prefix before
`__`. Available for: success rate, task success rate (excluding clean exit), DB-state match rate,
NL-assertion pass rate, and DB-state-assertion pass rate. Judge score, judge pass, action match, and
clean exit have no per-domain block.

## Aggregation on resume

Scenarios skipped by `--resume` do not re-run; their `metrics.json` is loaded from disk and folded into
the same buckets via `RunAggregator.add_scenario`. Each signal is appended only when its key is present
and well-typed, so an older `metrics.json` missing a field simply contributes nothing to that bucket.
NL-assertion verdicts are re-read from `judge_result.json` on this path, because only the pass *rate*
lives in `metrics.json`. The resulting `all_summary.txt` is identical whether a scenario ran live or
was loaded from a prior session.

## Related

- [Scoring](../../evaluate/understand-scoring/scoring.md) — how each of the six signals is derived and combined.
- [Reading Results](../../evaluate/run-evaluations/results.md) — the full artifact layout and triage recipes.
- [Eval CLI Reference](eval-cli.md) — the flags that switch these fields on.
- [tau2-telecom](../../evaluate/domain-guides/tau2-telecom.md) — the domain that exercises `db_state_assertions` and dual-side DB pulls.
