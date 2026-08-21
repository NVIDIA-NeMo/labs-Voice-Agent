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

# Scoring Model

The NeMo Labs Voice Agent evaluation runner emits **six orthogonal scoring signals** per scenario and combines
them into a single composite verdict, `is_successful`. The canonical list is the `SuccessSignal` enum in
`nemo_voice_agent/evaluation/scenarios/classes.py`; the composite is computed by
`Scenario.compute_is_successful` and written by `nemo_voice_agent/evaluation/runner.py`.

Every signal is opt-in per scenario through the `success_signals` whitelist. Signals that are computed but not
whitelisted are still saved — they land in `success_breakdown.excluded` as diagnostics.

## Signal matrix

The following matrix maps each supported signal to its persisted field, value type, and current domain use.

| Enum member | Value in `success_breakdown` | `metrics.json` field read | Type | Path-dependent | Opted in by |
|---|---|---|---|---|---|
| `ACTION_MATCH` | `is_action_match` | `is_action_match` | `bool` or `"N/A"` | Yes — compares the action list | `restaurant`, `customer_service`, `fastbite` (computed for any scenario with a `reference_answer`) |
| `DB_STATE_MATCH` | `db_state_match` | `db_state_match` | `bool` (absent when not applicable) | No — end state only | `eva_airline`, `tau2_airline`, `tau2_retail` (computed for any scenario with `expected_scenario_db`) |
| `DB_STATE_ASSERTION` | `db_state_assertion` | `db_state_assertion_pass_rate` | float in 0–1 | No | `tau2_telecom` (per-task) |
| `NL_ASSERTION` | `nl_assertion` | `nl_assertion_pass_rate` | float in 0–1 | No — judged on context/transcript | `tau2_retail`, `tau2_telecom` (per-task) |
| `JUDGE_PASSED` | `judge_passed` | `judge_passed` | `bool` | — | `qa` and other free-form domains |
| `CLEAN_EXIT` | `clean_exit` | `clean_exit` | `bool` | No | **Every domain** |

Note the two float signals: the enum value written to `success_breakdown` is `db_state_assertion` /
`nl_assertion`, while the numeric field in `metrics.json` carries the `_pass_rate` suffix.

## Signal details

**`ACTION_MATCH`** — `check_if_task_success` (in `nemo_voice_agent/evaluation/utils.py`) recursively compares
`scenario_config/reference_answer.json` against `final_agent_response.json`. Every dict in the reference must
find a match in the prediction (order-independent); extra prediction keys and extra list items are tolerated
unless the scenario sets `disallow_extra_items` or the run passes `--strict-match`. String comparison honors the
scenario's `ignore_capitalization`, `ignore_punctuation`, and `clean_text` flags. `"N/A"` when the scenario has
no reference answer; `False` when the reference exists but the agent produced no prediction file.

**`DB_STATE_MATCH`** — the bot computes `get_dict_hash(shared_state["db"])` inside the `get_scenario_summary`
RTVI handler and returns only the SHA-256 string; the runner hashes `scenario.expected_scenario_db` from its own
in-process gold replay and compares. The DB itself never crosses the WebSocket. Path-independent: any action
sequence landing on the correct end state passes. If the scenario declares an expected DB but the bot returned
no hash, the result is `False`, not `"N/A"`. Both hashes are also written to `final_scenario_db_hash.txt` for
triage. Only the agent-side hash is scored — the user-side `user_db_hash` pulled for dual-side domains is
recorded but does not gate this signal.

**`DB_STATE_ASSERTION`** — per-predicate scoring. Each entry in `scenario.db_state_assertions` is a
`func_name` / `arguments` / `assert_value` / `side` record; the runner dispatches it through
`evaluate_db_state_assertion` in `nemo_voice_agent/evaluation/db_state_predicates.py`, which picks the agent DB
or the user DB based on `side` and calls a pure predicate registered under `(domain, func_name)`. The pass rate
is passes over total predicates. Per-predicate verdicts land in `metrics.json` under
`db_state_assertion_verdicts`. Missing or raising predicates produce `passed=False` with an `error` field rather
than aborting the run. Used where the solution space is open and several valid end states satisfy the same
outcome — see [tau2_telecom](../domain-guides/tau2-telecom.md).

**`NL_ASSERTION`** — natural-language claims about the conversation, judged per assertion by the LLM judge.
Only populated when the scenario declares `nl_assertions` **and** the judge ran. Per-assertion verdicts live in
`judge_result.json` under `nl_assertion_verdicts`; the scenario-level rate is `nl_assertion_pass_rate` in
`metrics.json`.

**`JUDGE_PASSED`** — `judge_score >= --judge-threshold` (threshold default `0.9`). The raw float is saved
separately as `judge_score`. The judge receives the reference and prediction payloads when they exist, both
bots' `llm_context.json` histories, the numbered NL assertions, and — only with
`--judge-include-conversation` — the bridge transcript turns. Both `--judge-url` and `--judge-model` carry
defaults pointing at a local OpenAI-compatible endpoint, so the judge is constructed on every run; override them
to target your own judge. See [Eval CLI Reference](../../reference/evaluation/eval-cli.md).

**`CLEAN_EXIT`** — `True` only when `bridge.stop_reason` is `[EXIT]`, meaning the agent voluntarily called
`EndConversationTool`. `[TIMEOUT]` always fails. The raw reason is saved as `stop_reason`.

## Composite `is_successful`

`Scenario.compute_is_successful` takes the dict of all six verdicts and returns:

- the strict **AND** over the whitelist entries whose verdict is not `None`;
- the literal string `"N/A"` when no whitelisted signal was applicable (for example, a `qa` run with no
  reachable judge).

Two runner-level overrides sit on top of this:

- **Stalled scenarios.** With `--min-agent-turns` (default `3`), a scenario whose agent produced fewer
  completed turns is forced to `is_successful = False` and counted as a failure in the composite rate, while
  being skipped in the per-signal rates — the individual measurements are meaningless for a conversation that
  never happened. It is *not* excluded from the denominator.
- **`is_task_successful`.** The same conjunction with `clean_exit` removed from the failed set, so you can read
  "did the agent do the work" separately from "did the agent hang up properly". Reported as
  `Task Success Rate (excl. clean_exit)` in `all_summary.txt`.

## Per-domain whitelists

Whitelists are declared on each domain's base scenario — a `ClassVar` tuple when fixed, or a `cached_property`
when it depends on a per-task opt-in such as `nl_assertions`.

| Domain | Scenarios | Gating signals | Why |
|---|---|---|---|
| `eva_airline` | 50 | `DB_STATE_MATCH`, `CLEAN_EXIT` | Gold expected DB ships per scenario; path-independent |
| `tau2_airline` | 50 | `DB_STATE_MATCH`, `CLEAN_EXIT` | Expected DB derived from the upstream task; no judge dependency |
| `tau2_retail` | 114 | `DB_STATE_MATCH`, `CLEAN_EXIT`, plus `NL_ASSERTION` when the task declares assertions | 40 of 114 tasks carry NL claims |
| `tau2_telecom` | 114 | `DB_STATE_ASSERTION`, `CLEAN_EXIT`, plus `NL_ASSERTION` when declared | Open solution space — whole-DB hash and action match are informational only |
| `tau2_telecom_workflow` | 114 | Same as `tau2_telecom` | Parallel registration over the same tasks with the workflow policy file |
| `restaurant`, `customer_service` | 11, 10 | `ACTION_MATCH`, `CLEAN_EXIT` | Single canonical structured `reference_answer` |
| `qa` | 10 | `JUDGE_PASSED`, `CLEAN_EXIT` | Free-form text answers; no deterministic comparator applies |

Design rule: prefer `DB_STATE_MATCH` over `ACTION_MATCH` wherever a domain ships an expected DB, because it is
path-independent. Use `JUDGE_PASSED` as a gate only when no deterministic alternative exists. `NL_ASSERTION`
runs through the judge but contributes per-claim verdicts, so it is safe to gate on.

## Why `CLEAN_EXIT` is universal

Closure discipline gates every domain. An agent that performs the right work but never stops talking is not a
successful agent, and a timed-out scenario burns several times the compute of a clean one. The gate matters most
for **policy-refusal scenarios**, where the expected end state equals the initial state: without `CLEAN_EXIT`, an
agent that crashed at the greeting would pass `DB_STATE_MATCH` by doing nothing at all. The regression test
`test_every_concrete_scenario_includes_clean_exit` in `tests/unit/test_runner_is_successful.py` fails the build
if a new domain omits it.

`Scenario.__init_subclass__` also raises `TypeError` at class-definition time if a concrete scenario (one that
declares `name`) resolves `success_signals` to an empty sequence, so an unscored domain cannot be authored by
accident. See [Authoring Domains](../create-evaluations/authoring-domains.md).

## Strict thresholds for float signals

The two pass-rate signals are normalized to booleans with a threshold of exactly `1.0`. A 95% pass rate means one
assertion failed, and that is a defect to investigate rather than noise to round away. Values of `None` or
`"N/A"` normalize to "not applicable" and drop out of the conjunction entirely.

## `success_breakdown`

Every scenario's `metrics.json` carries a `success_breakdown` object with four buckets of enum-value strings:

| Bucket | Contents |
|---|---|
| `passed` | Whitelisted signals that returned `True` |
| `failed` | Whitelisted signals that returned `False` — these dragged the verdict down |
| `not_applicable` | Whitelisted signals with no verdict for this scenario |
| `excluded` | Signals computed and saved but **not** in this scenario's whitelist (diagnostics) |

The `excluded` bucket is how you spot "all gating signals passed, but the agent took an unusual path" — for
telecom, `db_state_match` and `is_action_match` land there on every scenario.

## Where the numbers land

Use the session and scenario artifacts according to the level of detail you need.

| Artifact | Contents |
|---|---|
| `<scenario>/metrics.json` | All six signals, `is_successful`, `is_task_successful`, `success_breakdown`, `stop_reason`, per-predicate `db_state_assertion_verdicts` |
| `<scenario>/judge_result.json` | Judge score, reason, `nl_assertion_verdicts`, and the verbatim judge input |
| `<scenario>/final_scenario_db_hash.txt` | Post-run `db_hash` (and `user_db_hash` for dual-side domains) |
| `<scenario>/scenario_config/metadata.json` | The scenario's `success_signals`, `expected_db_hash`, `db_state_assertions`, `nl_assertions`, `initialization_actions` — so an old run stays interpretable without reloading the scenario class |
| `all_summary.txt` | Composite headline rate, task success rate, and per-signal pass rates with their own denominators (scenarios, predicates, assertions) |

For the field-by-field schema, see [Metrics Reference](../../reference/evaluation/metrics.md); for the directory layout, see
[Reading Results](../run-evaluations/results.md).
