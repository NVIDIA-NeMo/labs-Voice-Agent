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

# tau2_retail

114 online-retail customer-support scenarios ported from tau2-bench. The simulated user calls about an
order — cancel it, change its address, exchange or return items, switch the payment method — and the agent
works the upstream retail `policy.md` against a shared products/users/orders database.

Retail is single-side: only the agent holds a database. What distinguishes it from
[tau2_airline](tau2-airline.md) is `nl_assertions` — natural-language claims about what the agent said,
which 40 of the 114 tasks carry and which the LLM judge scores one at a time.

## At a glance

| Property | Value |
| --- | --- |
| Scenario names | `tau2_retail__0` … `tau2_retail__113` (zero-based upstream task id) |
| Count | 114 (the `base` split: 74 train + 40 test) |
| Registry domain | `tau2_retail` |
| Gating signals | `DB_STATE_MATCH`, `CLEAN_EXIT`, plus `NL_ASSERTION` when the task declares assertions |
| Agent tools | 16 ported retail tools plus `EndConversationTool` |
| User tools | none (single-side domain) |
| `max_duration` | 900 s, inherited from `Tau2BaseScenario` |
| Fixtures | `nemo_voice_agent/evaluation/data/tau2_retail/` |
| Code | `nemo_voice_agent/evaluation/scenarios/data/tau2_retail/`, `nemo_voice_agent/evaluation/tools/tau2_retail_tools.py` |

## Run it

Start both bot servers first — see the [Evaluation quickstart](../quickstart.md). Then, from `evaluation/`:

```bash
cd evaluation

# The whole domain (114 scenarios; budget accordingly)
python run_evaluation.py --domain tau2_retail

# A handful of tasks, including one that carries nl_assertions
python run_evaluation.py --scenarios tau2_retail__0 tau2_retail__24
```

`--domain` filters by the `tau2_retail__` name prefix, so nothing else can be swept in.

## Scenario shape

Each of the 114 classes in `scenarios/data/tau2_retail/group_Nx.py` declares only two attributes:

```python
@register_eval_scenario
class Tau2Retail0(Tau2RetailBaseScenario):
    name = "tau2_retail__0"
    tau2_id = "0"
```

`Tau2RetailBaseScenario` derives everything else from `tau2_id`, mostly through the shared
`Tau2BaseScenario` machinery in `scenarios/data/tau2_common.py`:

| Derived property | Source |
| --- | --- |
| `tau2_task` | `tasks.json` entry, joined with `tasks_voice.json` and intersected with `split_tasks.json["base"]` |
| `persona_name` | `tasks_voice.json` control preset — a metric-slicing label only, never used in a prompt |
| `policy` | `policy.md`, read once per process |
| `db` | `db.json` (~2.8 MB — `products`, `users`, `orders`), deep-copied per scenario |
| `expected_scenario_db` | Gold replay of `evaluation_criteria.actions` against a fresh copy of `db` |
| `reference_answer` | The action records recorded by that same replay, wrapped as an `actions` list |
| `nl_assertions` | `evaluation_criteria.nl_assertions`, or `None` |

The gold replay runs once, in-process, and yields both ground-truth signals — so the expected DB and the
reference action list can never disagree. `initialization_actions` and `db_state_assertions` are `None` for
every retail task; those are telecom mechanics.

## Prompts

The agent prompt is the upstream `policy.md` **verbatim**, followed by an `## Additional Notes to Follow`
section carrying `GENERAL_PROMPT` and `VOICE_ALPHANUMERIC_RULE` from `nemo_voice_agent/utils/voice_prompts.py`
plus the end-conversation and execution-honesty guidelines. Splicing the policy body would break comparability
with published tau2 numbers, so the `agent_persona` / `agent_task` / `agent_resources` stubs on the base class
exist for interface introspection and do not participate in prompt assembly.

The simulated user is built the normal way from `tasks.json["user_scenario"]["instructions"]`:
`reason_for_call` becomes the goal, `task_instructions` the personality, and `known_info` / `unknown_info`
render as **Things you know** / **Things you don't know** sections. Exposing `unknown_info` matters here —
without it the user simulator invents order ids rather than admitting it does not have them, which hands the
agent identifiers that do not exist and short-circuits the intended discovery path.

## Tool surface

All 16 tools live in `nemo_voice_agent/evaluation/tools/tau2_retail_tools.py`, registered under the
`tau2_retail` namespace. `EndConversationTool` resolves through the registry's `default`-namespace fallback.

| Tool class | Kind | Action recorded |
| --- | --- | --- |
| `FindUserIdByNameZipTool` | read | — |
| `FindUserIdByEmailTool` | read | — |
| `GetUserDetailsTool` | read | — |
| `GetOrderDetailsTool` | read | — |
| `GetProductDetailsTool` | read | — |
| `GetItemDetailsTool` | read | — |
| `ListAllProductTypesTool` | read | — |
| `CalculateTool` | read | — |
| `CancelPendingOrderTool` | write | `cancel_pending_order` |
| `ExchangeDeliveredOrderItemsTool` | write | `exchange_delivered_order_items` |
| `ModifyPendingOrderAddressTool` | write | `modify_pending_order_address` |
| `ModifyPendingOrderItemsTool` | write | `modify_pending_order_items` |
| `ModifyPendingOrderPaymentTool` | write | `modify_pending_order_payment` |
| `ModifyUserAddressTool` | write | `modify_user_address` |
| `ReturnDeliveredOrderItemsTool` | write | `return_delivered_order_items` |
| `TransferToHumanAgentsTool` | write | `transfer_to_human_agents` |

Only write tools call `_record_action`, and the recorded `name` is the upstream snake_case method name — class
names serve the registry, action names serve action-list scoring. `TransferToHumanAgentsTool` mutates nothing
but still records, and emits `<exit>` from `_after_result` so the tool result is committed before the bridge
tears the session down. `TAU2_RETAIL_TOOL_NAME_TO_CLASS` maps the action names back to classes, and
`_build_tool_map` uses it to dispatch gold-replay actions.

## `nl_assertions` and per-assertion judging

40 of the 114 tasks carry a list of claims in `evaluation_criteria.nl_assertions`, for example:

> Agent should tell the user the materials of the two t-shirts are polyester and cotton.

These are outcome checks that no database diff can express — the fact never lands in the DB, it only has to be
said out loud. The mechanics:

1. `Tau2RetailBaseScenario.nl_assertions` reads the upstream list and normalizes an empty or missing list to
   `None`, so the runner's truthiness guard skips verdict aggregation for action-only tasks.
2. `success_signals` is a `cached_property`, not a fixed tuple: it adds `NL_ASSERTION` only when
   `self.nl_assertions` is non-empty. The whitelist therefore cannot drift from the per-task opt-in.
3. The runner passes the assertions to `LLMJudge.judge_scenario`, which numbers them into an
   `nl_assertions` section of the judge input and asks for one verdict per number.
4. The judge's reply is normalized to exactly one entry per assertion, in order, each carrying `index`,
   `assertion`, `passed`, and `reason`. Missing entries, out-of-range indices, non-boolean `passed`, and judge
   transport errors all become `passed=false` with explanatory text rather than a crash.
5. Verdicts land in `judge_result.json` under `nl_assertion_verdicts`; the scenario-level
   `nl_assertion_pass_rate` lands in `metrics.json`; the run-level roll-up prints as `NL-Assertion pass` in
   `all_summary.txt`, denominated in assertions rather than scenarios.

The pass-rate threshold is `1.0` — every assertion in a task must pass for the signal to gate green. The
judge's overall `judge_score` is deliberately **not** in the retail whitelist: `DB_STATE_MATCH` is
deterministic and `NL_ASSERTION` is already per-claim, so the aggregate judge opinion stays informational.
`ACTION_MATCH` is likewise computed and saved but lands in `success_breakdown.excluded`.

Two upstream tasks are worth knowing about: task `24` has assertions but no reference actions (the expected DB
equals the seeded DB, so the agent passes by answering correctly and changing nothing), and task `57` is
chitchat with neither actions nor assertions, scored on `DB_STATE_MATCH` and `CLEAN_EXIT` alone.

## ID normalization on lookup

Voice ASR returns identifiers that were spelled out letter by letter, so casing and punctuation arrive
inconsistently. The module-level helpers in `tau2_retail_tools.py` normalize on the way into the DB dict:

| Identifier | Shape in `db.json` | Normalization applied |
| --- | --- | --- |
| Order id | Uppercase with a leading `#`, e.g. `#W0000000` | `_get_order_dict` uppercases and prepends `#` when the speaker dropped it |
| User id | Lowercase, e.g. `ivan_hernandez_6923` | `_get_user_dict` lowercases |
| Product id, item id | Case-stable numeric strings | none needed |
| Email, first/last name | free text | compared case-insensitively inside the finder tools |

Zip codes are compared exactly. Normalization happens only at lookup — the keys written back to the DB stay in
their upstream form, which keeps the DB hash stable regardless of how the caller pronounced the id.

## Related pages

- [Scoring model](../scoring.md) — how the six signals combine into `is_successful`.
- [Benchmarks & domains](../benchmarks.md) — the full catalogue and naming conventions.
- [tau2_airline](tau2-airline.md) — same base machinery, no NL assertions.
- [tau2_telecom](tau2-telecom.md) — dual-side domain that adds `db_state_assertions`.
- [Data provenance](../data-provenance.md) — upstream pin and license for the retail fixtures.
- [Metrics reference](../../reference/metrics.md) — field-by-field schema for `metrics.json` and
  `judge_result.json`.
