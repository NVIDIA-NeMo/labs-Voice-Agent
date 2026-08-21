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

# eva_airline

`eva_airline` is a 50-scenario airline customer-service benchmark ported into the NeMo Labs Voice Agent
evaluation harness from [ServiceNow/eva](https://github.com/ServiceNow/eva) (v0.1.3, MIT). Each scenario
puts a simulated passenger on a phone call with the agent under test: change a flight, recover from an
airline cancellation, cancel and refund a booking, or resist a fabricated disruption claim.

Scoring is deterministic. Every scenario ships a gold post-run database from upstream, so the run is scored
by hashing the agent's final database rather than by an LLM judge.

## Run the domain

Three terminals. `SERVER_CONFIG_PATH` resolves against the current working directory, so `cd evaluation`
first in every one of them.

```bash
# Terminal 1 — simulated user bot
cd evaluation && WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml python bot_server.py

# Terminal 2 — agent under test
cd evaluation && WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py

# Terminal 3 — bridge + runner (all 50 scenarios)
cd evaluation && python run_evaluation.py --domain eva_airline
```

`--domain` filters registered scenarios by the `eva_airline__` name prefix. To run one scenario, pass its
registered name instead:

```bash
cd evaluation && python run_evaluation.py --scenarios eva_airline__voluntary_date_change
cd evaluation && python run_evaluation.py --list        # every registered scenario name
```

See [Evaluation quickstart](../run-evaluations/quickstart.md) and the [eval CLI reference](../../reference/evaluation/eval-cli.md)
for the full flag surface.

## Scenario layout

Scenario classes live in the package `nemo_voice_agent/evaluation/scenarios/data/eva_airline/`. `base.py`
holds `EvaAirlineBaseScenario` plus five hand-authored seed scenarios; the `group_Nx.py` shards hold
scenarios auto-scaffolded from the dataset by `scripts/prepare_eva_data/generate_airline_scaffolds.py`.
The package `__init__.py` imports every shard so the `@register_eval_scenario` decorators fire.

| Module | Scenarios | eva IDs | Representative content |
| --- | --- | --- | --- |
| `base.py` | 5 | 1.1.2, 2.1.1, 3.1.3, 5.1.1, 7.2.1 | Hand-authored seeds, one per major flow |
| `group_1x.py` | 8 | 1.1.3 – 1.3.2 | Voluntary date and segment changes on round trips |
| `group_2x.py` | 9 | 2.1.2 – 2.4.2 | IRROPS: airline cancellations and delays, rebooking plus vouchers |
| `group_3x.py` | 2 | 3.1.5, 3.3.4 | Missed-flight and itinerary-recovery cases |
| `group_4x.py` | 7 | 4.1.1 – 4.2.5 | Same-day changes and same-day fee waivers |
| `group_5x.py` | 7 | 5.1.2 – 5.2.6 | Cancellations, cash refunds, travel credits |
| `group_6x.py` | 4 | 6.1.1 – 6.3.4 | Rebooking with hotel vouchers, alternate airports, transfer requests |
| `group_7x.py` | 8 | 7.1.1 – 7.4.1 | Edge cases: missing auth details, fabricated disruptions, fee-waiver pressure |

Registered names are `eva_airline__voluntary_date_change`, `eva_airline__irrops_cancellation`,
`eva_airline__missed_flight_standby`, `eva_airline__cancellation_refund`,
`eva_airline__escalation_edge_case` for the seeds, and `eva_airline__<eva_id with dots as underscores>`
for the scaffolded ones (for example `eva_airline__3_1_5`).

## eva_id drives everything

A subclass declares `name`, `eva_id`, `description`, `user_persona`, `user_task`, and `user_actions` —
nothing else. `EvaAirlineBaseScenario` derives the rest lazily through `cached_property`.

| Member | Kind | Derived from |
| --- | --- | --- |
| `_scenario_db` | `cached_property` | `<data root>/eva_airline/<eva_id>.json`, read on first access |
| `current_date` | `cached_property` | the `_current_date` key of that JSON — single source of truth for "today" |
| `expected_scenario_db` | `cached_property` | `ground_truth.expected_scenario_db` in `eva_airline_dataset.jsonl` for the matching id |
| `domain` | class attribute | fixed to `"eva_airline"` — the tool-registry namespace |
| `success_signals` | class attribute | `(SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT)` |
| `max_duration` | class attribute | `900` seconds — voice round-trips run roughly ten times slower than text |
| `policy` | `cached_property` | the complete `instructions` field from the pinned upstream `airline_agent.yaml` |
| `agent_persona` / `agent_task` | properties | the upstream agent role and description, exposed for scenario-contract introspection |
| `agent_actions` / `agent_resources` | properties | an empty action stub and the fixed eva tool surface, respectively |

The dataset index is loaded once per process by `_load_eva_airline_dataset_index()`, cached with
`functools.cache`. An `eva_id` with no dataset entry raises `KeyError` when `expected_scenario_db` is first
touched.

`setup_shared_state(state, side)` seeds the **agent** side only: it assigns the whole scenario database
inline to `state["db"]`. eva fixtures are small enough (roughly 10–30 KB each) to travel inside the
`shared_state_init` argument of the `apply_initialization` RTVI client message, so unlike the tau2 domains
there is no `db_path` indirection. There is no user-side database.

## Agent policy

The live agent prompt starts with the `role` and complete `instructions` from ServiceNow/eva 0.1.3's
`configs/agents/airline_agent.yaml`. A pinned copy lives at
`nemo_voice_agent/evaluation/data/eva_airline/airline_agent.yaml`; only upstream trailing whitespace is
normalized. This keeps authentication, fees, rebooking, refunds, compensation, standby, elite-status, and
escalation rules in one auditable upstream-derived source instead of reconstructing a shorter policy in
`base.py`.

`get_agent_prompt()` preserves that policy content and then appends a clearly marked
`## Additional Notes to Follow` section containing only NeMo voice/runtime guidance:

- the scenario's current date;
- the general voice-agent prompt and spoken alphanumeric rule;
- the rule not to read internal journey IDs aloud;
- clean conversation termination and execution-honesty guidance.

The YAML's tool declarations are retained for provenance, but the callable surface is still defined by
`agent_resources` and the NeMo eva tool implementations described below.

## Fixture layout

Fixtures ship inside the installed library at `nemo_voice_agent/evaluation/data/eva_airline/`, resolved by
`get_eval_data_root()`. Set `EVAL_DATA_ROOT` to point at a different tree.

| Path | Contents |
| --- | --- |
| `eva_airline/<eva_id>.json` (50 files) | Per-scenario database: `_current_date`, `reservations`, `journeys`, `disruptions`, `travel_credits`, `meal_vouchers`, `refunds` |
| `eva_airline/eva_airline_dataset.jsonl` (50 lines) | Per-scenario metadata keyed by `id`: `user_goal`, `user_config`, `expected_flow`, `scenario_context`, `ground_truth` |
| `eva_airline/airline_agent.yaml` | Pinned upstream agent configuration; `role` and `instructions` form the policy portion of the live agent prompt |

Provenance and license notes are recorded in the data directory's `README.md`; see also
[Data provenance](data-provenance.md).

## Tool surface

Every eva_airline scenario exposes the same fixed 15-tool eva surface plus the harness-generic
`EndConversationTool`. Implementations live in `nemo_voice_agent/evaluation/tools/eva_airline_tools.py`,
registered under the `eva_airline` domain namespace.

| Kind | Tools |
| --- | --- |
| Read (4) | `GetReservationTool`, `GetFlightStatusTool`, `GetDisruptionInfoTool`, `SearchRebookingOptionsTool` |
| Write (10) | `RebookFlightTool`, `CancelReservationTool`, `ProcessRefundTool`, `AssignSeatTool`, `AddBaggageAllowanceTool`, `AddMealRequestTool`, `AddToStandbyTool`, `IssueTravelCreditTool`, `IssueHotelVoucherTool`, `IssueMealVoucherTool` |
| System (1) | `TransferToAgentTool` |
| Harness (1) | `EndConversationTool` |

Write tools subclass `WriteAirlineTool`, which binds `ACTION_TYPES` to `AIRLINE_ACTION_TYPES`
(`rebook_flight`, `cancel_reservation`, `process_refund`, `issue_meal_voucher`, `issue_hotel_voucher`,
`issue_travel_credit`, `assign_seat`, `add_baggage_allowance`, `add_meal_request`, `add_to_standby`,
`transfer_to_agent`) and appends a record to `shared_state["actions"]` on success. Read tools record
nothing. To add or change a tool, see [Authoring tools](../create-evaluations/authoring-tools.md).

### Ancillaries carried across a rebook

`RebookFlightTool` copies `bags_checked` and `meal_request` from the booking it replaces onto the new
segments; `seat` is deliberately not carried, because each aircraft has its own seat map and the gold
replay expects an explicit `AssignSeatTool` call. The rule is to carry what the dataset gives no
availability model for, and re-select what it does.

This is a **deliberate divergence from upstream**. ServiceNow's `eva` hard-codes both fields to
`0` / `None` on rebook, yet its own `expected_scenario_db` keeps the original checked-bag count for all
25 rebooking scenarios in the packaged dataset. With the upstream behaviour, the gold state for 17 of
the 50 scenarios was reachable only by additionally calling `AddBaggageAllowanceTool` with the original
count — a write that nothing in those scenarios asks for. Both fields remain defaults rather than locks:
`AddBaggageAllowanceTool` and `AddMealRequestTool` still override afterwards. Carrying `meal_request` is
a no-op against the packaged fixtures, which leave it unset throughout.

## Scoring

The domain whitelists two of the six scoring signals:

| Signal | How it is produced |
| --- | --- |
| `db_state_match` | The bot hashes its own `shared_state["db"]` and returns the SHA-256 string in the `get_scenario_summary` response, alongside the recorded action list. The runner hashes `scenario.expected_scenario_db` from its in-process gold replay and compares strings. The database itself never crosses the WebSocket. |
| `clean_exit` | The agent called `EndConversationTool` and the conversation terminated normally. |

Hash matching is path-independent: any sequence of tool calls that lands on the gold end state passes.
Both sides import the same canonicalization module, `nemo_voice_agent/evaluation/db_hash.py`, so the two
hashes are comparable byte for byte.

No eva_airline scenario declares a `reference_answer`, so `is_action_match` does not participate — the
commented-out block in `VoluntaryDateChange` is kept only as a worked example of the action-list shape.
Scenarios that complete fewer than `--min-agent-turns` agent turns (default `3`) are counted as failures
in the composite rate and skipped in the per-signal rates. Details in [Scoring](../understand-scoring/scoring.md) and
[Metrics reference](../../reference/evaluation/metrics.md).

## Voice-readability rule

Confirmation codes, flight numbers, and airport codes are the main failure surface in a spoken airline
call. `VOICE_ALPHANUMERIC_RULE` is a module-level constant in `nemo_voice_agent/utils/voice_prompts.py`;
`EvaAirlineBaseScenario` imports it into both the agent guidelines and each scenario's user guidelines.

The rule requires spelling each character one at a time — letters as letters, digits as words, punctuation
pronounced literally (`_` as "underscore", `-` as "dash", `@` as "at", `.` as "dot", `#` as "hash",
`*` as "star").

The load-bearing clause: **speak ONLY the spelled-out form, never the canonical sequence alongside it in
the same utterance.**

- Correct: "E, P, X, Y, E, K"
- Wrong: "EPXYEK, spelled E, P, X, Y, E, K"

The `CODE (spelled out as ...)` notation that appears throughout scenario prose is instructional metadata
for the model, not a response template. One exception: proper names such as "Johnson" are real words, so
the model may say the name and then spell it.

The agent guidelines add one airline-specific companion rule — internal journey IDs such as
`FL_SK621_20260320` are never read aloud; flights are referred to by flight number and date.

## Extending the domain

Choose the extension path that matches whether you are adding scenario coverage, tools, or fixture data.

- Add a scenario: subclass `EvaAirlineBaseScenario`, set `name` / `eva_id` / `description` and the three
  user-side members, and decorate with `@register_eval_scenario`. See
  [Authoring scenarios](../create-evaluations/authoring-scenarios.md).
- Regenerate the scaffolded shards from the dataset with
  `scripts/prepare_eva_data/generate_airline_scaffolds.py`. Generated prose is marked "Review prose before
  shipping" in the class docstring and is meant to be edited by hand.
- Build a new domain from a different upstream corpus:
  [Authoring domains](../create-evaluations/authoring-domains.md). For the tau2-based domains, see
  [tau2_airline](tau2-airline.md), [tau2_retail](tau2-retail.md), and [tau2_telecom](tau2-telecom.md).
