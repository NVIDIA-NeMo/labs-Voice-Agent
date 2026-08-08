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

# tau2_telecom

Telecom tech support, ported from tau2-bench. It is the first **dual-side** domain in the NeMo Labs
Voice Agent eval harness: the simulated user owns a mock phone (`TelecomUserDB`) with its own 30
LLM-callable tools, while the agent owns the carrier back office (`TelecomDB`) with 13 tools. Neither
side can see the other's database, so the bridge reconciles them after every write.

| | |
|---|---|
| Scenario count | 114 (`tau2_telecom__*`) plus a parallel 114 (`tau2_telecom_workflow__*`) over the same tasks |
| Registry domain | `tau2_telecom` for both registrations |
| Scenario base | `Tau2TelecomBaseScenario` / `Tau2TelecomWorkflowBaseScenario` in `nemo_voice_agent/evaluation/scenarios/data/tau2_telecom/base.py` |
| Agent DB | `nemo_voice_agent/evaluation/data/tau2_telecom/db.json` (plans, lines, customers, bills, devices) |
| User DB | `nemo_voice_agent/evaluation/data/tau2_telecom/user_db.json` (device state + `surroundings`) |
| Agent policy | `main_policy.md` + `tech_support_manual.md` or `tech_support_workflow.md`, concatenated |
| Gating signals | `db_state_assertion`, `clean_exit` (plus `nl_assertion` when a task declares any) |

Every one of the 114 tasks ships `db_state_assertions` and `initialization_actions`; none currently
ship `nl_assertions`.

## Run it

Three terminals. `SERVER_CONFIG_PATH` resolves against the current working directory, so `cd`
first.

```bash
# Terminal 1 — simulated user bot (owns the phone; shipped user.yaml has enable_tool_calling: true)
cd evaluation && WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml python bot_server.py

# Terminal 2 — agent under test
cd evaluation && WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py

# Terminal 3 — runner
cd evaluation && python run_evaluation.py --domain tau2_telecom
```

`--domain` filters on the `<domain>__` prefix, so `--domain tau2_telecom` selects only the manual
variant and `--domain tau2_telecom_workflow` only the workflow variant. Single scenarios go through
`--scenarios <name>`; see [Eval CLI reference](../../reference/eval-cli.md). If the user bot is
started with tool calling disabled, the simulated user cannot operate its phone and most scenarios
stall.

## Two databases, two tool surfaces

`Tau2TelecomBaseScenario.has_user_state = True` makes gold replay seed a user-side DB alongside the
agent-side one and tells the bridge to pull a scenario summary from **both** WebSockets at the end of
the run. Each bot still follows the one-DB-per-bot convention: the agent bot's `shared_state["db"]`
is the `TelecomDB`, the user bot's is the `TelecomUserDB`.

Agent-side tools (13, in `nemo_voice_agent/evaluation/tools/tau2_telecom_tools.py`):

| Group | Tools |
|---|---|
| Lookup | `get_customer_by_phone`, `get_customer_by_id`, `get_customer_by_name`, `get_details_by_id` |
| Billing | `get_bills_for_customer`, `send_payment_request` |
| Line control | `suspend_line`, `resume_line`, `enable_roaming`, `disable_roaming` |
| Data | `get_data_usage`, `refuel_data` |
| Escalation | `transfer_to_human_agents` |

The agent also gets the harness-level `EndConversationTool`, which resolves through the registry's
`default` namespace fallback.

User-side tools (30, in `nemo_voice_agent/evaluation/tools/tau2_telecom_user_tools.py`):

| Group | Tools |
|---|---|
| Toggles (6) | `toggle_airplane_mode`, `toggle_data`, `toggle_roaming`, `toggle_wifi`, `toggle_wifi_calling`, `toggle_data_saver_mode` |
| Inspection (15) | `check_status_bar`, `check_network_status`, `check_sim_status`, `check_wifi_status`, `check_wifi_calling_status`, `check_data_restriction_status`, `check_network_mode_preference`, `check_apn_settings`, `check_vpn_status`, `check_installed_apps`, `check_app_status`, `check_app_permissions`, `check_payment_request`, `can_send_mms`, `run_speed_test` |
| Repair and settings (9) | `reboot_device`, `reseat_sim_card`, `reset_apn_settings`, `set_apn_settings`, `set_network_mode_preference`, `grant_app_permission`, `connect_vpn`, `disconnect_vpn`, `make_payment` |

Telecom tools declare a class-level snake_case `name`, which is simultaneously the registry key, the
LLM-visible function name, the `name` field of the recorded action, and the gold-replay dispatch key.
The two surfaces have disjoint names, so `_build_tool_map` merges them into one map without
collisions.

## Policy variants

Upstream registers telecom twice, once per tech-support policy prose style. This repo mirrors that
with two scenario classes per task that differ **only** in `policy_variant`:

| Class | `policy_variant` | Concatenated policy | Scenario name prefix |
|---|---|---|---|
| `Tau2TelecomBaseScenario` | `manual` | `main_policy.md` + `tech_support_manual.md` (long-form prose) | `tau2_telecom__` |
| `Tau2TelecomWorkflowBaseScenario` | `workflow` | `main_policy.md` + `tech_support_workflow.md` (procedural steps) | `tau2_telecom_workflow__` |

Both resolve the same `tau2_id`, DBs, `reference_answer`, `db_state_assertions`,
`initialization_actions`, and tool surface, so running both is a clean A/B on policy prose. `domain`
stays `tau2_telecom` on the workflow class, keeping tool lookup, data paths, predicate registration,
and sync dispatch identical.

## Prompt addenda

Telecom appends three blocks to the agent prompt after the parent's voice-realization notes, and two
to the user-sim's `user_actions`. All five are module constants in the telecom `base.py`.

| Constant | Side | Purpose |
|---|---|---|
| `TELECOM_AGENT_TOOL_AVAILABILITY_NOTE` | agent | Enumerates the 13 callable tools and the 30 phone tools by name, so the model stops hallucinating user-side calls that the policy text mentions |
| `TELECOM_AGENT_STAY_ON_TASK_GUIDELINE` | agent | Keeps the agent on the user's stated symptom instead of pivoting to an incidentally-discovered overdue bill or suspended sibling line |
| `TELECOM_AGENT_HOME_NETWORK_NOTE` | agent | States the home network (US) and makes "where are you physically located?" the first diagnostic question for connectivity complaints, then check roaming |
| `TELECOM_USER_INSTRUCTIONS` | user | Four-step script: describe the symptom, follow one instruction at a time, report each tool result verbally, confirm and close |
| `TELECOM_PASSIVE_TOOL_USE_GUIDELINE` | user | Passive tool use only — never fire a phone tool unless the agent just asked for that specific action, and never narrate as the agent |

## Scoring

Telecom overrides `success_signals` to `DB_STATE_ASSERTION` plus `CLEAN_EXIT` (adding `NL_ASSERTION`
when a task declares assertions). `DB_STATE_MATCH` and `ACTION_MATCH` are still computed and written
to `metrics.json`, but land in `success_breakdown.excluded` rather than gating the verdict — telecom
has an open solution space where several valid action sequences produce different whole-DB states
while satisfying the same outcome predicates.

Assertion records are translated at load time from upstream's `env_assertions`: the `env_type` field
becomes `side`, and the value `"assistant"` becomes `"agent"`. The runner dispatches each record to
the agent DB or user DB based on that `side` field, then evaluates the predicate. A scenario's
`db_state_assertion_pass_rate` must be 1.0 for the signal to count as passed.

Registered predicates (`nemo_voice_agent/evaluation/tools/tau2_telecom_predicates.py`, all under
domain `tau2_telecom`):

| Predicate | Arguments beyond `db` |
|---|---|
| `assert_mobile_data_status` | `expected_status` (bool) |
| `assert_internet_speed` | `expected_speed` (Mbps floor), optional `expected_desc` bucket label |
| `assert_service_status` | `expected_status` (network status string) |
| `assert_can_send_mms` | `expected_status` (bool) |
| `assert_data_refueling_amount` | `customer_id`, `line_id`, `expected_amount` |
| `assert_no_overdue_bill` | `overdue_bill_id` |

Because the runner needs a real DB to evaluate predicates against, the bridge sets `include_db: true`
on `get_scenario_summary` whenever a scenario has `db_state_assertions`, and dual-pulls so both DBs
come back. See [Scoring signals](../scoring.md) and [Metrics](../../reference/metrics.md).

## Scenario initialization

Each task's `initialization_actions` are dispatched by the `apply_initialization` RTVI client message
before the conversation starts, after `shared_state_init` is merged and `db_path` is resolved to a
loaded DB. Telecom registers 20 initialization functions in
`nemo_voice_agent/evaluation/tools/tau2_telecom_init_functions.py` — `turn_airplane_mode_on`,
`unseat_sim_card`, `break_apn_settings`, `set_data_usage`, `suspend_line_for_overdue_bill`, and so
on. Each mutates the DB in place; the bridge filters records by `side` first, so each bot only
applies the mutations meant for its own DB.

## Cross-side state sync

Upstream tau2 runs both DBs inside one `Environment` and calls `sync_tools()` to reconcile them. In
voice mode the DBs live in two separate bot processes, so the bridge maintains in-process **shadow
DBs** and pushes deltas. The pipeline turns on only when a scenario overrides `Scenario.sync_state`;
telecom's override delegates to the pure function `sync_telecom_state(agent_db, user_db)` in
`nemo_voice_agent/evaluation/tools/tau2_telecom_sync.py`, which mutates both dicts in place and
returns a per-side delta mapping dotted paths to values.

Propagation paths:

| Trigger | Effect | Direction |
|---|---|---|
| `line.status` | `surroundings.line_active` | agent to user |
| `line.roaming_enabled` | `surroundings.roaming_allowed` | agent to user |
| `line.data_used_gb`, `line.data_refueling_gb`, `plan.data_limit_gb` | `surroundings.mobile_data_usage_exceeded` | agent to user |
| Any bill with status `AWAITING_PAYMENT` | `surroundings.payment_request` | agent to user |
| `surroundings.payment_request.paid` becomes true | that bill's `status` becomes paid, request cleared | user to agent |

Two invocation points mirror upstream's call sites:

1. **Post-initialization.** After `apply_initialization` succeeds, the bridge loads shadow DBs,
   replays the init actions onto them, runs `sync_state` once, and dispatches the resulting deltas so
   both bots start from coherent cross-side state.
2. **Per action.** Every `WriteScenarioTool._record_action` pushes an `action-applied` RTVI server
   message. The bridge replays that action onto the shadow DBs using the scenario's `_build_tool_map`
   (each tool exposes a synchronous `invoke`), runs `sync_state`, and sends each non-empty side delta
   as an `apply_sync_delta` RTVI client message.

Bot-side, `apply_sync_delta` dispatches through the per-domain applier registry in
`nemo_voice_agent/evaluation/sync_appliers.py`. Telecom registers `apply_telecom_sync_delta`, which
handles plain dotted paths, list-by-id paths such as `bills[B1002].status` that the generic applier
cannot parse, and re-derives network connection state when any `surroundings` field changed. See
[RTVI messages](../../reference/rtvi-messages.md).

Any new dual-side domain that overrides `sync_state` must also provide `_build_tool_map(state)`
returning tools with a synchronous `invoke`, since that is what the bridge uses for shadow replay.
Single-side domains keep the inherited no-op and skip the pipeline entirely.

## Related

- [tau2_airline](tau2-airline.md) and [tau2_retail](tau2-retail.md) — single-side tau2 domains
- [Authoring domains](../authoring-domains.md) — adding your own
- [Data provenance](../data-provenance.md) — upstream pin and license for the telecom fixtures
