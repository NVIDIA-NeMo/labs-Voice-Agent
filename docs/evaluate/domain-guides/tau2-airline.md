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

# tau2_airline

Fifty airline customer-support scenarios — cancel, rebook, upgrade, add baggage, compensate — ported from
[tau2-bench](https://github.com/sierra-research/tau2-bench) and run as full voice conversations. The agent
receives tau2's `policy.md` unchanged; scoring is a path-independent hash of the end-state database.

## At a glance

The following table summarizes the domain's registry, fixtures, tool surface, and state model.

| Property | Value |
| --- | --- |
| Scenario names | `tau2_airline__0` … `tau2_airline__49` (50, contiguous) |
| Registry namespace (`Scenario.domain`) | `tau2_airline` |
| Base class | `Tau2AirlineBaseScenario` in `nemo_voice_agent/evaluation/scenarios/data/tau2_airline/base.py` |
| Concrete scenarios | `group_0x.py` … `group_4x.py` in the same package (auto-scaffolded) |
| Tools | `nemo_voice_agent/evaluation/tools/tau2_airline_tools.py` — 14 ported tools |
| Fixtures | `nemo_voice_agent/evaluation/data/tau2_airline/` |
| Split | `base` (30 train + 20 test = 50 tasks) |
| Sides | Single-side — `has_user_state = False`, no user-side tools, no cross-side sync |
| Scored signals | `DB_STATE_MATCH`, `CLEAN_EXIT` |
| `max_duration` | 900 s, inherited from `Tau2BaseScenario` |
| Upstream | tag `voice-user-sim-v1.0` (commit `17e07b1`), MIT — see [Data provenance](data-provenance.md) |

## Run it

Start the two bots and the bridge as described in the [evaluation quickstart](../run-evaluations/quickstart.md), then:

```bash
cd evaluation
python run_evaluation.py --domain tau2_airline
```

`--domain` filters on the `tau2_airline__` name prefix. To run a subset, name scenarios explicitly:

```bash
cd evaluation
python run_evaluation.py --scenarios tau2_airline__11 tau2_airline__17
```

Two flags matter for a 50-scenario job. `--duration` is unset by default, so each scenario gets its own
900-second ceiling; pass an integer to cap it globally. `--min-agent-turns` defaults to `3` — scenarios that
end with fewer completed agent turns (a stalled vLLM server, a bot that never spoke) are counted as failures
in the composite rate and skipped in the per-signal rates rather than dropped. Full flag list:
[Evaluation CLI](../../reference/evaluation/eval-cli.md).

## One attribute per scenario

Every concrete class sets `name` and `tau2_id` and nothing else:

```python
@register_eval_scenario
class Tau2Airline11(Tau2AirlineBaseScenario):
    name = "tau2_airline__11"
    tau2_id = "11"
```

`Tau2BaseScenario` derives the rest from the fixtures as cached properties:

| Property | Source |
| --- | --- |
| `tau2_task` | `tasks.json` entry, joined with the voice-eligible id list in `tasks_voice.json` and intersected with `split_tasks.json["base"]` |
| `persona_name` | `tasks_voice.json` control preset — a metric-slicing label only; it never enters a prompt |
| `policy` | `policy.md`, read once per process |
| `db` | `db/` fixture, loaded by `load_db_artifact` |
| `expected_scenario_db` | Post-replay agent DB (see below) |
| `reference_answer` | Recorded actions from the same replay |
| `user_persona` / `user_task` / `user_resources` | `tasks.json` field `user_scenario.instructions` |

There is no `current_date` attribute and no `tool_map` attribute here; the tool map is built on demand by
`_build_tool_map(state)`, which instantiates every entry of `TAU2_AIRLINE_TOOL_NAME_TO_CLASS` bound to the
given state dict.

## The agent prompt

`get_agent_prompt()` returns `policy.md` **verbatim**, then appends one `## Additional Notes to Follow`
section containing four constants, in this order:

| Constant | Module | Purpose |
| --- | --- | --- |
| `GENERAL_PROMPT` | `nemo_voice_agent.utils.voice_prompts` | Spoken-output style: concise, plain prose, no Markdown |
| `VOICE_ALPHANUMERIC_RULE` | `nemo_voice_agent.utils.voice_prompts` | Speak IDs letter by letter, spelled form only |
| `END_CONVERSATION_GUIDELINE` | `nemo_voice_agent.evaluation.scenarios` | When to call `EndConversationTool` |
| `EXECUTION_HONESTY_GUIDELINE` | `nemo_voice_agent.evaluation.scenarios` | Report only steps that a tool actually performed |

Sierra's published voice numbers assume the policy reaches the agent unchanged, so nothing is spliced into
the policy body. The `agent_persona` / `agent_task` / `agent_actions` stubs on the base class exist purely so
that code iterating `Scenario` subclasses does not hit `NotImplementedError` — they do **not** participate in
prompt assembly. `agent_resources` is the one agent-side property the base overrides for real, because the
bot server reads it to register tools.

## The simulated user

The user side is assembled from the structured `user_scenario` block through the inherited
`get_user_prompt()`:

- `task_instructions` becomes the persona's `personality`; `reason_for_call` becomes the task goal.
- `known_info` and `unknown_info` render as `Things you know` / `Things you don't know` info sections. Naming
  what the caller does *not* know is what stops the simulator from inventing plausible reservation IDs.
- `user_persona.name` is deliberately `None`. Identity comes from `known_info` (for example, a user id like
  `daiki_muller_1116`); injecting the tau2 `persona_name` would contradict it.
- The only user-side guideline is `VOICE_ALPHANUMERIC_RULE`. The user simulator gets no tools in this domain.

## Tools

Fourteen tools are ported from tau2's `AirlineTools`, registered under the `tau2_airline` namespace. The
scenario also requests `EndConversationTool`, which resolves through the registry's `default` namespace, for
15 registered tools in total. Only write tools call `_record_action`.

| Tool class | Action name | Records |
| --- | --- | --- |
| `GetUserDetailsTool` | `get_user_details` | no |
| `GetReservationDetailsTool` | `get_reservation_details` | no |
| `ListAllAirportsTool` | `list_all_airports` | no |
| `SearchDirectFlightTool` | `search_direct_flight` | no |
| `SearchOnestopFlightTool` | `search_onestop_flight` | no |
| `GetFlightStatusTool` | `get_flight_status` | no |
| `CalculateTool` | `calculate` | no |
| `BookReservationTool` | `book_reservation` | yes |
| `CancelReservationTool` | `cancel_reservation` | yes |
| `SendCertificateTool` | `send_certificate` | yes |
| `UpdateReservationBaggagesTool` | `update_reservation_baggages` | yes |
| `UpdateReservationFlightsTool` | `update_reservation_flights` | yes |
| `UpdateReservationPassengersTool` | `update_reservation_passengers` | yes |
| `TransferToHumanAgentsTool` | `transfer_to_human_agents` | yes |

Class names are the registry keys; action names match tau2's method names and are what appear in recorded
records. `TAU2_AIRLINE_TOOL_NAME_TO_CLASS` maps between the two. `TransferToHumanAgentsTool` mutates nothing
but still records, and emits the `<exit>` marker after its result is delivered so the bridge can tear the
session down cleanly.

Every tool subclasses `_Tau2ReadTool` or `_Tau2WriteTool`, both of which mix in `_Tau2InvokeMixin`. That
mixin gives each tool a sync `invoke(**kwargs)` for gold replay and an async `_execute(**kwargs)` for live
pipecat calls, both routing into a single `_do_work(p)`. New tools implement only `_do_work`, `properties`,
`required_properties`, and `DESCRIPTION` — see [Authoring tools](../create-evaluations/authoring-tools.md).

## Gold replay, expected DB, and reference_answer

`Tau2BaseScenario._gold_replay` deep-copies the seeded DB, instantiates the full tool map against it, and
dispatches each entry of `evaluation_criteria.actions` through `invoke()`. One pass yields two ground truths:
`expected_scenario_db` (the final DB) and `reference_answer`.

`reference_answer` is wrapped as `{"actions": [...]}` so it matches the eva shape and one comparator path
serves both domains. One record, abridged to a single leg:

```json
{
  "actions": [
    {
      "action_type": "update_reservation_flights",
      "name": "update_reservation_flights",
      "arguments": {
        "reservation_id": "GV1N64",
        "cabin": "basic_economy",
        "flights": [{"flight_number": "HAT003", "date": "2024-05-19"}],
        "payment_id": "gift_card_1642017"
      },
      "result": {"reservation_id": "GV1N64", "charged": -5244},
      "side": "agent"
    }
  ]
}
```

Because read tools record nothing, only the 27 tasks whose gold list contains at least one write produce a
non-empty action list — 50 records across the domain. The other 23, including refusal tasks where the policy
forbids the requested change, correctly produce `{"actions": []}`: the agent passes by making no mutation.

## Scoring

`success_signals = (DB_STATE_MATCH, CLEAN_EXIT)`. The bot hashes its own `shared_state["db"]` inside the
`get_scenario_summary` handler and returns only the SHA-256 string; the runner hashes
`expected_scenario_db` from its in-process replay and compares. Any tool sequence that lands on the correct
end state passes.

`ACTION_MATCH` is still computed, because the scenario has a `reference_answer` — but it is not whitelisted,
so it lands in `success_breakdown.excluded` as a diagnostic rather than gating the verdict. This domain sets
no `nl_assertions`, no `db_state_assertions`, and no `initialization_actions`. See the
[scoring model](../understand-scoring/scoring.md) for how the composite is built and [Reading results](../run-evaluations/results.md) for where
each field is written.

## Database seeding and key casing

`setup_shared_state` writes `state["db_path"] = "tau2_airline/db.json"` for the agent side only; the bot
resolves it against `EVAL_DATA_ROOT` in its `apply_initialization` handler. The path is sent instead of the
DB itself because the airline database exceeds pipecat's WebSocket frame limit. On disk it is sharded as
`db/flights.json`, `db/users.json`, and `db/reservations.json` — `load_db_artifact` probes `<name>.json`
first, then the `<name>/` directory, and reassembles an identical in-memory dict either way, so hashes are
unaffected by the layout.

ASR after letter-by-letter spelling returns inconsistent case, so the lookup helpers normalize:

| Key | Casing in the fixture | Helper |
| --- | --- | --- |
| `reservation_id` | Uppercase (`XEHM4B`) | `_get_reservation_dict` uppercases |
| `flight_number` | Uppercase (`HAT001`) | `_get_flight_dict` uppercases |
| `user_id` | Lowercase (`daiki_muller_1116`) | `_get_user_dict` lowercases |

Replay determinism relies on three more upstream behaviors reproduced verbatim: a frozen clock of
`2024-05-15T15:00:00` for `created_at`, new reservation ids allocated from `HATHAT` / `HATHAU` / `HATHAV` in
order, and certificate payment ids from a fixed triple.

## Related

[tau2_retail](tau2-retail.md) adds NL assertions on the same machinery, [tau2_telecom](tau2-telecom.md) is
the dual-side variant with cross-side state sync, and [eva_airline](eva-airline.md) is the other airline
domain. To add scenarios of your own, see [Authoring scenarios](../create-evaluations/authoring-scenarios.md).
