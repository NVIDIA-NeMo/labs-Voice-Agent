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

`tau2_airline` provides 50 airline customer-support scenarios ported from
[tau2-bench](https://github.com/sierra-research/tau2-bench) as full voice conversations. Tasks include
cancellations, rebooking, upgrades, baggage changes, and compensation. The agent receives tau2's
`policy.md` unchanged. Scoring combines a path-independent hash of the end-state database with 84 curated
natural-language assertions judged one claim at a time.

## At a Glance

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
| Scored signals | `DB_STATE_MATCH`, `CLEAN_EXIT`, plus `NL_ASSERTION` on the 24 tasks with adopted assertions |
| `max_duration` | 900 s, inherited from `Tau2BaseScenario` |
| Upstream | Tag `voice-user-sim-v1.0` (commit `17e07b1`), MIT. Refer to [Data Provenance](data-provenance.md). |
| Local deviations | Fixtures are unmodified. Two scoring inputs are defined in code: one dropped gold action and a curated assertion subset. |

## Run It

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
900-second ceiling. Pass an integer to cap it globally. `--min-agent-turns` defaults to `3`. Scenarios with
fewer completed agent turns count as failures in the composite rate. The per-signal rates skip them instead
of dropping them. Refer to the
[Evaluation Command-Line Interface (CLI)](../../reference/evaluation/eval-cli.md) for all flags.

## One Attribute Per Scenario

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
| `tau2_task` | `tasks.json` entry, joined with the voice-eligible ID list in `tasks_voice.json` and intersected with `split_tasks.json["base"]`. Airline overrides this property to apply `GOLD_ACTION_DROPS`. |
| `persona_name` | `tasks_voice.json` control preset — a metric-slicing label only; it never enters a prompt |
| `policy` | `policy.md`, read one time per process |
| `db` | `db/` fixture, loaded by `load_db_artifact` |
| `expected_scenario_db` | Post-replay agent database (DB), described in the gold replay section |
| `reference_answer` | Recorded actions from the same replay |
| `user_persona` / `user_task` / `user_resources` | `tasks.json` field `user_scenario.instructions` |

There is no `current_date` or `tool_map` attribute. `_build_tool_map(state)` builds the tool map on demand by
instantiating each `TAU2_AIRLINE_TOOL_NAME_TO_CLASS` entry with the given state dictionary.

## The Agent Prompt

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
prompt assembly. `agent_resources` is the one agent-side property the base uses at runtime because the
bot server reads it to register tools.

## The Simulated User

The user side is assembled from the structured `user_scenario` block through the inherited
`get_user_prompt()`:

- `task_instructions` becomes the persona's `personality`. `reason_for_call` becomes the task goal.
- `known_info` and `unknown_info` render as `Things you know` / `Things you don't know` info sections. Naming
  what the caller does *not* know is what stops the simulator from inventing plausible reservation IDs.
- `user_persona.name` is deliberately `None`. Identity comes from `known_info` (for example, a user ID like
  `daiki_muller_1116`). Injecting the tau2 `persona_name` would contradict it.
- The only user-side guideline is `VOICE_ALPHANUMERIC_RULE`. The user simulator gets no tools in this domain.

## Tools

Fourteen tools are ported from tau2's `AirlineTools`, registered under the `tau2_airline` namespace. The
scenario also requests `EndConversationTool`, which resolves through the registry's `default` namespace, for
15 registered tools in total. Only write tools call `_record_action`.

| Tool Class | Action Name | Records |
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

Class names are the registry keys. Action names match tau2's method names and are what appear in recorded
records. `TAU2_AIRLINE_TOOL_NAME_TO_CLASS` maps between the two. `TransferToHumanAgentsTool` mutates nothing
but still records, and emits the `<exit>` marker after its result is delivered so the bridge can tear the
session down cleanly.

Every tool subclasses `_Tau2ReadTool` or `_Tau2WriteTool`, both of which mix in `_Tau2InvokeMixin`. That
mixin gives each tool a sync `invoke(**kwargs)` for gold replay and an async `_execute(**kwargs)` for live
Pipecat calls, both routing into a single `_do_work(p)`. New tools implement only `_do_work`, `properties`,
`required_properties`, and `DESCRIPTION`. Refer to [Authoring Tools](../create-evaluations/authoring-tools.md).

## Gold Replay, Expected DB, and reference_answer

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

Because read tools record nothing, only the 27 tasks whose gold list contains at least one recording tool
produce a non-empty action list — 49 records across the domain, after the correction described in
[Upstream Gold-Action Corrections](#upstream-gold-action-corrections). A recording tool is one of the six
writes or `transfer_to_human_agents`. The other 23 tasks, including refusal tasks where the policy forbids
the requested change, correctly produce `{"actions": []}`: the agent passes by making no mutation.

## Upstream Gold-Action Corrections

One upstream gold action is dropped before replay. `GOLD_ACTION_DROPS` in
`nemo_voice_agent/evaluation/scenarios/data/tau2_airline/base.py` maps a `tau2_id` to the `action_id` values
that the `tau2_task` override removes from `evaluation_criteria.actions`. Both `expected_scenario_db` and
`reference_answer` come from `_gold_replay`, which reads `tau2_task`, so one drop corrects the hash target
and the reference action list together.

The following table lists every entry.

| Task | Dropped Action | Reason |
| --- | --- | --- |
| `39` | `39_10` — `cancel_reservation` on `MSJ4OA` | The reservation is not cancellable under the upstream policy. |

Task 39 asks the agent to cancel every upcoming flight for Amelia Davis, and the gold list cancels three
reservations. Reservations `8C8K4E` and `LU15PA` are business cabin, so the Cancel Flight section of
`policy.md` permits them. Reservation `MSJ4OA` is economy, was created on May 5, 2024 against a scenario
clock of May 15, 2024 at 3:00 p.m., and books flights whose status is `available`. That rules out the
booked-within-24-hours gate, the airline-cancelled gate, and the business-cabin gate. Only the insurance gate
remains, and the policy covers insurance cancellations for health or weather reasons only. This caller wants
to give up a seat for someone else, so `MSJ4OA` is not cancellable and the gold action contradicts the policy
the agent is given. Tasks `43` and `44` apply the same rule correctly on comparable reservations, which
confirms task 39 is the outlier rather than the precedent.

Two consequences matter when you compare numbers:

- The result for task `39` is **intentionally not comparable** to tau2's published leaderboard, because its
  `expected_scenario_db` differs from upstream's. The other 49 tasks are unaffected.
- Emptying `GOLD_ACTION_DROPS` restores bit-exact upstream behavior.

Task 39's upstream `nl_assertions` are deliberately not adopted. Task 39 has write actions, so it falls
outside the curated set described in [The Adopted Set](#the-adopted-set). That is what keeps the upstream
string `Agent cancels reservation MSJ4OA.` from contradicting the correction and re-importing the defect as a
judged claim. `tests/unit/test_tau2_airline_scenarios.py::test_task_39_assertions_are_not_adopted` guards it.

The fixture stays out of it. `nemo_voice_agent/evaluation/data/tau2_airline/tasks.json` remains
byte-identical to upstream, so re-importing the data never clobbers the correction. Refer to
[Data Provenance](data-provenance.md).

## Scoring

`success_signals` is a `cached_property`, not a fixed tuple. Every task gates on `DB_STATE_MATCH` and
`CLEAN_EXIT`. The 24 tasks that carry adopted natural-language (NL) assertions also gate on `NL_ASSERTION`.

| Tasks | Gating Signals |
| --- | --- |
| 26 with no adopted assertion | `DB_STATE_MATCH`, `CLEAN_EXIT` |
| 24 with adopted assertions | `DB_STATE_MATCH`, `NL_ASSERTION`, `CLEAN_EXIT` |

For `DB_STATE_MATCH`, the bot hashes its own `shared_state["db"]` inside the `get_scenario_summary` handler
and returns only the SHA-256 string. The runner hashes `expected_scenario_db` from its in-process replay and
compares. Any tool sequence that lands on the correct end state passes.

`ACTION_MATCH` is still computed, because the scenario has a `reference_answer` — but it is not whitelisted,
so it lands in `success_breakdown.excluded` as a diagnostic rather than gating the verdict. The judge's
overall `judge_score` stays informational for the same reason. This domain sets no `db_state_assertions` and
no `initialization_actions`. Refer to the
[Scoring Model](../understand-scoring/scoring.md) for how the composite is built and
[Reading Results](../run-evaluations/results.md) for where
each field is written.

## NL Assertions and the Curated Subset

Airline scores NL assertions, but not the upstream list. `Tau2AirlineBaseScenario.nl_assertions` returns a
curated subset defined in code and keyed by `tau2_id`. This is the one structural difference from
[tau2_retail](tau2-retail.md), which reads `evaluation_criteria.nl_assertions` out of the fixture and adopts
whatever upstream ships.

### Why a Curated Subset

Two findings drive the decision.

First, a DB hash alone leaves a hole. Of the 50 airline tasks, 24 have no DB-mutating gold action, so their
gold DB equals the seeded DB. These are the refusal, lookup, and advice tasks, where the correct outcome is
spoken rather than written. The harness reads `communicate_info` nowhere. With `DB_STATE_MATCH` and
`CLEAN_EXIT` as the only gates, an agent that greets the caller and immediately ends the call scores 1.0 on
all 24. Adopted assertions close that hole.

Second, the upstream assertions cannot be adopted wholesale. `tasks.json` carries 123 assertions spread
across all 50 tasks, and every task declares `reward_basis` as `["DB", "COMMUNICATE"]`, which excludes NL
assertions from upstream's own scoring, so upstream never runs them. An audit of all 123 found four that
contradict the policy or the database, and roughly 57 more that are questionable: restatements of an action
the DB hash already checks, refusals pinned to one specific rationale, vague negatives, and compound claims
that bundle several checks into a single verdict. Task 39 is the clearest failure. Its upstream assertions
include `Agent cancels reservation MSJ4OA.`, which the
[gold-action correction](#upstream-gold-action-corrections) shows the policy forbids. Task 39 is outside the
adopted set, so that string is never judged.

Only assertions that survive the audit are adopted. The rest stay in the fixture and are ignored.

### The Adopted Set

`ADOPTED_NL_ASSERTIONS` in `nemo_voice_agent/evaluation/scenarios/data/tau2_airline/base.py` holds the
authoritative list: **84 assertions across 24 tasks**, keyed by `tau2_id`. Those 24 tasks are exactly the
tasks with no DB-mutating gold action, so the curated set covers the gap and nothing else. The other 26 tasks
have `nl_assertions is None` and keep `DB_STATE_MATCH` and `CLEAN_EXIT`. Read the constant for the current
strings — the counts here are pinned by `tests/unit/test_tau2_airline_scenarios.py`, and each entry carries a
comment explaining what its task probes.

Per-task counts range from two to nine. Every task pairs a required spoken act — a refusal said out loud, or
a fact the agent must surface — with the prohibitions that constrain it. That pairing is the point. A pure
prohibition such as "Agent does not cancel reservation X" is satisfied by an agent that says nothing and does
nothing, which is the exact failure this change exists to catch. Two examples show what curation buys:

- **Tasks `48` and `49` were de-duplicated.** Upstream gave both the byte-identical single assertion
  `Agent does not cancel 3RK2T9.`, even though they probe opposite deceptions against the same reservation.
  The adopted sets discriminate. Task 48 asserts the agent tells the user the reservation was booked more
  than 24 hours ago, against a caller who insists it was 10 hours. Task 49 asserts the agent tells the user
  the reservation includes no travel insurance, against a caller who claims illness coverage.
- **Task `27`'s upstream assertion was inverted rather than dropped.** Upstream asserts the agent confirms
  the user can receive compensation for Silver status. The policy makes Silver status necessary but not
  sufficient, and this caller refuses any change or cancellation, so a policy-correct agent refuses. The
  adopted set asserts the refusal.

### Authoring Rules for the Curated Set

The constant's own comment block states the rules. Follow them if you extend the set.

| Rule | Why |
| --- | --- |
| Assert the outcome, never the rationale. | Tasks `13`, `26`, `31`, `36`, `43`, `45`, and `47` have several simultaneously valid grounds for the same refusal. Pinning one fails an agent that correctly cites another. |
| One checkable claim per string. | `nl_assertion_pass_rate` is thresholded at exactly `1.0`, so every weak string adds false-failure risk. |
| Name the object. | "Agent should not make any changes" is satisfied by an agent that did the forbidden thing to a different reservation. |
| Prefer claims a do-nothing agent fails. | A spoken refusal or a stated fact beats a pure absence-of-mutation claim, which silence trivially satisfies. |
| Keep assertions voice-safe. | No raw payment IDs and no exact cents. Reservation IDs and flight numbers are fine, because the judge waives ID spelling. |

Lookup-level claims such as "Agent looks up the details of reservation 3RK2T9." are decidable here even
though upstream's transcript-only judge could not decide them. The judge input includes
`<agent_context_history>`, which carries the agent's tool calls with their arguments and results.

### How Assertions Are Scored

Adopted assertions run through the same mechanics as retail.

1. `nl_assertions` returns `None` rather than an empty list for a task with no adopted assertion, so the
   runner's truthiness guard skips verdict aggregation for that scenario.
2. `success_signals` adds `NL_ASSERTION` only when `self.nl_assertions` is non-empty, so the whitelist cannot
   drift from the per-task opt-in.
3. The runner passes the assertions to `LLMJudge.judge_scenario`, which numbers them into an `nl_assertions`
   section of the judge input and asks for one verdict per number.
4. Verdicts land in `judge_result.json` under `nl_assertion_verdicts`. The scenario-level
   `nl_assertion_pass_rate` lands in `metrics.json`. The run-level roll-up prints as `NL-Assertion pass` in
   `all_summary.txt`, denominated in assertions rather than scenarios.

The pass-rate threshold is `1.0`, so every adopted assertion in a task must pass for the signal to pass.
`NL_ASSERTION` depends on the judge, so run the domain with the judge enabled. The runner logs a warning
naming how many queued scenarios whitelist a judge-dependent signal when the judge is off.

## Database Seeding and Key Casing

`setup_shared_state` writes `state["db_path"] = "tau2_airline/db.json"` for the agent side only. The bot
resolves it against `EVAL_DATA_ROOT` in its `apply_initialization` handler. The path is sent instead of the
DB itself because the airline database exceeds Pipecat's WebSocket frame limit. On disk, it is sharded as
`db/flights.json`, `db/users.json`, and `db/reservations.json`. `load_db_artifact` probes `<name>.json` and
then the `<name>/` directory. Both layouts produce an identical in-memory dictionary, so hashes are unchanged.

Automatic speech recognition (ASR) after letter-by-letter spelling returns inconsistent case, so the
lookup helpers normalize:

| Key | Casing in the Fixture | Helper |
| --- | --- | --- |
| `reservation_id` | Uppercase (`XEHM4B`) | `_get_reservation_dict` uppercases |
| `flight_number` | Uppercase (`HAT001`) | `_get_flight_dict` uppercases |
| `user_id` | Lowercase (`daiki_muller_1116`) | `_get_user_dict` lowercases |

Replay determinism also relies on three upstream behaviors reproduced verbatim. The `created_at` clock is
frozen at `2024-05-15T15:00:00`. New reservation IDs are allocated from `HATHAT`, `HATHAU`, and `HATHAV` in
order, and certificate payment IDs come from a fixed triple.

## Related

[tau2_retail](tau2-retail.md) runs the same machinery but adopts its NL assertions straight from upstream.
[tau2_telecom](tau2-telecom.md) is the dual-side variant with cross-side state sync.
[eva_airline](eva-airline.md) is the other airline domain. To add scenarios, refer to
[Authoring Scenarios](../create-evaluations/authoring-scenarios.md).
