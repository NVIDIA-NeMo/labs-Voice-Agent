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

# Authoring Scenarios

A scenario is a Python class that fully specifies one evaluation run: what the simulated user wants,
what the agent under test is told, which tools each side gets, and how the result is scored. Scenario
classes live under `nemo_voice_agent/evaluation/scenarios/data/`.

Every domain has a **base class** that implements domain-level defaults and is *not* registered.
Concrete scenarios subclass the base, override only what differs, and register themselves with
`@register_eval_scenario`. In the tau2 and eva domains most subclasses are under 20 lines because
everything derives from a single `tau2_id` / `eva_id` class attribute.

## Where scenarios live

The following table shows the package shape used by each benchmark-derived domain.

| Domain | Location | Shape |
|---|---|---|
| `eva_airline` | `scenarios/data/eva_airline/` | package: `base.py` + `group_Nx.py` shards |
| `tau2_airline` | `scenarios/data/tau2_airline/` | package: `base.py` + `group_Nx.py` shards |
| `tau2_retail` | `scenarios/data/tau2_retail/` | package: `base.py` + `group_Nx.py` shards |
| `tau2_telecom` | `scenarios/data/tau2_telecom/` | package; also emits the parallel `tau2_telecom_workflow__` registrations |
| `restaurant`, `customer_service`, `qa`, `fastbite`, `simple_qa` | `scenarios/data/<name>.py` | single file (in-repo smoke sets) |

New modules must be side-imported from `scenarios/data/__init__.py` so the decorators fire at import
time. See [Authoring Domains](authoring-domains.md) for the full new-domain checklist.

## The eight per-side properties

Each scenario supplies four dataclasses per side, for both `user` and `agent` — eight properties in
total. They are defined in `nemo_voice_agent/evaluation/scenarios/classes.py` and rendered into the
system prompt by `get_user_prompt()` / `get_agent_prompt()`.

| Property | Type | Contents |
|---|---|---|
| `user_persona` / `agent_persona` | `Persona` | `role` (required), plus optional `name`, `background`, `personality`, `language`, `accent`. Rendered as the opening lines of the prompt. |
| `user_task` / `agent_task` | `Task` | `goal` (required) and optional `background`. One objective per side. |
| `user_actions` / `agent_actions` | `Actions` | `instructions` (a numbered, ordered script) and `guidelines` (always-apply rules). |
| `user_resources` / `agent_resources` | `Resources` | `tools` (`Dict[str, Dict[str, str]]` — tool class name to constructor kwargs), `documents`, `information` strings, and optional `info_sections` for structured `### heading` blocks. |

`Persona` also carries `behavior_config` and `voice_config`. Both are parked — neither is consumed by
prompt rendering or wired into the pipeline; treat them as metric-slicing labels only.

## Scenario-level fields

Use these fields to define the scenario identity, runtime limits, scoring contract, and fixture state.

| Field | Purpose |
|---|---|
| `name` | Unique scenario ID and the registry key. Convention: `<domain>__<id>`. `--domain` filters on this **name prefix**, not on the `domain` attribute. |
| `domain` | ClassVar keying the per-domain tool registry, the fixture subdirectory, and the `tool_domain` argument the bridge sends to the bots. Defaults to `"default"`. It may differ from the name prefix — `tau2_telecom_workflow__*` scenarios keep `domain = "tau2_telecom"`. |
| `description` | Short human-readable summary. |
| `max_duration` | Per-scenario cap in seconds. The CLI `--duration` defaults to `None`, so this value is what actually applies unless you pass the flag. |
| `success_signals` | **Required on every concrete scenario.** See below. |
| `reference_answer` | Expected action list, or the structured payload for legacy summary scenarios. Drives `ACTION_MATCH`. |
| `expected_scenario_db` | Optional `cached_property` holding the gold end-state DB. Drives `DB_STATE_MATCH` via SHA-256 comparison. |
| `expected_user_db` | Optional gold end-state for the user-side DB in dual-side domains. |
| `db_state_assertions` | Optional list of records shaped `side`, `func_name`, `arguments`, `assert_value`, `message`. Drives `DB_STATE_ASSERTION`. |
| `nl_assertions` | Optional list of natural-language claims judged per-claim by the LLM judge. Drives `NL_ASSERTION`. |
| `initialization_actions` | Optional list of `side` / `func_name` / `arguments` records replayed bot-side to seed fixture state before the conversation starts. |
| `ignore_capitalization`, `ignore_punctuation`, `clean_text` | String-matching normalization for the action-match comparator. |
| `disallow_extra_items` | When `True`, the list-of-dicts comparator requires an exact bijection instead of tolerating extra predicted items. |
| `noise_config` | Optional `NoiseConfig` (from `nemo_voice_agent.utils.audio`) that injects background noise into the user-to-agent channel. |

## The `success_signals` contract

`success_signals` is the whitelist of signals that gate the composite `is_successful` verdict. It must
resolve to a non-empty sequence of `SuccessSignal` members; `Scenario.__init_subclass__` raises
`TypeError` at class-definition time for any class that declares `name` without one.

| Member | Metric key | Use when |
|---|---|---|
| `ACTION_MATCH` | `is_action_match` | The domain has one canonical correct payload or trajectory. |
| `DB_STATE_MATCH` | `db_state_match` | The scenario ships `expected_scenario_db` and there is a single deterministic end state. Path-independent, so prefer it over `ACTION_MATCH`. |
| `DB_STATE_ASSERTION` | `db_state_assertion` | The solution space is open — several valid end states satisfy the same outcome predicates. |
| `NL_ASSERTION` | `nl_assertion` | The scenario carries `nl_assertions`. Requires the judge to be enabled. |
| `JUDGE_PASSED` | `judge_passed` | No deterministic check applies at all (free-form QA). |
| `CLEAN_EXIT` | `clean_exit` | Always. Every shipped domain includes it — it is `True` only when the agent voluntarily emitted the `<exit>` signal, either by calling `EndConversationTool` or via a terminal transfer tool (`TransferToAgentTool` / `TransferToHumanAgentsTool`), which prevents a timed-out conversation from scoring as a win by inaction. |

Pass-rate signals (`db_state_assertion_pass_rate`, `nl_assertion_pass_rate`) are binarized at a
threshold of `1.0` — every assertion must pass. The default verdict is a strict AND over the
whitelisted signals that produced a non-`None` value; if none were applicable the scenario scores
`"N/A"` and is excluded from the run rate. Signals *outside* the whitelist are still computed and
saved under `success_breakdown.excluded` in `metrics.json`. See [Scoring](../understand-scoring/scoring.md) and
[Metrics Reference](../../reference/evaluation/metrics.md).

Two declaration patterns cover every shipped domain — a ClassVar tuple when the whitelist is uniform,
and a `cached_property` when it depends on per-task opt-ins. A single outlier scenario can also
declare its own tuple, which shadows the base.

```python
# Uniform across the domain.
class Tau2AirlineBaseScenario(Tau2BaseScenario):
    success_signals = (SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT)

# Derived from a per-task opt-in, so it cannot drift from the data.
class Tau2RetailBaseScenario(Tau2BaseScenario):
    @cached_property
    def success_signals(self) -> tuple:
        if self.nl_assertions:
            return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.NL_ASSERTION, SuccessSignal.CLEAN_EXIT)
        return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT)
```

If strict AND is the wrong combinator for your scenario, override
`compute_is_successful(self, signals)` instead of contorting the whitelist.

## Worked example

A complete scenario for the in-repo `restaurant` domain. It inherits `agent_persona`, `agent_task`,
`user_resources`, `max_duration`, the text-normalization flags, and `success_signals` from
`RestaurantBaseScenario`, so only the scenario-specific pieces appear here.

```python
from nemo_voice_agent.evaluation.scenarios import register_eval_scenario
from nemo_voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Task
from nemo_voice_agent.evaluation.scenarios.data.restaurant import (
    PIZZA_PALACE_MENU,
    RestaurantBaseScenario,
)


@register_eval_scenario
class PizzaPepperoni(RestaurantBaseScenario):
    """Order a pepperoni pizza with extra cheese at Pizza Palace."""

    name = "restaurant__pizza_pepperoni"
    description = "Order a pepperoni pizza with extra cheese at Pizza Palace"
    reference_answer = {
        "items": [
            {"name": "Pepperoni Pizza", "unit_price": "9.99", "quantity": "1"},
            {"name": "Extra Cheese", "unit_price": "1.50", "quantity": "1"},
        ],
        "customer_name": "Charlie",
        "customer_phone": "314-527-8960",
        "total_price": "11.49",
    }

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="human user",
            name="Charlie",
            background="You are a graphic designer. Your phone number is 314-527-8960.",
            personality="Communicative and positive, with clear needs and prompt decision-making.",
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal="Order a pepperoni pizza with extra cheese.",
            background="You are hungry after work and just walked into Pizza Palace.",
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Ask the agent what pizza options are available.",
                "Order one pepperoni pizza.",
                "Ask if you can add extra cheese, and add it to the order.",
                "Confirm the order and ask for the total price.",
            ],
            guidelines=["Provide your name and phone number when asked."],
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Greet the user and ask what they would like to order.",
                "Summarize the order and confirm it is correct.",
                "Ask for the user's name and phone number.",
                "Place the order with the `PlaceOrderTool` tool and confirm it succeeded.",
                "Say goodbye and call the `EndConversationTool` tool.",
            ],
            guidelines=["Do not make up any items that are not on the menu."],
        )

    @property
    def agent_resources(self) -> Resources:
        return Resources(
            tools={
                "GetMenuTool": {"menu": PIZZA_PALACE_MENU},
                "PlaceOrderTool": {"auto_validate": "False"},
                "EndConversationTool": {},
            },
            information=["You can use the `GetMenuTool` tool to retrieve the restaurant menu."],
        )
```

`EndConversationTool` is mandatory in every domain: it emits the exit signal the bridge waits for, and
it is what makes the `CLEAN_EXIT` signal pass. Without it the bridge idles until `max_duration`
expires. Tool base classes and registration are covered in [Authoring Tools](authoring-tools.md).

## Seeding fixture data

Scenarios that need a database override `setup_shared_state(self, state, side)`. The runner calls it
once per side; the resulting dict is JSON-serialized into the `shared_state_init` argument of the
`apply_initialization` RTVI action, which the bot handler merges into its own `shared_state` before
tools are instantiated.

```python
def setup_shared_state(self, state: dict, side: str) -> None:
    if side == "agent":
        state["db_path"] = f"{self.domain}/db.json"
```

Any `db_path` value is resolved bot-side against `get_eval_data_root()` and replaced with the loaded
`db` key. Fixtures live in `nemo_voice_agent/evaluation/data/`, overridable with the `EVAL_DATA_ROOT`
environment variable. Send a path rather than inline content for anything large — the tau2 databases
exceed the WebSocket frame limit if inlined. At end of scenario the bridge pulls
`get_scenario_summary` from each bot, which returns the recorded `actions` plus a `db_hash`; the
inline DB comes back only when the bridge opts in with `include_db`, which it does for scenarios
declaring `db_state_assertions`. Dual-side domains that must propagate state between the two DBs also
override `sync_state` — see [tau2-telecom](../domain-guides/tau2-telecom.md).

## Verify

Run from the `evaluation/` directory, since `SERVER_CONFIG_PATH` and the scenario runner resolve paths
against the current working directory.

```bash
cd evaluation

# The new scenario should appear under its domain heading.
python run_evaluation.py --list

# Run it alone against a live user bot and agent bot.
python run_evaluation.py --scenarios restaurant__pizza_pepperoni
```

Scenarios that produce fewer than `--min-agent-turns` agent turns (default `3`) are counted as
**failures** in the composite success rate and skipped in the per-signal rates — a scenario that
never got off the ground is a defect, not an exclusion. Pass `--min-agent-turns 0` to disable the
filter. For bringing up the two bot servers, see the [Evaluation Quickstart](../run-evaluations/quickstart.md); for
reading the output, see [Results](../run-evaluations/results.md) and the [Eval CLI reference](../../reference/evaluation/eval-cli.md).
