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

# Authoring Tools

Evaluation tools are the functions the agent bot — and, in dual-side domains, the user-simulator bot —
can call during a scenario. They live in `nemo_voice_agent/evaluation/tools/`, register into a per-domain
registry, and are instantiated fresh per scenario by the bot server. For tools used by the interactive
server rather than the evaluator, see [Custom Tools](../../build-voice-agents/tools/custom-tools.md).

## Base classes

All evaluation tools derive from `StandardSchemaTool`
(`nemo_voice_agent/utils/tool_calling/base.py`). Pick the base that matches the tool's role.

| Base class | Module | Use for |
| --- | --- | --- |
| `StandardSchemaTool` | `nemo_voice_agent/utils/tool_calling/base.py` | Read-only lookups. No action record. |
| `WriteScenarioTool` | `nemo_voice_agent/evaluation/tools/_write_tool_base.py` | Mutating tools whose calls must land in the bridge-pulled action list. |
| `SendRTVIMessageTool` / `SendScenarioSummaryTool` / `SendExitMessageTool` | `nemo_voice_agent/evaluation/tools/rtvi_control.py` | Harness control signals sent to the bridge over RTVI. |
| `EndConversationTool` | `nemo_voice_agent/evaluation/tools/basic_tools.py` | The `<exit>` signal; include it in every scenario's agent tool list. |

A subclass implements exactly three members: `properties` (a property returning the JSON-Schema
properties dict), `required_properties` (a property returning the list of names the model must supply),
and `async def _execute(**kwargs)`, which receives the call arguments as keyword arguments and
**returns** the result.

`StandardSchemaTool.__call__` is the only pipecat-facing entry point. It calls `_execute`, converts a
raised exception into a structured `{"error": ...}` result, delivers the value through
`params.result_callback` exactly once, and then runs the `_after_result` hook. Do not accept a
`FunctionCallParams` argument in `_execute` and do not deliver the result yourself — doing both produced
duplicate `tool_call_id` deliveries before delivery moved into `__call__`.

Falsy results are wrapped before delivery by `_normalize_empty_result`, because pipecat rewrites a falsy
tool result to the literal string `COMPLETED`, which the model reads as success; an empty lookup instead
reaches the LLM as an explicit "No matching records found" envelope. Never apply that normalization
inside `_execute` — the synchronous replay path needs the raw return shape.

The LLM-visible function name defaults to the Python class name. Set a class-level `name` attribute to
override it, as the tau2 telecom tools do (`name = "get_customer_by_phone"`), which makes the registry
key, the schema name, the action record's `name`, and the replay dispatch key one snake_case identifier.

## Shared state

Every tool in a scenario receives the *same* mutable `shared_state` dict, auto-injected into any
constructor that declares `shared_state: Optional[dict]`. The convention is to store it as `self.state`.

| Key | Written by | Read by |
| --- | --- | --- |
| `db` | `apply_initialization` (resolves `db_path` under the eval data root) | Tool `_do_work` / `_execute` bodies |
| `actions` | `WriteScenarioTool._record_action` | The bridge's `get_scenario_summary` pull |
| `_call_counts` | `WriteScenarioTool._next_call_index` | Tools that mint unique IDs |
| `__rtvi__` | `update_system_prompt` handler | `_record_action` (emits `action-applied`), exit-message helpers |
| `__tool_domain__` | `update_system_prompt` handler | `action-applied` payload routing |

Scenario fixture data arrives through the `shared_state_init` argument of the `apply_initialization` RTVI
client message, not `update_system_prompt`. Dunder sentinels live on `shared_state` itself, not inside
`shared_state["db"]`, so they never reach the DB hash — `get_dict_hash` hashes only the `db` dict and drops
just the top-level keys listed in `HASH_EXCLUDED_KEYS` (currently only `session`). See
[RTVI Messages](../../reference/runtime/rtvi-messages.md).

## Termination contracts

Two patterns coexist. New benchmarks should use the bridge-pull pattern.

| Pattern | Domains | How the bridge captures results |
| --- | --- | --- |
| Bridge-pull (preferred) | `eva_airline`, all `tau2_*` | Write tools call `self._record_action(...)`; at scenario end the bridge pulls `{actions, db_hash}` from each bot via the `get_scenario_summary` action (with an opt-in `include_db` when the scenario has DB-state assertions). No LLM-callable summary exists. |
| LLM summary (legacy) | in-repo smoke sets: `restaurant` (including its waitlist scenario), `customer_service`, `qa`, `fastbite`, `simple_qa` | A `SendScenarioSummaryTool` subclass wraps the agent's structured result in `<final_response>` tags; the bridge writes it to `final_agent_response.json`. |

Both patterns need `EndConversationTool` in the agent's tool list. It emits `<exit>`, which stops the
scenario early; without it the bridge waits out the scenario's `max_duration`. `CLEAN_EXIT` is one of the
six scoring signals — see [Scoring](../understand-scoring/scoring.md).

Terminal tools that both record an action and end the call (`TransferToHumanAgentsTool`) emit the exit
signal from `_after_result`, never from `_execute`: pipecat must commit the tool-call record before the
bridge tears the session down, or the captured `llm_context.json` loses the final tool call.

## Add a tool

Subclass the right base, register it with the domain, and side-import the module.

```python
from typing import Any, Dict, List, Optional

from nemo_voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool


COFFEE_ACTION_TYPES: List[str] = ["cancel_order"]


@register_schema_tool_for_eval(domain="coffee")
class CancelOrderTool(WriteScenarioTool):
    """Mutating tool — records an action the bridge pulls at scenario end."""

    ACTION_TYPES = COFFEE_ACTION_TYPES

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or "Cancel a pending coffee order.")
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {"order_id": {"type": "string", "description": "The order id, such as 'A123'."}}

    @property
    def required_properties(self) -> List[str]:
        return ["order_id"]

    async def _execute(self, order_id: str = "") -> Dict[str, Any]:
        # Normalize case: voice ASR of a spelled-out id is inconsistent.
        orders = (self.state.get("db") or {}).get("orders") or {}
        order = orders.get((order_id or "").upper())
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {order_id} not found"}
        order["status"] = "cancelled"
        self._record_action(
            {
                "action_type": "cancel_order",
                "name": "cancel_order",
                "arguments": {"order_id": order_id},
                "result": {"order_id": order_id},
            }
        )
        return {"status": "success", "order_id": order_id}
```

A read-only tool has the same shape with `StandardSchemaTool` as the base, no `ACTION_TYPES`, and no
`_record_action` call. `ACTION_TYPES` is a `ClassVar` list that `_record_action` validates `action_type`
against; a mismatch logs a warning rather than raising, so check the bot log when an action fails to
score. The record's `name` field is the upstream method name used for action-list comparison and is
independent of the class name.

Add the side-import to `nemo_voice_agent/evaluation/tools/__init__.py` so the decorator fires at import
time, then check the registration:

```bash
python -c "
from nemo_voice_agent.evaluation.tools import list_schema_tools_for_eval
print(sorted(list_schema_tools_for_eval('coffee')))
"
```

## Constructor auto-injection

Scenarios reference tools by registry name in `Resources.tools`, as `{name: constructor_kwargs}`. The
bridge serializes that mapping to JSON, and the bot server instantiates each entry through
`get_schema_tool_for_eval(name, domain=..., rtvi=..., shared_state=..., **tool_args)`.

That factory inspects the constructor signature and injects `shared_state` and `rtvi` only if the tool
declares them; `rtvi` is what `SendRTVIMessageTool` subclasses need, while `WriteScenarioTool` reads the
processor from `shared_state["__rtvi__"]` instead. Every other parameter comes from the scenario's
per-tool kwargs, so one class can be reused with different fixtures (a different `menu` string per
scenario, for example). Keep constructors keyword-only with `Optional` defaults so a tool stays
constructible in a unit test with no arguments.

## Per-domain registry

`ALL_SCHEMA_TOOLS_FOR_EVAL` is a `dict` of domain to a `dict` of name to class. The same short class name
can exist in several domains; within one domain a duplicate name raises `ValueError` at decoration time.

```python
@register_schema_tool_for_eval(domain="tau2_airline")   # keyword form
@register_schema_tool_for_eval("tau2_airline")          # positional shortcut
@register_schema_tool_for_eval                          # bare: registers into "default"
```

Use the bare form only for cross-domain harness tools such as `EndConversationTool`.

| Domain | Modules | Registered tools |
| --- | --- | --- |
| `default` | `basic_tools.py`, `rtvi_control.py`, `customer_service_tools.py`, `restaurant_tools.py`, `waitlist_tools.py` | 20 |
| `eva_airline` | `eva_airline_tools.py` | 15 |
| `tau2_airline` | `tau2_airline_tools.py` | 14 |
| `tau2_retail` | `tau2_retail_tools.py` | 16 |
| `tau2_telecom` | `tau2_telecom_tools.py` (13 agent-side), `tau2_telecom_user_tools.py` (30 user-side) | 43 |

The domain key comes from `Scenario.domain`, which the bridge forwards as the `tool_domain` argument of
`update_system_prompt`. Lookup tries that domain first, then falls back to `"default"` with a logged
warning, and raises `KeyError` listing the available names if neither has the tool. The telecom user-side
tools are exposed to the user simulator, which is why the eval user bot config sets
`llm.enable_tool_calling: true`.

## Synchronous `invoke` for replay

The tau2 domains add a sync `invoke(**kwargs)` alongside the async `_execute`, via `_Tau2InvokeMixin`.
Both route through `_do_work(p)` after validating arguments with a pydantic `PARAMS_MODEL`, so a
validation failure returns a structured error dict instead of a traceback (`error_type: "validation_error"`
for airline and retail, `error_type: "invalid_arguments"` for telecom). `invoke` is what
the runner's gold replay and the cross-side sync shadow-DB replay call in process. Any scenario that
overrides `Scenario.sync_state` must also provide `_build_tool_map(state)` returning tools with a sync
`invoke`. Single-side domains do not need it.

## Next Steps

After implementing a tool, register it in a scenario and verify that live execution and any gold replay use
the same state mutation.

- [Author scenarios](authoring-scenarios.md) to expose the tool and select its success signals.
- [Add a domain](authoring-domains.md) to register a new tool namespace and fixture set.
