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

# RTVI Control Plane

Every NeMo Labs Voice Agent bot embeds a pipecat `RTVIProcessor` in its pipeline. Alongside the audio
stream, a connected client can send **client messages** over the same WebSocket to inspect or mutate bot
state: clear the conversation, swap the system prompt and tool surface, snapshot the LLM context, or seed
scenario fixtures. That is the control plane.

The handlers live in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`. Each is produced by a
`create_*_action` factory that returns a `(wire_name, handler)` pair; the bot installs them all with a single
call to `register_client_message_handlers`. For the on-the-wire envelope (`client-message` /
`server-response` / `error-response` and the `t` / `d` argument encoding), see
[RTVI Message Reference](../../../reference/runtime/rtvi-messages.md).

> Pipecat 1.0 removed the `RTVIAction` / `RTVIProcessor.register_action()` API. The wire names below survive
> from that era — the code still calls them "actions" — but they are now plain client-message types.

## The six handlers

| Wire name | Factory | Returns | Registered by |
| --- | --- | --- | --- |
| `reset` | `create_reset_context_action` | `bool` | example server + eval bot |
| `update_system_prompt` | `create_update_system_prompt_action` | `bool` | eval bot |
| `get_context_history` | `create_get_context_history_action` | `context` string | eval bot |
| `get_scenario_summary` | `create_get_scenario_summary_action` | `actions` + `db_hash` | eval bot |
| `apply_initialization` | `create_apply_initialization_action` | `success` + `errors` | eval bot |
| `apply_sync_delta` | `create_apply_sync_delta_action` | `success` + `errors` | eval bot |

`examples/generic_voice_agent/server/server.py` registers only `reset` — that is all the browser client
needs. `evaluation/bot_server.py` registers all six, because the evaluation bridge drives the full scenario
lifecycle over this channel.

### `reset`

Takes no arguments. Resets the user and assistant aggregators, re-seeds both with a deep copy of
`original_messages` (captured by reference, so it always reflects whatever `update_system_prompt` last
wrote), and calls `.reset()` on every entry of `resettable_services` that defines one. `None` entries are
skipped. The browser client wires this to its **Reset** button via `sendClientRequest('reset', {})`.

### `update_system_prompt`

Arguments: `prompt` (required), `tools` (JSON string), `add_suffix` (default `true`), `tool_domain`
(default `"default"`).

Replaces the system message, optionally appending the configured system-prompt suffix, then resets both
aggregators. It is also the **scenario-start gate**: it clears `shared_state` in place — via `dict.clear()`,
so tools holding a reference keep seeing the same object — and stashes two bot-side runtime sentinels,
`__rtvi__` (used by write tools to emit `action-applied` messages) and `__tool_domain__`. When tool calling
is enabled and a `tools` payload is present, each entry is built through the injected `tool_factory` and
swapped onto the LLM by `register_schema_tools`, replacing the previous tool set.

It does **not** load scenario fixture data. `db_path`, DB contents, and custom shared-state keys all arrive
through `apply_initialization`.

### `get_context_history`

No arguments. Returns the assistant aggregator's message list, stringified, as `context`. Before
snapshotting it polls the aggregator's `has_function_calls_in_progress` property until in-flight function
calls commit, with a hard 3-second deadline so a stuck tool degrades to a warning instead of a deadlock. The
snapshot then passes through `sanitize_context_for_transport`, which replaces raw audio/image/file blobs with
placeholder tags — omni models keep audio inline, and serializing it verbatim overflows pipecat's WebSocket
frame cap. Sanitization is non-mutating; the live context keeps its bytes.

### `get_scenario_summary`

Argument: `include_db` (default `false`).

Returns `actions` (the auto-aggregated write-tool records from `shared_state`) and `db_hash`, the SHA-256 of
the canonicalized DB computed by `get_dict_hash` from `nemo_voice_agent.evaluation.db_hash`. The DB itself
stays on the bot server — the runner computes the same hash from its in-process gold replay and compares
strings, which keeps the payload small regardless of DB size. Setting `include_db` to true adds the inline
`db` dict; the bridge only does that for domains whose per-predicate `db_state_assertions` need real values
and whose DB is small enough to cross the frame limit.

Each bot returns only its own DB. The `db` versus `user_db` distinction is applied bridge-side, based on
which WebSocket the response came from.

### `apply_initialization`

Arguments: `domain`, `shared_state_init` (JSON string), `actions` (list).

The single scenario-state initializer, called once per bot immediately after `update_system_prompt`. It does
three things in order:

1. Merges the decoded `shared_state_init` into `shared_state`, preserving the runtime sentinels.
2. If the merged state carries a `db_path`, resolves it under the eval data root and loads it into `db`.
   Idempotent — skipped when `db` is already present, and the redundant `db_path` key is dropped either way.
3. Dispatches each `func_name` / `arguments` record in `actions` against the loaded `db`, using the
   registry in `nemo_voice_agent/evaluation/initialization_functions.py`.

Returns `success` and `errors`. The bridge calls it for every scenario even when there are no init actions,
because steps 1 and 2 must still run. Any `success: false` aborts the scenario rather than scoring partially
seeded state.

### `apply_sync_delta`

Arguments: `domain`, `delta` (dict).

Bot-side endpoint of the cross-side state-sync pipeline. After a write tool on one bot emits `action-applied`,
the bridge replays it onto its shadow DBs, runs the scenario's `sync_state`, and pushes any non-empty per-side
delta here. The handler dispatches through the per-domain applier registry in
`nemo_voice_agent/evaluation/sync_appliers.py`, which mutates `shared_state["db"]` in place; unregistered
domains fall back to a generic dotted-path setter. Returns `success` and `errors`, both informational — a
malformed delta should not stall the conversation. Only scenarios that override `Scenario.sync_state` (today:
telecom) trigger it, but the handler is registered domain-agnostically.

## `register_client_message_handlers`

```python
from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    TaskRef,
    create_reset_context_action,
    register_client_message_handlers,
)

task_ref = TaskRef()
register_client_message_handlers(
    rtvi,
    [create_reset_context_action(task_ref, user_agg, assistant_agg, original_messages, resettable)],
)
```

The function builds a dispatch table from the `(name, handler)` pairs and installs one `on_client_message`
event handler on the processor. A handler's return value becomes the `d` payload of the `server-response`
keyed to the request id. An unknown message type or a raised exception produces an `error-response` instead,
so a caller fails loudly rather than blocking until its read timeout expires.

Import path matters. The `nemo_voice_agent.pipecat.processors.frameworks` package re-exports a subset of the
module: `TaskRef`, `register_client_message_handlers`, `sanitize_context_for_transport`, and the `reset` /
`update_system_prompt` / `get_context_history` factories. `SharedStateRef`, `ClientMessageHandler`, and the
three evaluation-specific factories must be imported from the `rtvi_actions` module directly.

## `TaskRef` and `SharedStateRef`

Both are small mutable dataclasses that exist to break a construction-order cycle — the handlers are built
before the objects they need.

| Holder | Fields | Populated by |
| --- | --- | --- |
| `TaskRef` | `task`, `running` | `run_bot_websocket_server` in `nemo_voice_agent/pipecat/bot_server.py` |
| `SharedStateRef` | `state` (dict) | `update_system_prompt` clears it; `apply_initialization` fills it |

`TaskRef` exists because the `PipelineWorker` cannot be constructed until after the RTVI processor (the
worker takes `rtvi` in its observer list). Create the ref early, hand it to the factories, and
`run_bot_websocket_server` sets `task` and flips `running` to true; the runner flips it back during shutdown
so a handler can avoid queueing frames onto a dead task.

`SharedStateRef.state` is the same dict passed to tool constructors, which is why handlers can read
`actions` and `db` without holding tool references. Its identity is never reassigned, so tools registered for
an earlier scenario stay valid.

## Adding your own handler

Write a factory that closes over whatever pipeline objects it needs and returns a `(name, coroutine)` pair,
then append it to the list you pass to `register_client_message_handlers`:

```python
from typing import Any

from pipecat.processors.frameworks.rtvi import RTVIProcessor

from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import ClientMessageHandler, TaskRef


def create_ping_action(task_ref: TaskRef) -> ClientMessageHandler:
    async def handler(rtvi_processor: RTVIProcessor, arguments: dict[str, Any]) -> dict:
        return {
            "echo": arguments.get("payload", ""),
            "running": task_ref.running,
        }

    return ("ping", handler)
```

Two rules:

- **Read arguments defensively.** There is no schema or defaults layer on this path. Every shipped handler
  uses `arguments.get(name, default)`.
- **Never end the pipeline from a handler.** The WebSocket server lives inside the input transport, so
  tearing down the pipeline also kills the server and nothing can reconnect. `reset` and
  `update_system_prompt` run at the start of every evaluation scenario; ending the pipeline there would kill
  the bot before its first turn.

If you need bot-to-client notifications instead of request/response, push an `RTVI.ServerMessage` with
`rtvi.push_transport_message(...)` — that is how write tools emit `action-applied`.

## Related pages

- [RTVI Message Reference](../../../reference/runtime/rtvi-messages.md) — wire format and per-message payloads
- [Client Protocol](client-protocol.md) — connecting a client to the bot
- [Custom Pipeline](../pipelines/custom-pipeline.md) — assembling a bot that registers its own handlers
