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

# RTVI Message Reference

Use this reference to look up the custom real-time voice interface (RTVI) client messages that a NeMo Labs
Voice Agent bot handles. The handlers live in
`nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`. For handler behavior and extension
instructions, refer to [RTVI Control Plane](../../build-voice-agents/extend/protocols/rtvi-actions.md). For
the audio interface on the same socket, refer to
[Client Protocol](../../build-voice-agents/extend/protocols/client-protocol.md).

## Envelope

Control messages travel on the same WebSocket as audio. Pipecat wraps each message in a `MessageFrame`, and
`ProtobufFrameSerializer` encodes the frame. The frame body is a JSON string in Pipecat's RTVI envelope:

```json
{
  "label": "rtvi-ai",
  "type": "client-message",
  "id": "get_scenario_summary_1754500000.0",
  "data": { "t": "get_scenario_summary", "d": { "include_db": false } }
}
```

`t` is the message type (the wire name in the table below) and `d` is the argument object. `id` is
caller-chosen and echoed back — match on it, because responses to concurrent requests arrive interleaved.

The `d` field of a `server-response` contains the handler return value:

```json
{
  "label": "rtvi-ai",
  "type": "server-response",
  "id": "get_scenario_summary_1754500000.0",
  "data": { "t": "get_scenario_summary", "d": { "actions": [], "db_hash": null } }
}
```

An unknown `t` or a handler exception produces an `error-response`. This response lets callers fail fast
instead of blocking until a read timeout:

```json
{
  "label": "rtvi-ai",
  "type": "error-response",
  "id": "get_scenario_summary_1754500000.0",
  "data": { "error": "KeyError: 'db'" }
}
```

For browser clients built on `@pipecat-ai/client-js`, `sendClientRequest(type, args)` builds the envelope and
resolves with the `d` payload. The evaluation bridge (`nemo_voice_agent/evaluation/bridge.py`) constructs the
envelope directly, as shown in the examples on this page.

## Message Types

The following table summarizes the custom request types registered by the example and evaluation servers.

| `t` | Arguments | Response `d` | Registered by |
| --- | --- | --- | --- |
| `reset` | none | `true` or `false` | example server and eval bot |
| `update_system_prompt` | `prompt`, `tools`, `add_suffix`, `tool_domain` | `true` or `false` | eval bot |
| `get_context_history` | none | `context` | eval bot |
| `get_scenario_summary` | `include_db` | `actions`, `db_hash`, optional `db` | eval bot |
| `apply_initialization` | `domain`, `shared_state_init`, `actions` | `success`, `errors` | eval bot |
| `apply_sync_delta` | `domain`, `delta` | `success`, `errors` | eval bot |

`examples/generic_voice_agent/server/server.py` registers `reset` only. `evaluation/bot_server.py` registers
all six. There is no schema or defaults layer on this path: every handler reads arguments with
`arguments.get(name, default)`, so unknown keys are ignored and missing keys fall back silently.

### reset

Send `"d": {}`. Response `d` is a bare boolean — `true` on success, `false` if the aggregator reset raised.

### update_system_prompt

Use the following arguments to replace the evaluation bot's system prompt and register scenario tools.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `prompt` | string | `""` | Required. An empty value returns `false` without changing anything. |
| `tools` | string | `"{}"` | JSON string mapping tool name to constructor kwargs. |
| `add_suffix` | bool | `true` | Appends the server's configured system-prompt suffix. |
| `tool_domain` | string | `"default"` | Tool-registry namespace for the lookup. |

Response `d` is a bare boolean. The `tools` value is a **JSON string**, not a nested object.

```json
{"t": "update_system_prompt",
 "d": {"prompt": "You are a support agent.",
       "tools": "{\"GetUserDetailsTool\": {}}",
       "add_suffix": false,
       "tool_domain": "tau2_airline"}}
```

Scenario fixture data does not travel here — `shared_state_init` is an argument of `apply_initialization`.

### get_context_history

Send `"d": {}`. Response `d` is `{"context": "..."}` where the value is the assistant aggregator's message
list rendered with Python `str()`, after `sanitize_context_for_transport` has replaced inline audio, image,
and file blobs with placeholder tags. On an internal error the handler returns `context` as an empty list.

### get_scenario_summary

Use the following argument to control whether the response includes the bot's inline database.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `include_db` | bool | `false` | Also inline this bot's `db` dict in the response. |

```json
{"actions": [{"name": "book_reservation", "action_type": "...", "arguments": {}}],
 "db_hash": "3f2a…",
 "db": null}
```

`db_hash` is the SHA-256 of the canonicalized DB from `nemo_voice_agent.evaluation.db_hash.get_dict_hash`, or
`null` when the bot holds no DB. The `db` key is present only when `include_db` was true; keep it off for
large databases, which overflow Pipecat's 1 MB WebSocket frame cap and close the connection with code 1009. Each
bot returns only its own DB — the agent-versus-user labeling is applied by the caller, based on which socket
the response arrived on.

### apply_initialization

Use the following arguments to seed shared state, load fixture data, and apply initialization actions.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `domain` | string | `"default"` | Registry bucket for the initialization functions. |
| `shared_state_init` | string | `"{}"` | JSON string; must decode to an object. Merged into `shared_state`. |
| `actions` | list | `[]` | Records of `func_name` plus `arguments`; an upstream `side` field is carried but ignored bot-side. |

```json
{"t": "apply_initialization",
 "d": {"domain": "tau2_telecom",
       "shared_state_init": "{\"db_path\": \"tau2_telecom/db.json\"}",
       "actions": [{"func_name": "set_data_usage", "arguments": {"usage_gb": 15.1}}]}}
```

Response `d` is `{"success": bool, "errors": [string]}`. `success` is false — with an explanatory entry in
`errors` — when `shared_state_init` is not valid JSON or does not decode to an object, when `actions` is not
a list, or when actions were supplied but no `db` could be resolved. A `db_path` is resolved against the eval
data root (`nemo_voice_agent/evaluation/data/`, overridable with `EVAL_DATA_ROOT`) and is skipped when `db` is
already present.

### apply_sync_delta

Use the following arguments to apply a cross-side state update to the bot's database.

| Argument | Type | Default | Notes |
| --- | --- | --- | --- |
| `domain` | string | `"default"` | Selects the per-domain delta applier. |
| `delta` | object | `{}` | Dotted-path keys to new values; must be an object. |

```json
{"t": "apply_sync_delta",
 "d": {"domain": "tau2_telecom",
       "delta": {"surroundings.line_active": true,
                 "bills[B1002].status": "Paid"}}}
```

Response `d` is `{"success": bool, "errors": [string]}`. It is false when `delta` is not a dict or the bot has
no `db` in `shared_state`. Failures are informational for the caller — a malformed delta must not stall the
conversation.

## Server-to-Client Messages

The bot also pushes unsolicited messages that the evaluation bridge monitors. These are Pipecat
`server-message` frames with a custom `data` payload, plus the standard RTVI lifecycle and transcript events.

| `type` | `data` shape | Emitted by |
| --- | --- | --- |
| `bot-ready` | Pipecat's ready payload | Answer to the `client-ready` handshake. |
| `server-message` | `type` = `action-applied`, plus `action` and `tool_domain` | `WriteScenarioTool._record_action` in `nemo_voice_agent/evaluation/tools/_write_tool_base.py` |
| `server-message` | `text` containing `<exit>` or `<final_response>` markers | The RTVI control tools in `nemo_voice_agent/evaluation/tools/rtvi_control.py` |

The bridge also uses two Pipecat built-ins on the outbound path. The `client-ready` handshake awaits
`bot-ready`. The `send-text` message injects a text turn into the bot's large language model (LLM) with the
`run_immediately` and `audio_response` options. Both are Pipecat protocol messages, not handlers defined in
this repository.

The evaluation harness captures wire traffic in `bridge_log.txt` inside each scenario result directory. For
the result workflow, refer to [Evaluation](../../evaluate/index.md).
