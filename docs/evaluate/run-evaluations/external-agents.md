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

# Evaluating an External Agent

The evaluation harness never imports your agent. It talks to two bots — the simulated user and the agent
under test — over WebSockets, so any agent that speaks the same wire protocol can be scored on the shipped
domains. This page is the contract: what your bot must implement, what the harness reads back, and how to
verify a new bot before you spend GPU hours on a full run.

The reference implementation is `evaluation/bot_server.py`. Read it alongside this page — everything below
is visible there in about 100 lines.

## The contract

An external agent must satisfy two hard requirements before the harness can initialize or score it.

**1. A pipecat WebSocket server transport.** The bridge connects with the `websockets` client library and
frames everything through `ProtobufFrameSerializer`, so use a pipecat WS transport rather than
reimplementing the framing. The shipped builder (`build_ws_transport` in
`nemo_voice_agent/pipecat/services/nemo/builders.py`) constructs a
`SingleClientWebsocketServerTransport` with `audio_in_enabled`, `audio_out_enabled`, no session timeout, and
no WAV header. Bind it to the port in `WEBSOCKET_PORT` — the bridge defaults to `ws://localhost:8765` for
the agent and `ws://localhost:8766` for the user sim, overridable with `--agent-url` / `--user-url`.

**2. An `RTVIProcessor` with six client-message handlers registered.** These are the entire control plane
for a scenario.

| Wire name | Direction | The harness needs it for |
| --- | --- | --- |
| `update_system_prompt` | bridge to bot | Per-scenario system prompt, tool surface, and `tool_domain`; clears prior `shared_state`. |
| `apply_initialization` | bridge to bot | Merges `shared_state_init`, resolves `db_path` to `db`, applies init-function mutations. |
| `apply_sync_delta` | bridge to bot | Cross-side state sync (telecom only); harmless no-op elsewhere, but must be registered. |
| `get_scenario_summary` | bot to bridge | End-of-scenario pull of `actions` plus `db_hash`, with opt-in `include_db`. |
| `get_context_history` | bot to bridge | End-of-scenario LLM context, saved as `bot_logs_agent/llm_context.json` and fed to the judge. |
| `reset` | bridge to bot | Clears conversation history and resets stateful services between scenarios. |

All six factories live in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`. Each returns a
`(wire_name, handler)` pair; install them with one call to `register_client_message_handlers`. Per-handler
argument and return shapes are documented in [RTVI Control Plane](../../build-voice-agents/extend/protocols/rtvi-actions.md). An unregistered
type produces an `error-response`, which surfaces in `bridge_log.txt` as `unknown message type`.

Reusing the shipped factories is by far the cheapest path — they are pure functions over your pipeline
objects and carry no dependency on the NeMo services. For pipeline assembly around them, see
[Building Your Own Pipeline](../../build-voice-agents/extend/pipelines/custom-pipeline.md).

```python
from nemo_voice_agent.evaluation.tools import get_schema_tool_for_eval
from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    SharedStateRef,
    TaskRef,
    create_apply_initialization_action,
    create_apply_sync_delta_action,
    create_get_context_history_action,
    create_get_scenario_summary_action,
    create_reset_context_action,
    create_update_system_prompt_action,
    register_client_message_handlers,
)
from nemo_voice_agent.utils.tool_calling import register_schema_tools_to_llm

task_ref = TaskRef()
shared_state_ref = SharedStateRef()

register_client_message_handlers(
    rtvi,
    [
        create_reset_context_action(task_ref, user_agg, assistant_agg, original_messages, resettable),
        create_update_system_prompt_action(
            task_ref, user_agg, assistant_agg, original_messages, resettable,
            system_role="system",
            system_prompt_suffix="",
            enable_tool_calling=True,
            llm=llm,
            context=context,
            rtvi=rtvi,
            tool_factory=get_schema_tool_for_eval,
            register_schema_tools=register_schema_tools_to_llm,
            shared_state_ref=shared_state_ref,
        ),
        create_get_context_history_action(task_ref, assistant_agg),
        create_get_scenario_summary_action(task_ref, shared_state_ref),
        create_apply_initialization_action(shared_state_ref),
        create_apply_sync_delta_action(shared_state_ref),
    ],
)
```

If you supply your own `tool_factory` instead of `get_schema_tool_for_eval`, keep the signature
`(name, domain=..., rtvi=..., shared_state=..., **tool_args)` — the bridge sends the scenario's `domain` as
`tool_domain` and the factory must resolve names in that namespace.

## Runtime behaviors the harness assumes

Beyond the six handlers, the bridge relies on several behaviors that the reference server gets from
`run_bot_websocket_server` in `nemo_voice_agent/pipecat/bot_server.py` and from the `RTVIObserver` in
`nemo_voice_agent/pipecat/processors/frameworks/rtvi.py`.

| Behavior | Why it matters |
| --- | --- |
| Answer `client-ready` with `bot-ready` (`rtvi.set_bot_ready()`) | The bridge waits up to 5 s for the handshake on every connection. |
| Survive disconnect and reconnect without losing scenario state | `prepare_for_scenario` opens a setup connection, sends the prompt and initialization, then **closes it**; the audio phase reconnects. Prompt, tools, and `shared_state` must persist. |
| Never end the pipeline on client disconnect | The WebSocket server lives inside the input transport — ending the pipeline kills the port and nothing can reconnect. |
| Accept the `send-text` kickoff | The agent bot is started by an RTVI `send-text` with `run_immediately`, sent 1 s into the scenario. Pipecat's `RTVIProcessor` handles this natively. |
| Emit `bot-started-speaking`, `bot-tts-text`, `bot-stopped-speaking` | The bridge builds `conversation_log.txt`, the segLST file, and per-turn latency from these events. |
| Emit `metrics` messages carrying token usage | `token_usage.agent.n_calls` is the turn counter behind `--min-agent-turns` (default 3). A bot that never reports usage counts 0 turns and every scenario is scored a **failure**. Set `enable_metrics=True` and `enable_usage_metrics=True` on `PipelineParams`. |
| Push `<exit>` as an RTVI server message when the agent ends the call | This is the `CLEAN_EXIT` signal, which is in every domain's whitelist. `EndConversationTool` in `nemo_voice_agent/evaluation/tools/basic_tools.py` does it; without it every scenario terminates on timeout and fails. |
| Push `action-applied` server messages from write tools | Only needed for dual-side domains such as `tau2_telecom`, where the bridge uses them to drive cross-side sync. |

Scoring reads two pieces of bot-owned state, both keyed off `shared_state`:

- `shared_state["actions"]` — appended to by each write tool; returned by `get_scenario_summary` and used for
  `ACTION_MATCH` and judge input.
- `shared_state["db"]` — mutated in place by write tools; hashed with `get_dict_hash` for `DB_STATE_MATCH`,
  or returned inline when the bridge asks for `include_db` so `DB_STATE_ASSERTION` predicates can run.

If you reuse the tool classes under `nemo_voice_agent/evaluation/tools/`, both come for free. If you route
tool calls through your own agent framework, mirror the same bookkeeping or those signals evaluate as not
applicable. See [Scoring](../understand-scoring/scoring.md) for the six signals and how they combine.

## Running a scenario against your bot

`SERVER_CONFIG_PATH` is resolved against the current working directory, so run everything from `evaluation/`.

```bash
cd evaluation

# Terminal 1 — your agent bot on 8765
WEBSOCKET_PORT=8765 python /path/to/your_bot_server.py

# Terminal 2 — the stock simulated user on 8766
WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml python bot_server.py

# Terminal 3 — one fast scenario first
python run_evaluation.py --scenarios restaurant__pizza_pepperoni
```

Point the bridge elsewhere if your bot is on another host or port:

```bash
python run_evaluation.py --agent-url ws://10.0.0.7:9000 --scenarios restaurant__pizza_pepperoni
```

Then inspect `eval_results/eval_<timestamp>/restaurant__pizza_pepperoni/`:

| Artifact | What a healthy run looks like |
| --- | --- |
| `bridge_log.txt` | No `unknown message type` lines. A `[AGENT SERVER MESSAGE]` line containing `<exit>`. |
| `metrics.json` | `is_successful` is `true`, `false`, or `"N/A"` — never missing. `token_usage.agent.n_calls` is above `--min-agent-turns`. |
| `conversation_log.txt` | Alternating user and agent turns. Empty means the speaking or TTS-text events are not reaching the bridge. |
| `bot_logs_agent/llm_context.json` | Starts with the scenario system prompt; tool calls appear as assistant `tool_calls`. |
| `final_scenario_db_hash.txt` | Contains a `db_hash:` line for DB-scored domains. |

## Optional: `trace_metrics.json` passthrough

Architecture-specific diagnostics — internal handoff quality in a cascaded agent, router confidence, retry
counts — do not belong in the evaluator's fixed metric set, so the runner offers a passthrough instead. Write
a JSON object to either path inside the scenario directory:

```
<output-dir>/eval_<timestamp>/<scenario-name>/trace_metrics.json
<output-dir>/eval_<timestamp>/<scenario-name>/bot_logs_agent/trace_metrics.json
```

The first path that exists is loaded verbatim into `metrics.json["trace_metrics"]`. The runner does not
validate, interpret, or aggregate the contents, and the file is optional — when absent, the key is simply
missing from `metrics.json`. The loader is `_load_optional_trace_metrics` in
`nemo_voice_agent/evaluation/runner.py`.

Your bot has to know where to write. The scenario directory name is the scenario name, under the run's
`eval_<timestamp>` session directory beneath `--output-dir`; pass it into your bot out of band, or write from
a post-run script that walks the session directory.

## Non-pipecat agents

Out of scope. An agent built on a different runtime would have to reimplement pipecat's WebSocket transport
framing and the RTVI message layer before the bridge could talk to it. If you take that on, treat
`pipecat.transports.websocket.server` and `pipecat.serializers.protobuf` as the wire specification, and
[RTVI Message Reference](../../reference/runtime/rtvi-messages.md) as the message-level one.

## Related pages

Use these pages to assemble an agent around the contract, run it, and interpret the resulting evidence.

- [Building Your Own Pipeline](../../build-voice-agents/extend/pipelines/custom-pipeline.md) — assembling a bot around the contract
- [RTVI Control Plane](../../build-voice-agents/extend/protocols/rtvi-actions.md) — per-handler arguments and return shapes
- [RTVI Message Reference](../../reference/runtime/rtvi-messages.md) — the on-the-wire envelope
- [Scoring](../understand-scoring/scoring.md) — the six success signals
- [Evaluation CLI](../../reference/evaluation/eval-cli.md) — every flag the runner accepts
- [Results](results.md) — what lands in the scenario directory
