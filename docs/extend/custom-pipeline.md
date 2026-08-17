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

# Building Your Own Pipeline

This is the deepest extension tier in NeMo Labs Voice Agent: you write your own `run_bot_websocket()`
instead of reusing the shipped one. Every service, every processor, and the pipeline shape itself
become yours. Only a narrow transport + control-plane contract has to survive.

Reach for it only after the cheaper tiers are exhausted: swapping a model behind an existing stage is
YAML-only ([Server Configuration](../configure/server-config.md)), and a transform between two stages
is one class ([Custom Frame Processors](./custom-processor.md)). This tier is for a genuinely
different pipeline shape, or services the [builders](./builders.md) do not cover.

## Two entry points, two contracts

There are two `run_bot_websocket()` implementations, and they demand different things from a replacement.

| Entry point | Consumer | RTVI handlers it registers |
|---|---|---|
| `examples/generic_voice_agent/server/server.py` | Browser client over the pipecat WebSocket transport | `reset` only |
| `evaluation/bot_server.py` | The eval bridge (`nemo_voice_agent/evaluation/bridge.py`) | all six |

Both call the shared runner `run_bot_websocket_server()` from `nemo_voice_agent/pipecat/bot_server.py`,
which owns the boilerplate around the task: transport connect/disconnect handlers, the RTVI
`on_client_ready` kickoff, audio-logger finalization, and shutdown. It makes no assumptions about
pipeline contents, so a custom pipeline can keep using it.

## The contract

A replacement bot must satisfy exactly two requirements.

**1. Speak pipecat's WebSocket server protocol.** Use `SingleClientWebsocketServerTransport` from
`pipecat.transports.websocket.server` with a `ProtobufFrameSerializer`, bound to the port in
`WEBSOCKET_PORT`. `build_ws_transport()` in `nemo_voice_agent/pipecat/services/nemo/builders.py`
constructs exactly this and is the path of least resistance — it also reads the transport sample
rates from your config. Since pipecat 1.0 the transport no longer runs VAD, so a VAD processor goes
in the pipeline right after `transport.input()`.

**2. Carry an `RTVIProcessor` with the handlers your consumer expects.** Build the handlers with the
factories in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py` and install them in one
shot with `register_client_message_handlers()`. Each factory returns a `(message_type, handler)` pair;
the registrar installs a single `on_client_message` dispatcher over all of them. An unhandled type
produces an `error-response` rather than a silent hang.

The six factories, keyed by the wire message type they answer:

| Wire type | Factory | Direction | Purpose |
|---|---|---|---|
| `reset` | `create_reset_context_action` | bridge to bot | Clear conversation history and call `.reset()` on stateful services. |
| `update_system_prompt` | `create_update_system_prompt_action` | bridge to bot | Set the system prompt, register tools from the `tool_domain` registry, re-point `shared_state`. |
| `apply_initialization` | `create_apply_initialization_action` | bridge to bot | Merge `shared_state_init` JSON, resolve `db_path` to `db`, apply init-function mutations. |
| `apply_sync_delta` | `create_apply_sync_delta_action` | bridge to bot | Apply cross-side state deltas. Only dual-side domains send it; harmless elsewhere. |
| `get_scenario_summary` | `create_get_scenario_summary_action` | bot to bridge | Return `actions` plus `db_hash`; the inline `db` only when `include_db` is requested. |
| `get_context_history` | `create_get_context_history_action` | bot to bridge | Return the LLM message list for `bot_logs_*/llm_context.json`. |

See [RTVI Control Plane](./rtvi-actions.md) for handler semantics and
[RTVI Messages](../reference/rtvi-messages.md) for payload shapes.

## What you may not change

- **The wire protocol.** Use a pipecat WebSocket server transport. Re-implementing protobuf framing
  by hand is out of scope.
- **The six message types and their response shapes.** The bridge sends and parses these literally.
  Renaming `get_scenario_summary`, or returning something other than the `actions` plus `db_hash`
  pair, breaks scoring silently — the run completes with unusable metrics.
- **The tool-registry namespace key.** The bridge passes `scenario.domain` as `tool_domain`; your
  tool-registration callback must accept the same `tool_factory(name, domain=...)` interface that
  `get_schema_tool_for_eval` implements. See [Custom Tools](../features/custom-tools.md).

## Free-choice points

| Area | Freedom |
|---|---|
| Services | Any STT, LLM, TTS, or VAD that emits and consumes pipecat frames. Wrap a non-pipecat service in a `FrameProcessor` subclass. |
| Pipeline shape | Extra processors, reordered stages, parallel branches. Only `transport.input()`, `transport.output()`, and the `RTVIProcessor` have fixed roles. |
| Context | `build_context_and_aggregators()` returns `(context, user_agg, assistant_agg, original_messages)`. Substitute your own as long as the aggregators honor pipecat's frame protocol. |
| Observers | `PipelineWorker` takes an `observers` list; the stock bots attach the repo's `RTVIObserver` subclass, `RTVIAudioLoggerObserver`, and pipecat's `UserBotLatencyObserver`. |
| Reasoning, tool parsers, logits processors | Entirely outside the contract. |

Pass your turn-taking service to `build_context_and_aggregators()` if you use it. In pipecat 1.0+
exactly one component may emit user-speaking frames, and that argument decides which — omitting it
re-derives the answer from `turn_taking.enabled`, which is wrong for a bot that builds its
turn-taking service inline.

## Skeleton

Mirrors the structure of `evaluation/bot_server.py`, trimmed to the required parts.

```python
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frameworks.rtvi import RTVIProcessor

from nemo_voice_agent.evaluation.tools import get_schema_tool_for_eval
from nemo_voice_agent.pipecat.bot_server import run_bot_websocket_server
from nemo_voice_agent.pipecat.processors.frameworks.rtvi import RTVIObserver
from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    SharedStateRef, TaskRef, register_client_message_handlers,
    create_apply_initialization_action, create_apply_sync_delta_action,
    create_get_context_history_action, create_get_scenario_summary_action,
    create_reset_context_action, create_update_system_prompt_action,
)
from nemo_voice_agent.pipecat.services.nemo.builders import build_ws_transport
from nemo_voice_agent.utils import ConfigManager
from nemo_voice_agent.utils.tool_calling import register_schema_tools_to_llm


async def run_custom_bot(host: str, port: int):
    config_manager = ConfigManager(server_base_path=..., server_config_path=...)

    # 1. Your services: stock builders, your own classes, third-party plugins.
    vad_analyzer, stt, llm, tts = ..., ..., ..., ...

    # 2. Transport (required) — applies the protobuf serializer and config sample rates.
    ws_transport = build_ws_transport(config_manager, vad_analyzer, host, port)

    # 3. Context + aggregators. Substitute your own if you need a different shape.
    context, user_agg, assistant_agg, original_messages = ...

    # 4. RTVI processor + handlers (required).
    rtvi = RTVIProcessor()
    task_ref = TaskRef()
    shared_state_ref = SharedStateRef()
    resettable = [stt, tts]
    prompt_args = (task_ref, user_agg, assistant_agg, original_messages, resettable)

    register_client_message_handlers(
        rtvi,
        [
            create_reset_context_action(*prompt_args),
            create_update_system_prompt_action(
                *prompt_args,
                system_role=config_manager.SYSTEM_ROLE,
                system_prompt_suffix=config_manager.SYSTEM_PROMPT_SUFFIX,
                enable_tool_calling=True,
                llm=llm, context=context, rtvi=rtvi,
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

    # 5. Pipeline. The shape is yours; only these positions are fixed.
    pipeline = Pipeline(
        [ws_transport.input(), rtvi, stt, user_agg, llm, tts, ws_transport.output(), assistant_agg]
    )

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True, idle_timeout=None),
        observers=[RTVIObserver(rtvi)],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    # 6. Shared runner: transport handlers, client-ready kickoff, shutdown.
    await run_bot_websocket_server(
        task=task, ws_transport=ws_transport, rtvi=rtvi, task_ref=task_ref,
        talk_first=True, initial_frame_factory=LLMRunFrame,
        on_disconnect_reset_services=resettable,
    )
```

`task_ref.task` and `task_ref.running` are populated by `run_bot_websocket_server()`, so construct the
`TaskRef` before the handlers and hand it over unpopulated. To keep the FastAPI `/connect` endpoint the
browser client uses, wrap the coroutine with `create_fastapi_app()` and `run_bot_with_fastapi()` from
the same module.

## Verify against the eval harness

The eval bridge is the strictest consumer of the contract, so run one scenario through it. All three
commands start from `evaluation/` — `SERVER_CONFIG_PATH` resolves against the current working
directory, not the script directory.

```bash
cd evaluation

# Terminal 1 — your custom agent bot
WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python path/to/your_bot.py
# Terminal 2 — stock user-simulator bot
WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml python bot_server.py
# Terminal 3 — one short scenario
python run_evaluation.py --scenarios restaurant__pizza_pepperoni
```

Then read `eval_results/eval_<timestamp>/restaurant__pizza_pepperoni/`:

| File | What a correct bot produces |
|---|---|
| `metrics.json` | An `is_successful` value that is `true`, `false`, or `"N/A"` — never absent. Absent means `get_scenario_summary` returned an unusable payload. |
| `bridge_log.txt` | No `unknown message type` warnings. One of those names an RTVI handler you forgot to register. |
| `bot_logs_agent/llm_context.json` | The scenario system prompt as the first message, and tool calls on assistant turns. A wrong prompt means `update_system_prompt` did not land; missing tool calls mean your tool-registration callback never reached the LLM service. |
| `final_scenario_db_hash.txt` | A `db_hash:` line, for domains that carry a fixture DB. |

For the full run procedure and artifact reference, see [Evaluation Quickstart](../evaluate/quickstart.md)
and [Reading Results](../evaluate/results.md).

## Non-pipecat agents

Evaluating an agent that is not built on pipecat means re-implementing the transport protocol and the
RTVI control plane yourself. Treat `pipecat.transports.websocket.server` and
`pipecat.serializers.protobuf` as the wire specification. See
[External Agents](../evaluate/external-agents.md) for the harness-side view.
