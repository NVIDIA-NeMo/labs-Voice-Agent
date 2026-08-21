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

# Extension Overview

NeMo Labs Voice Agent is a Pipecat pipeline assembled from a small set of builder functions. You can change
the models that builders instantiate, the processors in the chain, or the assembly code. Choose the lowest
tier that supports your requirement.

## Workflow Overview

Choose one of three extension tiers, starting with the least invasive option that supports your change.

| Tier | What You Change | Python Needed | Use When |
|---|---|---|---|
| 1. YAML swap | A file under `server_configs/` | None | A different model or endpoint that the existing builders already support. |
| 2. Custom frame processor | One `FrameProcessor` subclass, inserted into `pipeline_list` | ~20 lines | You need a transform between two existing stages. |
| 3. New pipeline | Your own `run_bot_websocket()` | ~150 lines | Different services or a different pipeline shape entirely. |

## Key Concepts

Both shipped entrypoints use the same pipeline shape, builders, and extension boundaries.

### Two Entrypoints, One Pipeline

Everything on this page applies to both shipped entrypoints. They differ only in their config directory and
the number of real-time voice inference (RTVI) control-plane handlers that they register.

| Entrypoint | Script | Config Directory | RTVI Handlers |
|---|---|---|---|
| Demo server (browser client) | `examples/generic_voice_agent/server/server.py` | `examples/generic_voice_agent/server/server_configs/` | `reset` only |
| Eval bot (agent or user-sim role) | `evaluation/bot_server.py` | `evaluation/server_configs/` | all six |

Both define `run_bot_websocket()`, both import the same `build_*` functions from `nemo_voice_agent/pipecat/services/nemo/builders.py`, and both assemble the same shape:

```text
ws.input -> VAD -> RTVI -> STT -> [Diar] -> [TurnTaking] -> [UserAudioBuffer]
  -> UserAggregator -> LLM -> [LLMTextProcessor] -> TTS -> ws.output -> AssistantAggregator
```

Bracketed stages are dropped when their builder returns `None`. These stages include `Diar` when
`diar.enabled` is false and `TurnTaking` when turn taking is disabled. They also include `LLMTextProcessor`
when `tts.use_text_aggregator` is false and `UserAudioBuffer` for non-omni models
(`llm.is_omni_model`). Refer to
[How It Works](../../about/architecture.md) for the runtime view.

## Three Tiers

The following tiers describe what each extension approach changes and where to implement it.

### Tier 1 — Swap Models via YAML

`ConfigManager` (`nemo_voice_agent/utils/config_manager.py`) reads the file named by the `SERVER_CONFIG_PATH`
environment variable. It gives each section to the matching builder: `stt:` to `build_stt`, `diar:` to
`build_diar`, `llm:` to `build_llm`, and `tts:` to `build_tts`. It also maps `vad:` to `build_vad_analyzer`,
`turn_taking:` to `build_turn_taking`, and `transport:` to `build_ws_transport`.

The two config directories use different structures:

- **Demo configs** (`default.yaml`, `default_nvidia.yaml`) delegate per-model parameters to a sub-YAML through
  `model_config:`. The field points into `llm_configs/`, `tts_configs/`, or `stt_configs/`. The sub-YAML
  **overrides** the top-level file, so its value wins when both files set the same key. Editing that key in
  `default.yaml` alone has no effect.
- **Eval configs** (`agent.yaml`, `agent_nvidia.yaml`, `agent_nvidia_omni.yaml`, `user.yaml`,
  `user_nvidia.yaml`) are flat and self-contained. They have no `model_config:` key or sub-config directories
  under `evaluation/server_configs/`. Sub-config paths resolve against the entrypoint's directory, so a
  relative path cannot point an eval config at a demo sub-YAML. Edit the eval file directly.

Reasoning mode adds another condition. The automatic swap to a sibling `*_think.yaml` fires only when
`server.use_model_registry: true`, the model appears in `model_registry.yaml`, **and** that entry sets
`reasoning_supported: true`. An explicit `llm.model_config:` short-circuits the registry lookup, so the
shipped `default.yaml` never swaps. Point `model_config:` at the `_think.yaml` yourself. The eval configs set
`use_model_registry: false` and pass the toggle directly to vLLM by interpolating `${llm.enable_reasoning}`
into `vllm_generation_params`. For details, refer to
[Reasoning Mode](../../about/core-concepts/language-models/reasoning.md) and
[Model Registry](../configure/model-registry.md).

What Tier 1 covers without any Python:

| Change | Key |
|---|---|
| LLM backend (`auto`, `hf`, `vllm`, `nvidia`) | `llm.type` |
| Swap the served model | `llm.model`, `llm.base_url`, `llm.vllm_server_params` |
| Swap TTS voice or engine | `tts.model`, `tts.main_model_id`, `tts.sub_model_id` |
| Swap the streaming ASR model | `stt.model`, `stt.att_context_size` |
| Toggle diarization | `diar.enabled` |
| Backchannel handling | `turn_taking.backchannel_phrases_path` |
| VAD sensitivity and endpointing | `vad.confidence`, `vad.stop_secs` |
| Tool calling | `llm.enable_tool_calling` |

Run a demo server against your edited config:

```bash
cd examples/generic_voice_agent/server
SERVER_CONFIG_PATH=server_configs/default_nvidia.yaml python server.py
```

Run an eval bot against yours. `SERVER_CONFIG_PATH` resolves against the current working directory, so the `cd` is required:

```bash
cd evaluation
WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py
```

Full key reference: [Server Config](../configure/server-config.md) and [Config Schema](../../reference/runtime/config-schema.md).

### Tier 2 — Insert a Frame Processor

When no builder exposes the required transformation between two stages, write a `FrameProcessor` subclass.
Then insert it into `pipeline_list` before the `Pipeline(...)` call. Both entrypoints build that list inline,
so the change is localized.

| Insertion Point | Typical Use |
|---|---|
| After `ws_transport.input()` | Raw-audio processing: noise suppression, a custom VAD. |
| Between STT and the user aggregator | Transcript transforms: ASR error correction, redaction, language detection. |
| Between the user aggregator and LLM | Context shaping: memory injection, intent classification. |
| Between LLM and TTS | Output transforms: Markdown stripping, profanity filtering. |
| Between TTS and `ws_transport.output()` | Output-audio processing: gain, resampling, earcons. |

Three rules cover most processors: call `super().process_frame(frame, direction)` first, filter with
`isinstance` so unrelated frame types pass through untouched, and always finish by pushing the frame onward.
For worked examples, refer to [Custom Frame Processors](pipelines/custom-processor.md).

### Tier 3 — Replace the Assembly

At this tier you write your own `run_bot_websocket()` and choose every service yourself. Two things stay fixed.

**A Pipecat WebSocket transport.** `build_ws_transport` returns a `SingleClientWebsocketServerTransport` with a
`ProtobufFrameSerializer`. That pairing is the wire contract that browser clients and the evaluation bridge
both use. For protocol details, refer to [Client Protocol](protocols/client-protocol.md).

**An `RTVIProcessor` with handlers registered.** Handler factories live in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`; each returns a `(message_type, handler)` pair that you pass to `register_client_message_handlers`. A demo server needs only `reset`. An eval bot needs all six, because the bridge drives scenario lifecycle over them.

| Message Type | Factory | Purpose |
|---|---|---|
| `reset` | `create_reset_context_action` | Clear conversation history and reset stateful services between scenarios. |
| `update_system_prompt` | `create_update_system_prompt_action` | Set the system prompt and register tools from the `tool_domain` registry. |
| `apply_initialization` | `create_apply_initialization_action` | Merge `shared_state_init`, resolve `db_path` to a loaded `db`, apply init-function mutations. |
| `apply_sync_delta` | `create_apply_sync_delta_action` | Apply a cross-side state delta (dual-side domains only; harmless elsewhere). |
| `get_scenario_summary` | `create_get_scenario_summary_action` | Return `actions` plus `db_hash`, and the inline `db` when `include_db` is requested. |
| `get_context_history` | `create_get_context_history_action` | Return the LLM conversation history for the run artifacts. |

An unregistered type produces an `error-response` rather than a hang, so a missing handler shows up immediately in the bridge log. Skeleton code in [Building Your Own Pipeline](pipelines/custom-pipeline.md); message payloads in [RTVI Control Plane](protocols/rtvi-actions.md) and [RTVI Messages](../../reference/runtime/rtvi-messages.md).

## Verify Your Customization

Whichever tier you used, exercise it against one fast scenario before a full run. Three terminals, all started from `evaluation/`:

```bash
cd evaluation
WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml  python bot_server.py
WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py
python run_evaluation.py --scenarios restaurant__pizza_pepperoni
```

Then read the per-scenario output directory:

| Artifact | Check |
|---|---|
| `metrics.json` | Present and populated. Missing means the bot never returned a usable `get_scenario_summary` payload. |
| `bridge_log.txt` | Grep for `unknown message type` — that is a handler you forgot to register. |
| `bot_logs_agent/llm_context.json` | The scenario system prompt should be the first message, and tool calls should appear as `assistant.tool_calls`. |
| `final_scenario_db_hash.txt` | Should carry a hash line; absence means `get_scenario_summary` returned the wrong shape. |

## Related Pages

Use these pages for implementation details and evaluation integration:

- [The Builder API](pipelines/builders.md) — every `build_*` function and the config keys it reads.
- [Writing Your Own Tools](../tools/custom-tools.md) — extending the agent's capabilities rather than its pipeline.
- [Evaluating External Agents](../../evaluate/run-evaluations/external-agents.md) — pointing the harness at a bot you built yourself.
