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

NeMo Labs Voice Agent is a Pipecat pipeline assembled from a small set of builder functions. Customizing it means changing one of three things: which models the builders instantiate, which processors sit in the chain, or the assembly code itself. Pick the lowest tier that does the job.

## Two entrypoints, one pipeline

Everything on this page applies to both shipped entrypoints. They differ only in config directory and in how many RTVI control-plane handlers they register.

| Entrypoint | Script | Config directory | RTVI handlers |
|---|---|---|---|
| Demo server (browser client) | `examples/generic_voice_agent/server/server.py` | `examples/generic_voice_agent/server/server_configs/` | `reset` only |
| Eval bot (agent or user-sim role) | `evaluation/bot_server.py` | `evaluation/server_configs/` | all six |

Both define `run_bot_websocket()`, both import the same `build_*` functions from `nemo_voice_agent/pipecat/services/nemo/builders.py`, and both assemble the same shape:

```text
ws.input -> VAD -> RTVI -> STT -> [Diar] -> [TurnTaking] -> [UserAudioBuffer]
  -> UserAggregator -> LLM -> [LLMTextProcessor] -> TTS -> ws.output -> AssistantAggregator
```

Bracketed stages are dropped when their builder returns `None`: `Diar` when `diar.enabled` is false, `TurnTaking` when turn taking is disabled, `LLMTextProcessor` when `tts.use_text_aggregator` is false, and `UserAudioBuffer` for everything except omni models (`llm.is_omni_model`). See [How It Works](../get-started/architecture.md) for the runtime view.

## Three tiers

| Tier | What you change | Python needed | Use when |
|---|---|---|---|
| 1. YAML swap | A file under `server_configs/` | None | A different model or endpoint that the existing builders already support. |
| 2. Custom frame processor | One `FrameProcessor` subclass, inserted into `pipeline_list` | ~20 lines | You need a transform between two existing stages. |
| 3. New pipeline | Your own `run_bot_websocket()` | ~150 lines | Different services or a different pipeline shape entirely. |

## Tier 1 — swap models via YAML

`ConfigManager` (`nemo_voice_agent/utils/config_manager.py`) reads the file named by the `SERVER_CONFIG_PATH` environment variable and hands each section to the matching builder: `stt:` to `build_stt`, `diar:` to `build_diar`, `llm:` to `build_llm`, `tts:` to `build_tts`, `vad:` to `build_vad_analyzer`, `turn_taking:` to `build_turn_taking`, `transport:` to `build_ws_transport`.

The two config directories are structured differently, and this trips people up:

- **Demo configs** (`default.yaml`, `default_nvidia.yaml`) delegate per-model parameters to a sub-YAML through a `model_config:` field pointing into `llm_configs/`, `tts_configs/`, or `stt_configs/`. The sub-YAML **overrides** the top-level file, not the reverse — a key set in both places takes the sub-YAML's value, so editing it in `default.yaml` alone has no effect.
- **Eval configs** (`agent.yaml`, `agent_nvidia.yaml`, `agent_nvidia_omni.yaml`, `user.yaml`, `user_nvidia.yaml`) are flat and self-contained: no `model_config:` key, and no sub-config directories exist under `evaluation/server_configs/`. Sub-config paths resolve against the entrypoint's own directory, so you cannot point an eval config at a demo sub-YAML by relative path. Edit the eval file in place.

Reasoning mode is a further wrinkle. The automatic swap to a sibling `*_think.yaml` fires only when `server.use_model_registry: true`, the model appears in `model_registry.yaml`, **and** that entry sets `reasoning_supported: true`. An explicit `llm.model_config:` short-circuits the registry lookup, so the shipped `default.yaml` never swaps — point `model_config:` at the `_think.yaml` yourself. The eval configs set `use_model_registry: false` and instead pass the toggle straight to vLLM by interpolating `${llm.enable_reasoning}` into `vllm_generation_params`. Details in [Reasoning Mode](../models/reasoning.md) and [Model Registry](../configure/model-registry.md).

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

Full key reference: [Server Config](../configure/server-config.md) and [Config Schema](../reference/config-schema.md).

## Tier 2 — insert a frame processor

When no builder exposes what you want — typically a transform between two stages — write a `FrameProcessor` subclass and slot it into `pipeline_list` before the `Pipeline(...)` call. Both entrypoints build that list inline, so the edit is a couple of lines.

| Insertion point | Typical use |
|---|---|
| After `ws_transport.input()` | Raw-audio processing: noise suppression, a custom VAD. |
| Between STT and the user aggregator | Transcript transforms: ASR error correction, redaction, language detection. |
| Between the user aggregator and LLM | Context shaping: memory injection, intent classification. |
| Between LLM and TTS | Output transforms: Markdown stripping, profanity filtering. |
| Between TTS and `ws_transport.output()` | Output-audio processing: gain, resampling, earcons. |

Three rules cover most processors: call `super().process_frame(frame, direction)` first, filter with `isinstance` so unrelated frame types pass through untouched, and always finish by pushing the frame onward. Worked examples in [Custom Frame Processors](custom-processor.md).

## Tier 3 — replace the assembly

At this tier you write your own `run_bot_websocket()` and choose every service yourself. Two things stay fixed.

**A Pipecat WebSocket transport.** `build_ws_transport` returns a `SingleClientWebsocketServerTransport` with a `ProtobufFrameSerializer`. That pairing is the wire contract that browser clients and the eval bridge both speak — see [Client Protocol](client-protocol.md).

**An `RTVIProcessor` with handlers registered.** Handler factories live in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`; each returns a `(message_type, handler)` pair that you pass to `register_client_message_handlers`. A demo server needs only `reset`. An eval bot needs all six, because the bridge drives scenario lifecycle over them.

| Message type | Factory | Purpose |
|---|---|---|
| `reset` | `create_reset_context_action` | Clear conversation history and reset stateful services between scenarios. |
| `update_system_prompt` | `create_update_system_prompt_action` | Set the system prompt and register tools from the `tool_domain` registry. |
| `apply_initialization` | `create_apply_initialization_action` | Merge `shared_state_init`, resolve `db_path` to a loaded `db`, apply init-function mutations. |
| `apply_sync_delta` | `create_apply_sync_delta_action` | Apply a cross-side state delta (dual-side domains only; harmless elsewhere). |
| `get_scenario_summary` | `create_get_scenario_summary_action` | Return `actions` plus `db_hash`, and the inline `db` when `include_db` is requested. |
| `get_context_history` | `create_get_context_history_action` | Return the LLM conversation history for the run artifacts. |

An unregistered type produces an `error-response` rather than a hang, so a missing handler shows up immediately in the bridge log. Skeleton code in [Building Your Own Pipeline](custom-pipeline.md); message payloads in [RTVI Control Plane](rtvi-actions.md) and [RTVI Messages](../reference/rtvi-messages.md).

## Verify your customization

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

## Related pages

- [The Builder API](builders.md) — every `build_*` function and the config keys it reads.
- [Writing Your Own Tools](../features/custom-tools.md) — extending the agent's capabilities rather than its pipeline.
- [Evaluating External Agents](../evaluate/external-agents.md) — pointing the harness at a bot you built yourself.
