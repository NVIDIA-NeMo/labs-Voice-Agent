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

# Architecture Overview for NeMo Voice Agent

NeMo Labs Voice Agent is a [Pipecat](https://github.com/pipecat-ai/pipecat) pipeline. The
`run_bot_websocket()` function in `examples/generic_voice_agent/server/server.py` reads a YAML configuration,
builds each service, and assembles a linear list of frame processors for a `PipelineWorker`. The remaining
runtime code provides services for that list or configurations that determine which services the pipeline builds.

## Components

Each stage is constructed by a small, independent builder in
`nemo_voice_agent/pipecat/services/nemo/builders.py`. They are thin wrappers over the service
constructors. They read `ConfigManager` properties, which keeps `server.py` close to a declarative
description of the pipeline and limits how often you need to edit it.

| Builder | Return Value |
| --- | --- |
| `build_audio_logger` | `AudioLogger` or `None` (`transport.record_audio_data`) |
| `build_vad_analyzer` | `SileroVADAnalyzer` (never `None`) |
| `build_vad_processor` | `VADProcessor`, or `None` if given no analyzer |
| `build_ws_transport` | `SingleClientWebsocketServerTransport` |
| `build_stt` | STT service |
| `build_diar` | Diarization service or `None` |
| `build_turn_taking` | Turn-taking service or `None` |
| `build_llm` | LLM service for the configured `llm.type` |
| `build_llm_text_processor` | `LLMTextProcessor` or `None` |
| `build_tts` | TTS service |
| `build_context_and_aggregators` | `(context, user_aggregator, assistant_aggregator, original_messages)` |

Because the builders are independent, a custom bot can import only the required builders and construct
custom components inline. For implementation guidance, refer to
[Builders](../build-voice-agents/extend/pipelines/builders.md) and
[Build a custom pipeline](../build-voice-agents/extend/pipelines/custom-pipeline.md).

## Data Flow

Frames flow left to right. Audio arrives from the browser on a WebSocket, and synthesized audio goes
back out on the same socket.

```text
ws.input → VAD → RTVI → STT → [Diar?] → [TurnTaking?] → [UserAudioBuffer?] → UserAggregator
        → LLM → [LLMTextProcessor?] → TTS → ws.output → AssistantAggregator
```

Stages in `[brackets?]` are omitted from the list when their builder returns `None`. The assembly
code appends each one behind an `if x is not None` check, so a disabled stage leaves no placeholder
behind and produces a shorter pipeline.

| Stage | What It Does |
| --- | --- |
| `ws.input` | `SingleClientWebsocketServerTransport` input side. Deserializes protobuf frames from the client into audio frames. |
| VAD | `VADProcessor` wrapping a Silero analyzer. Emits `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`. Pipecat 1.x removed VAD from the input transport, so it runs here as its own processor. |
| RTVI | `RTVIProcessor`. Handles the client control protocol — client-ready handshake, client messages, and server responses. |
| STT | Streaming NeMo ASR. Pushes transcription frames **and** passes the raw audio through (`audio_passthrough=True`), so downstream stages still see audio. |
| Diar | Streaming speaker diarization. Pushes a `DiarResultFrame` whenever the dominant speaker changes. |
| TurnTaking | `NeMoTurnTakingService`. Decides when the user's turn has ended, filters backchannels, and prefixes transcripts with `<speaker_N>` tags when diarization is on. |
| UserAudioBuffer | Buffers the user's raw audio and attaches it to the large language model (LLM) context for audio-input (omni) models. |
| UserAggregator | Turns finalized user speech into a user message on the shared `LLMContext`, then triggers the LLM. |
| LLM | Hugging Face, vLLM, or hosted NVIDIA endpoint, selected by `llm.type`. Streams text back token by token. |
| LLMTextProcessor | Re-segments the LLM's token stream into TTS-sized chunks using a `SimpleSegmentedTextAggregator`. |
| TTS | Speech synthesis (Kokoro by default), producing output audio frames. |
| `ws.output` | Transport output side. Serializes audio back to the client. |
| AssistantAggregator | Appends the assistant's final text to the `LLMContext` so the next turn has history. |

## Optional Pipeline Stages

The following table shows when each conditional stage is present and why it can be omitted.

| Stage | Included When | Notes |
| --- | --- | --- |
| VAD | always | `build_vad_analyzer` always returns a `SileroVADAnalyzer`, so `build_vad_processor` never returns `None` on this path. VAD is not an opt-out. |
| Diar | `diar.enabled: true` | `true` in the shipped `default.yaml`. Requires GPU. |
| TurnTaking | `turn_taking.enabled` is not `false` | Defaults to on when the key is absent. The `*_nvidia.yaml` configurations set it to `false` and let VAD alone drive turn boundaries. |
| UserAudioBuffer | `llm.is_omni_model: true` | Only the `nemotron_nano_v3_omni*` LLM configurations set this. |
| LLMTextProcessor | `tts.use_text_aggregator` is not `false` | Defaults to on. With it off, TTS falls back to plain sentence splitting. |

Turn detection has exactly one owner. `build_context_and_aggregators` inspects whether a turn-taking
service exists: when it does, the user aggregator is configured with `ExternalUserTurnStrategies` so
it stays quiet and lets `NeMoTurnTakingService` emit the user-turn frames. When no service exists, the
aggregator emits those frames directly from the VAD frames. Only one component emits
`UserStartedSpeakingFrame`, so there is no double emission either way.

## Configuration Flow

`ConfigManager` (`nemo_voice_agent/utils/config_manager.py`) loads
`server_configs/default.yaml`, then merges the model-specific YAML named by each component's
`model_config:` field. This order has two important consequences:

- **The model sub-YAML overrides the top-level configuration.** `default.yaml` sets
  `llm.type: auto`, but `llm_configs/nemotron_nano_v3.yaml` sets `type: vllm`, so the effective value
  is `vllm`. A key in `default.yaml` has no effect when the sub-YAML also defines it.
- The shipped default LLM, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, sets
  `start_vllm_on_init: false` — the server does **not** launch vLLM for you. Start it in a separate
  terminal first, as shown in the [Quickstart](../get-started/quickstart.md).

For details, refer to [Configuration](../build-voice-agents/configure/index.md) and
[Server configuration reference](../build-voice-agents/configure/server-config.md).

## Service Interactions

Two servers run concurrently, wired up by `run_bot_with_fastapi`:

| Server | Default Port | Environment Variable | Purpose |
| --- | --- | --- | --- |
| WebSocket (pipeline I/O) | 8765 | `WEBSOCKET_PORT` | Carries audio and RTVI messages. |
| FastAPI | 7860 | `FASTAPI_PORT` | `POST /connect` returns the `ws_url` the client should dial. |

`POST /connect` builds that URL from `SERVER_PUBLIC_HOST` and `WEBSOCKET_SCHEME`. For the complete
list, refer to [Environment variables](../reference/runtime/environment.md).

**One client at a time.** The transport serves a single client. While a client is connected, a second
connection is **rejected**. The server closes the new connection with WebSocket close code `1013` and
reason `Server already has a connected client`. The existing client remains connected. This behavior
reverses the Pipecat 1.0 behavior, in which a new connection disconnected the existing client. Do not
add multi-tenant logic here because this example is single-user by design.

**The pipeline outlives the connection.** On disconnect, the server deliberately does not end the
pipeline task because the WebSocket server lives inside the input transport. Ending the task would
stop the listener while `/connect` continued to return a URL for an inactive port. Instead, the
transport drops its socket reference and accepts the next client into the same running pipeline.
The pipeline also preserves the `LLMContext` across reconnects, so a reconnection resumes the
conversation with its history intact.

To clear history explicitly, the client sends the RTVI `reset` message. Its handler
(`create_reset_context_action`) resets both aggregators back to the original system prompt and resets
the resettable services. Refer to [RTVI actions](../build-voice-agents/extend/protocols/rtvi-actions.md) and
[Client protocol](../build-voice-agents/extend/protocols/client-protocol.md).

**Initial greeting.** When RTVI reports the client is ready and `talk_first` is set, the server queues an
`LLMRunFrame` so the bot greets first instead of waiting for speech.

## Observers

Observers watch frames without participating in the processor chain. Two observers are attached to the
`PipelineWorker`:

- `RTVIObserver` — forwards transcripts, bot-speaking events, and other pipeline events to the client
  as RTVI messages.
- `RTVIAudioLoggerObserver` — writes session audio to disk when audio logging is on. For configuration
  details, refer to [Audio logging](../build-voice-agents/configure/audio-logging.md).

## Related Topics

Use these pages to run, configure, extend, or troubleshoot the architecture described here.

- [Quickstart](../get-started/quickstart.md) — run the default voice agent.
- [Speech recognition](core-concepts/speech-pipeline/asr.md), [Diarization](core-concepts/speech-pipeline/diarization.md),
  [Turn taking](core-concepts/speech-pipeline/turn-taking.md), [Text to speech](core-concepts/speech-pipeline/tts.md),
  [LLM backends](core-concepts/language-models/llm.md), [Multimodal models](core-concepts/language-models/multimodal.md) — per-stage tuning.
- [Tool calling](../build-voice-agents/tools/tool-calling.md) — let the LLM call functions.
- [Troubleshooting](../troubleshooting/index.md) — diagnose runtime and pipeline problems.
