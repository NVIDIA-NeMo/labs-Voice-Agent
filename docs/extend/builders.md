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

# The Builder API

`nemo_voice_agent/pipecat/services/nemo/builders.py` holds one `build_*` function per pipeline stage.
Each is a thin wrapper around a service constructor that reads the relevant block of the loaded
`ConfigManager`, so a bot script does not repeat the same config plumbing. The builders are
independent — import only the ones you need, and construct anything they do not cover inline.

Two bots in this repo use them: `examples/generic_voice_agent/server/server.py` and
`evaluation/bot_server.py`. Both call the same eleven builders in the same order.

## Every builder

| Builder | Reads | Returns |
| --- | --- | --- |
| `build_audio_logger(config_manager)` | `transport.record_audio_data`, `transport.audio_log_dir` | `AudioLogger`, or `None` when recording is off. Session id is a timestamp. |
| `build_vad_analyzer(config_manager)` | `vad.*` (via `ConfigManager.get_vad_params()`), `transport.audio_in_sample_rate` | `SileroVADAnalyzer`. Never `None`. |
| `build_vad_processor(vad_analyzer)` | nothing — takes the analyzer, not the config | `VADProcessor` wrapping the analyzer, or `None` if you passed `None`. |
| `build_ws_transport(config_manager, vad_analyzer, host, port)` | `transport.audio_in_sample_rate`, `transport.audio_out_sample_rate`, `transport.audio_out_10ms_chunks` | `SingleClientWebsocketServerTransport` with a Protobuf serializer and `session_timeout=None`. |
| `build_stt(config_manager, audio_logger=None)` | the whole `stt` block | An `STTService` from `get_stt_service_from_config`. Never `None`. |
| `build_diar(config_manager, audio_logger=None)` | `diar.enabled`, `diar.model`, `diar.threshold`, `diar.frame_len_in_secs`, and `stt.device` | `NemoDiarService`, or `None` when `diar.enabled` is false. |
| `build_turn_taking(config_manager, audio_logger=None, *, use_diar=None, use_vad=True)` | `turn_taking.enabled`, `turn_taking.max_buffer_size`, `turn_taking.bot_stop_delay`, `turn_taking.backchannel_phrases_path`, `diar.enabled` | `NeMoTurnTakingService`, or `None` when `turn_taking.enabled` is false. |
| `build_tts(config_manager, audio_logger=None)` | the whole `tts` block | A `TTSService` from `get_tts_service_from_config`. Never `None`. |
| `build_llm_text_processor(config_manager)` | `tts.use_text_aggregator`, plus `tts.extra_separator`, `tts.ignore_strings`, `tts.min_sentence_length`, `tts.use_legacy_eos_detection` | `LLMTextProcessor` holding the segmenting text aggregator, or `None` when `tts.use_text_aggregator` is false. |
| `build_llm(config_manager)` | the whole `llm` block | An `LLMService` from `get_llm_service_from_config`. Never `None`. |
| `build_context_and_aggregators(llm, config_manager, turn_taking=None)` | `llm.system_role`, `llm.system_prompt` (plus suffix), `llm.inject_dummy_user_message`, `llm.dummy_user_message`, `turn_taking.enabled` | A 4-tuple: `(context, user_aggregator, assistant_aggregator, original_messages)`. |

Two helpers in the same module are not pipeline stages:

| Helper | Reads | Returns |
| --- | --- | --- |
| `resolve_log_file_path(config_manager, default_name="bot_server.log")` | `server.log_file`, `server.log_level`, `server.create_new_log` | `(log_file, log_level, create_new_log)`, to pair with `setup_rotating_log` from `nemo_voice_agent.utils.misc`. |
| `overwrite_existing_log(config_manager)` | `server.overwrite_existing_log` | `True` to delete a pre-existing log on startup, `False` to rename it. |

Details worth knowing:

- `build_ws_transport` accepts `vad_analyzer` but ignores it. Since Pipecat 1.0 the input transport no
  longer runs VAD; the argument is kept only so existing call sites still work. Pass the analyzer to
  `build_vad_processor` and place the result right after `transport.input()` instead.
- `build_diar` takes the diarization device from `stt.device`, not from `diar.device`.
- `build_diar` and `build_turn_taking` accept `audio_logger` for call-site symmetry; only
  `build_turn_taking` forwards it to the service. `build_stt` and `build_tts` forward it into their
  `get_stt_service_from_config` / `get_tts_service_from_config` factory.
- `build_turn_taking` is annotated as returning `NeMoTurnTakingService` but returns `None` when
  `turn_taking.enabled` is false — treat it as optional like the others.
- `build_llm_text_processor` exists because Pipecat 1.0 dropped `TTSService(text_aggregator=...)`.
  Pipecat silently ignores unknown constructor kwargs, so passing an aggregator to the TTS service
  would fall back to plain sentence splitting with no error.

## Optional stages

`build_audio_logger`, `build_diar`, `build_turn_taking`, and `build_llm_text_processor` return `None`
to mean "omit this stage". `build_vad_processor` returns `None` only if you hand it `None`, and
`build_vad_analyzer` always returns an analyzer — so VAD is never actually dropped on the shipped
path. Assemble the pipeline behind `if x is not None` checks rather than leaving placeholders:

```python
pipeline_list = [ws_transport.input()]
if vad_processor is not None:
    pipeline_list.append(vad_processor)
pipeline_list.extend([rtvi, stt])
if diar is not None:
    pipeline_list.append(diar)
if turn_taking is not None:
    pipeline_list.append(turn_taking)
pipeline_list.extend([user_agg, llm])
if llm_text_processor is not None:
    pipeline_list.append(llm_text_processor)
pipeline_list.extend([tts, ws_transport.output(), assistant_agg])
```

## Call order

Three dependencies constrain the order:

1. `build_vad_analyzer` before `build_vad_processor` and `build_ws_transport`.
2. `build_llm` before `build_context_and_aggregators`.
3. `build_turn_taking` before `build_context_and_aggregators` — **and pass its result through**.

The third one matters most. In Pipecat 1.0+ exactly one component may emit
`UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame`, and the `turn_taking` argument is what
decides which. Given a service, the builder selects `ExternalUserTurnStrategies` so
`NeMoTurnTakingService` owns turn detection and the aggregator stays quiet. Passing `None` is
indistinguishable from omitting the argument — `None` is the parameter's default — so in both cases
the builder re-derives the answer from `turn_taking.enabled`. When that key is false (the
`*_nvidia.yaml` configs) it builds `UserTurnStrategies` from `VADUserTurnStartStrategy` plus
`SpeechTimeoutUserTurnStopStrategy` so the aggregator drives the turn from VAD frames directly; when
it is true or absent (the shipped `default.yaml`) it still selects `ExternalUserTurnStrategies`, and
nothing is left in the pipeline to emit the user-turn frames. That fallback is right for the stock
builders but silently wrong for a bot that constructs its turn-taking service inline — such a bot
must pass the service explicitly.

`original_messages`, the fourth tuple element, is a fresh deep copy of the initial message list. Hand
it to the reset and update-prompt RTVI handler factories — see [RTVI actions](rtvi-actions.md).

## Substituting your own service

Every builder returns a stock Pipecat base type, so swapping one out is a one-line change in the bot
script. Replace the call, keep the object in the same position in `pipeline_list`, and the rest of
the pipeline is unaffected.

```python
# Instead of: stt = build_stt(config_manager, audio_logger)
from my_package.stt import MySTTService          # must subclass pipecat STTService

stt = MySTTService(model="my-model", sample_rate=16000, audio_passthrough=True)
```

The contracts your replacement must honor:

| Slot | Base class to subclass |
| --- | --- |
| STT | `pipecat.services.stt_service.STTService` (`NemoDiarService` also subclasses this) |
| TTS | `pipecat.services.tts_service.TTSService` |
| LLM | `pipecat.services.llm_service.LLMService`; `build_context_and_aggregators` is typed against `BaseOpenAILLMService` |
| Turn taking, VAD, audio buffering | `pipecat.processors.frame_processor.FrameProcessor` |

Two extra behaviors are duck-typed rather than enforced by the base classes:

- **Reset.** The bot passes a `resettable` list to `create_reset_context_action`. The handler calls
  `.reset()` on each entry that defines one and skips `None` entries, so a service without `reset()`
  is simply not reset between scenarios.
- **Tool calling.** A service only exposes tools if it mixes in `ToolCallingMixin` from
  `nemo_voice_agent/utils/tool_calling/mixins.py` and is listed in the `tool_mixins` argument of
  `register_direct_tools_to_llm`. See [Custom tools](../features/custom-tools.md).

If your service needs config of its own, add a block to the YAML and read it from
`config_manager.server_config` rather than threading new arguments through the shipped builders —
`ConfigManager` passes unknown keys through untouched. See
[Server config](../configure/server-config.md).

## Next steps

- [Build a custom pipeline](custom-pipeline.md) — writing a bot script from scratch.
- [Write a custom processor](custom-processor.md) — a new stage rather than a replacement one.
- [How it works](../get-started/architecture.md) — where each stage sits in the frame flow.
