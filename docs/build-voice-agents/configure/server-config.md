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

# Server Config

The NeMo Labs Voice Agent server is driven entirely by one YAML file. This page walks through the
top-level blocks of `examples/generic_voice_agent/server/server_configs/default.yaml` and names the
builder that consumes each one.

## How the file is loaded

`ConfigManager` (`nemo_voice_agent/utils/config_manager.py`) loads the top-level YAML, then for the
`stt`, `llm`, and `tts` blocks merges in a model-specific sub-config. `examples/generic_voice_agent/server/server.py`
passes `server_base_path=os.path.dirname(__file__)` and reads the config path from the
`SERVER_CONFIG_PATH` environment variable, defaulting to `<server dir>/server_configs/default.yaml`.

```bash
cd examples/generic_voice_agent/server
# Uses default server_configs/default.yaml if SERVER_CONFIG_PATH not specified: 
python server.py
# Or use a specific server cofnig pointed by SERVER_CONFIG_PATH:                                                   
# SERVER_CONFIG_PATH=./server_configs/default_nvidia.yaml python server.py
```

Three rules govern the merge:

- **The sub-config wins.** For every key present in the sub-YAML, `ConfigManager` assigns it over the
  top-level value and logs an `... is overridden from ... to ...` line. Editing a key in `default.yaml`
  that the sub-YAML also sets has no effect.
- **Only the basename of `model_config` matters.** The directory is fixed per component:
  `server_configs/stt_configs/`, `server_configs/llm_configs/`, `server_configs/tts_configs/`.
- **Interpolation in the top-level file is resolved eagerly at load** (`OmegaConf.to_container(..., resolve=True)`),
  while sub-config values are copied over unresolved and resolved lazily against the merged config. That is how
  `nemotron_nano_v3.yaml` can write `temperature: ${llm.temperature}` — it resolves to the `temperature: 0.6` the
  same sub-config contributed, since `default.yaml` defines no `llm.temperature` at all.

If a component omits `model_config` and `server.use_model_registry` is `true`, the model name is looked
up in `model_registry.yaml` instead — see [Model Registry](model-registry.md).

## `server`

The `server` block controls model-registry resolution and the connection lifecycle.

| Key | Effect |
| --- | --- |
| `log_file`, `log_level` | Log destination and level. Consumed through `resolve_log_file_path` in `builders.py`; the example server calls `setup_logging()` with its own defaults (`bot_server.log`, `DEBUG`), which happen to match the shipped values. The evaluation bot server honors them. |
| `create_new_log`, `overwrite_existing_log` | Whether a pre-existing log is rotated aside or deleted on startup. Read by `resolve_log_file_path` / `overwrite_existing_log`; used by `evaluation/bot_server.py`. |
| `use_model_registry` | Load `model_registry.yaml` and auto-resolve sub-configs for components that set no `model_config`. |
| `talk_first` | Whether the bot speaks first on client-ready. Read by `evaluation/bot_server.py`; the example `server.py` passes `talk_first=True` directly and ignores the YAML value. |

## `transport`

Consumed by `build_ws_transport` and `build_audio_logger`. The transport is
`SingleClientWebsocketServerTransport` from pipecat: while a client is connected, a second connection is
closed with code `1013` and the incumbent is kept.

| Key | Effect |
| --- | --- |
| `audio_out_10ms_chunks` | Number of 10 ms chunks per outbound audio write. Larger values reduce TTS glitching at the cost of latency. Required — read as a direct attribute. |
| `record_audio_data` | Turns on the `AudioLogger`; when false, `build_audio_logger` returns `None`. |
| `audio_log_dir` | Directory for recorded WAV/metadata output. See [Audio Logging](audio-logging.md). |
| `audio_in_sample_rate` | Optional. Input rate for both the transport and the VAD analyzer. Defaults to 16000. |
| `audio_out_sample_rate` | Optional. Output rate; defaults to `None`, which lets the TTS service decide. |
| `audio_in_frame_len_secs` | Optional. Inbound frame length used to derive the STT buffer size. Defaults to 0.016. |

## `vad`

Consumed by `build_vad_analyzer`, which always constructs a `SileroVADAnalyzer` — VAD cannot be turned
off on this pipeline, and `type: silero` is descriptive only. `build_vad_processor` wraps the analyzer in
the processor placed right after `transport.input()`.

| Key | Effect |
| --- | --- |
| `confidence` | Speech-probability threshold. |
| `start_secs` | Minimum speech duration before a user-started-speaking frame. |
| `stop_secs` | Minimum silence before a user-stopped-speaking frame. Lower it for snappier turn ends, raise it if the bot cuts users off. |
| `min_volume` | Microphone volume floor. |

All four are read as direct attributes and passed to pipecat's `VADParams`, so all four must be present.

## `stt`

Consumed by `build_stt` via `get_stt_service_from_config`. `type` accepts `nemo` or `nvidia`; anything
else raises. Details in [ASR](../../about/core-concepts/speech-pipeline/asr.md).

| Key | Effect |
| --- | --- |
| `type` | `nemo` for a local NeMo streaming model, `nvidia` for a hosted Riva/NVCF endpoint. |
| `model` | Model identifier, for example `nvidia/parakeet_realtime_eou_120m-v1`. |
| `model_config` | Sub-config under `stt_configs/`, for example `nemo_cache_aware_streaming.yaml`, which supplies `att_context_size`, `frame_len_in_secs`, and `audio_chunk_size_in_secs`. |
| `device` | Torch device for the local model. Also used for diarization (see below). |

For `type: nvidia`, the relevant keys are `language`, `model`, and `function_id` — the model name and
function id address one specific NVCF deployment and must be changed together. `NVIDIA_API_KEY` from the
environment takes precedence over any `api_key` in YAML.

## `diar`

Consumed by `build_diar`, which returns `None` when `enabled` is false, so the diarization stage drops
out of the pipeline entirely. See [Diarization](../../about/core-concepts/speech-pipeline/diarization.md).

| Key | Effect |
| --- | --- |
| `enabled` | Insert the Sortformer diarization stage. |
| `model` | Streaming diarization model, for example `nvidia/diar_streaming_sortformer_4spk-v2.1`. |
| `threshold` | Speaker-presence threshold; lower is more sensitive. |
| `frame_len_in_secs` | Frame length; leave at the Sortformer default unless the architecture changes. |

Two keys in the shipped file are inert: `diar.device` is not read — `build_diar` runs the diarizer on
`stt.device` — and `diar.type` has no consumer.

## `turn_taking`

Consumed by `build_turn_taking`, which returns `None` when `enabled` is false (the key is absent from
`default.yaml` and defaults to true). Whether this service exists also decides who emits user-turn
frames: with turn-taking on, `build_context_and_aggregators` selects `ExternalUserTurnStrategies`;
with it off, the aggregator drives turns from VAD. See [Turn Taking](../../about/core-concepts/speech-pipeline/turn-taking.md).

| Key | Effect |
| --- | --- |
| `enabled` | Optional; omit to keep turn-taking on. |
| `backchannel_phrases_path` | Path to a YAML list (the server ships `backchannel_phrases.yaml`), an inline list of phrases, or `null` to let any speech interrupt. A relative path is tried against the working directory, then the server directory, and raises `FileNotFoundError` naming both if neither exists. |
| `max_buffer_size` | Word count above which speech interrupts the bot immediately, regardless of backchannel matching. |
| `bot_stop_delay` | Seconds of slack between server-side and client-side audio end. |

## `llm`

Consumed by `build_llm` via `get_llm_service_from_config`, and by `build_context_and_aggregators` for the
system message. See [LLM](../../about/core-concepts/language-models/llm.md).

| Key | Effect |
| --- | --- |
| `type` | `auto`, `hf`, `vllm`, or `nvidia`. `auto` probes vLLM support for the model and falls back to HuggingFace. |
| `model` | Model id. The shipped default is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`. |
| `model_config` | Sub-config under `llm_configs/`. The default points at `nemotron_nano_v3.yaml`, which overrides `type` to `vllm` and turns on `enable_tool_calling`. |
| `device` | Torch device for the `hf` backend. |
| `enable_reasoning` | Swap to the sibling `*_think.yaml`, but only when the model was resolved through the registry and its entry sets `reasoning_supported: true`. An explicit `model_config` short-circuits the lookup — point `model_config` at the `_think.yaml` yourself. See [Reasoning](../../about/core-concepts/language-models/reasoning.md). |
| `function_call_timeout_secs` | Seconds to wait for a tool call before giving up. Defaults to 10.0 here; set it to `null` for pipecat's unbounded behavior. |
| `system_prompt` | Literal prompt text, or a path to a `.txt` file, which is read if the value names an existing file. See [Prompts](prompts.md). |

Backend-specific keys arrive from the sub-config: `nemotron_nano_v3.yaml` adds `system_role`,
`system_prompt_suffix`, `enable_tool_calling`, `inject_dummy_user_message`, sampling knobs
(`temperature`, `top_k`, `top_p`, `min_p`, `max_new_tokens`), the vLLM endpoint (`api_key`, `base_url`),
`start_vllm_on_init`, `vllm_server_params`, and the `vllm_generation_params` block sent to the OpenAI
API. Because `start_vllm_on_init` is `false`, `python server.py` alone will not work — start vLLM first
with the flags from `vllm_server_params`:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
  --max-num-seqs 1 --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3
```

Setting `llm.is_omni_model` inserts a `UserAudioBuffer` before the user aggregator, which reads the
`pre_cache_duration_secs`, `use_stt_transcript`, `keep_only_last_audio_turn`, `text_prompt_for_audio`,
and `text_prompt_for_transcript` keys — see [Multimodal](../../about/core-concepts/language-models/multimodal.md).

## `tts`

Consumed by `build_tts` (the synthesis service) and `build_llm_text_processor` (the upstream text
segmenter). `type` accepts `nemo`, `nvidia`, or `nemotron`; for `type: nemo` the supported `model` values
are `kokoro`, `fastpitch-hifigan`, and `magpie`. See [TTS](../../about/core-concepts/speech-pipeline/tts.md).

| Key | Effect |
| --- | --- |
| `type`, `model`, `model_config`, `device` | Backend selection and the sub-config under `tts_configs/`. |
| `ignore_strings` | Tokens stripped before synthesis, for example `*` and `<unk>`. Also passed to the text aggregator. |
| `main_model_id`, `sub_model_id` | From the sub-config: the checkpoint and the voice, for example `hexgrad/Kokoro-82M` with `af_heart`. |
| `speed` | Speaking rate (Kokoro). |
| `think_tokens` | Pair of markers whose enclosed span is skipped, so reasoning is never spoken. Set to `null` to think out loud. |
| `extra_separator` | Extra punctuation used to chunk LLM output into shorter TTS segments, lowering time-to-first-audio. |
| `use_text_aggregator` | Optional. Set false to drop the `LLMTextProcessor` and fall back to plain sentence splitting. `min_sentence_length` and `use_legacy_eos_detection` tune the aggregator. |

## The NIM variant

`server_configs/default_nvidia.yaml` runs the same pipeline against hosted NVIDIA endpoints: `stt.type`,
`llm.type`, and `tts.type` are all `nvidia`, `NVIDIA_API_KEY` must be set, and `diar.enabled` and
`turn_taking.enabled` are both `false` — so the pipeline has no diarization stage and VAD alone drives
turns. Its `llm` block carries `base_url`, `system_role`, `system_prompt_suffix`, and a
`nvidia_generation_params` block instead of the vLLM ones. Because none of its models appear in
`model_registry.yaml` and none set `model_config`, `ConfigManager` logs a not-in-registry warning per
component and every key is taken from this one file. See [NVIDIA NIM](../model-serving/nvidia-nim.md).

## Next steps

Continue with the guide for the component or behavior you need to refine:

- [Model Registry](model-registry.md) — how a model name resolves to a sub-config.
- [Prompts](prompts.md) — system prompt files and suffixes.
- [Config Schema](../../reference/runtime/config-schema.md) — full key reference.
- [Builders](../extend/pipelines/builders.md) — the functions that turn each block into a pipeline stage.
