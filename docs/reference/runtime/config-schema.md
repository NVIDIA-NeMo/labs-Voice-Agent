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

# Server Config Schema

Every key the NeMo Labs Voice Agent server reads out of a YAML server config, with its type, default, and
the code that consumes it. Derived from `examples/generic_voice_agent/server/server_configs/default.yaml`,
`nemo_voice_agent/utils/config_manager.py`, `nemo_voice_agent/pipecat/services/nemo/builders.py`, and the
three service factories (`get_stt_service_from_config`, `get_llm_service_from_config`,
`get_tts_service_from_config`).

For a task-oriented walkthrough of *how* to edit these files, see [Server Configuration](../../build-voice-agents/configure/server-config.md).

**This table is hand-maintained.** There is no schema class to generate it from today — the configs are plain
OmegaConf dicts read with `.get()` at each call site. Treat the named source files as authoritative and open
an issue if you spot drift.

## How a config resolves

1. `ConfigManager` loads the top-level YAML (`SERVER_CONFIG_PATH`, else `server_configs/default.yaml`) and
   resolves all OmegaConf interpolation such as `${llm.temperature}`.
2. For each of `stt`, `llm`, `tts` it picks a sub-config: the **basename** of that block's `model_config`
   key, else the `yaml_id` from `model_registry.yaml` when `server.use_model_registry` is true.
   Only the basename is used — the file is always loaded from
   `<server_base_path>/server_configs/{stt,llm,tts}_configs/`.
3. **The sub-config overwrites the top-level block**, key by key. Setting a key in `default.yaml` that the
   sub-config also sets has no effect. Overrides are logged at INFO.
4. The fully resolved config is logged once at startup (`Server config: ...`) — read that line to confirm
   what actually took effect.

"Required" below means the code reads the key without a fallback, so the server raises if it is missing.

## `server`

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `use_model_registry` | bool | `true` | `ConfigManager` — gates loading `model_registry.yaml` |
| `log_file` | string | `"bot_server.log"` | `resolve_log_file_path` in `builders.py`; `evaluation/bot_server.py` |
| `log_level` | string | `"DEBUG"` | same |
| `create_new_log` | bool | `false` | same — roll the previous log aside on startup |
| `overwrite_existing_log` | bool | `false` | `overwrite_existing_log` in `builders.py` — delete (true) vs. rename (false) |
| `talk_first` | bool | `true` | `evaluation/bot_server.py` — queue the first LLM turn on client-ready |

The example server `examples/generic_voice_agent/server/server.py` calls `setup_logging()` with no arguments
and passes `talk_first=True` literally, so the five logging/`talk_first` keys above take effect only in
`evaluation/bot_server.py` or in your own bot script that wires them through the builders.

## `transport`

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `audio_out_10ms_chunks` | int | required (shipped: `8`) | `build_ws_transport` — larger values reduce TTS glitches |
| `audio_in_sample_rate` | int | `16000` | `build_ws_transport`, `build_vad_analyzer` |
| `audio_out_sample_rate` | int or null | `null` | `build_ws_transport` |
| `audio_in_frame_len_secs` | float | `0.016` | `ConfigManager` — websocket frame length feeding STT buffering |
| `record_audio_data` | bool | `false` | `build_audio_logger` — returns `None` when false |
| `audio_log_dir` | string | `"./audio_logs"` | `build_audio_logger` |

See [Audio Logging](../../build-voice-agents/configure/audio-logging.md) for the on-disk layout.

## `vad`

All four are read without a fallback and passed to pipecat's `VADParams`. VAD is not optional:
`build_vad_analyzer` always returns a `SileroVADAnalyzer`.

| Key | Type | Shipped value | Meaning |
| --- | --- | --- | --- |
| `confidence` | float | `0.6` | speech-vs-nonspeech threshold |
| `start_secs` | float | `0.1` | minimum speech before user-start |
| `stop_secs` | float | `1.2` | minimum silence before user-stop |
| `min_volume` | float | `0.4` | microphone volume gate |

`vad.type` appears in the shipped configs but is not read by any code — Silero is always used.

## `stt`

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `type` | `nemo` \| `nvidia` | required | `get_stt_service_from_config` (asserted) |
| `model` | string | required | both backends |
| `model_config` | path | none | `ConfigManager` — basename selects the file under `stt_configs/` |
| `device` | string | required for `nemo` | `NemoSTTService` |
| `sample_rate` | int | `16000` | both backends |
| `att_context_size` | list | `[70, 1]` | `NeMoSTTInputParams` — left/right streaming context |
| `frame_len_in_secs` | float | `0.08` | `NeMoSTTInputParams` |
| `audio_chunk_size_in_secs` | float | `0.08` | derives `buffer_size` when it is unset |
| `buffer_size` | int | `audio_chunk_size_in_secs // transport.audio_in_frame_len_secs` | `NeMoSTTInputParams` |
| `ignore_eou_eob` | bool | `false` | `NemoSTTService` — ignore end-of-utterance/barge-in tokens |
| `ttfs_p99_latency` | float or null | `null` | `NemoSTTService` |
| `server` | string | `"grpc.nvcf.nvidia.com:443"` | `nvidia` backend only |
| `function_id` | string | see `default_nvidia.yaml` | `nvidia` backend — paired with `model`; change both together |
| `language` | string | `"en-US"` | `nvidia` backend |
| `api_key` | string | `"None"`, overridden by `NVIDIA_API_KEY` | `nvidia` backend |

More detail in [ASR](../../about/core-concepts/speech-pipeline/asr.md).

## `diar`

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `enabled` | bool | required | `ConfigManager`, `build_diar` — returns `None` when false |
| `model` | string | required when enabled | `NemoDiarService` |
| `threshold` | float | required when enabled (shipped: `0.5`) | `NeMoDiarInputParams` — lower is more sensitive |
| `frame_len_in_secs` | float | required when enabled (shipped: `0.08`) | `NeMoDiarInputParams` |

`diar.type` and `diar.device` are present in the shipped configs but not read: `build_diar` reuses
`stt.device` for the diarizer. See [Diarization](../../about/core-concepts/speech-pipeline/diarization.md).

## `turn_taking`

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | `ConfigManager`, `build_turn_taking` — returns `None` when false |
| `backchannel_phrases_path` | path, list, or null | required when enabled | `NeMoTurnTakingService`; a relative path is tried against the CWD then the server base dir, and a missing file raises |
| `max_buffer_size` | int | required when enabled (shipped: `2`) | word count above which a non-backchannel utterance interrupts immediately |
| `bot_stop_delay` | float | required when enabled (shipped: `0.5`) | seconds of server/client audio slack before bot-stop is honored |

Disabling turn-taking also changes turn detection ownership: `build_context_and_aggregators` falls back to
VAD-driven strategies in the user aggregator. See [Turn Taking](../../about/core-concepts/speech-pipeline/turn-taking.md).

## `llm`

Backend-independent keys:

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `type` | `auto` \| `hf` \| `vllm` \| `nvidia` | required | `get_llm_service_from_config`; `auto` probes vLLM and falls back to HF |
| `model` | string | required | `ConfigManager`, all backends |
| `model_config` | path | none | `ConfigManager` — basename selects the file under `llm_configs/`; setting it short-circuits registry lookup |
| `dtype` | string | required for `hf` and `vllm` | HF service; prefixed onto `vllm_server_params` as `--dtype` when absent there |
| `device` | string | required for `hf` | HF service |
| `system_prompt` | string or path | built-in fallback | `ConfigManager` — read as a file if the value is an existing path, else used literally |
| `system_role` | string | `"system"` | `ConfigManager` — use `"user"` for models without a system role |
| `system_prompt_suffix` | string | none | `ConfigManager` — appended to the prompt on a new line |
| `enable_reasoning` | bool | `false` | `ConfigManager` — swaps to the sibling `*_think.yaml` only for registry-resolved models whose entry sets `reasoning_supported: true` |
| `enable_tool_calling` | bool | `false` | `server.py` — gates `register_direct_tools_to_llm` |
| `function_call_timeout_secs` | float or null | `10.0` | all backends; `null` restores pipecat's unbounded wait |
| `inject_dummy_user_message` | bool | `false` | `build_context_and_aggregators` |
| `dummy_user_message` | string | `"Hello."` | `build_context_and_aggregators`, only when the above is true |

HuggingFace backend:

| Key | Type | Default |
| --- | --- | --- |
| `generation_kwargs` | dict | `{}` — passed to `model.generate()` |
| `apply_chat_template_kwargs` | dict | none — passed to `tokenizer.apply_chat_template()`; `tokenize` is forced off |
| `reasoning_budget` | int | `0` |

vLLM backend (see [vLLM](../../build-voice-agents/model-serving/vllm.md)):

| Key | Type | Default |
| --- | --- | --- |
| `api_key` | string | `"None"` |
| `base_url` | string | `"http://localhost:8000/v1"` |
| `organization` | string | `"None"` |
| `project` | string | `"None"` |
| `default_headers` | dict | none |
| `start_vllm_on_init` | bool | `false` |
| `vllm_server_params` | string | none — appended to `vllm serve <model>` |
| `vllm_generation_params` | dict | pipecat defaults — cast to `OpenAILLMService.Settings`; put model-specific fields under its `extra` sub-key |

NVIDIA hosted backend (see [NVIDIA NIM](../../build-voice-agents/model-serving/nvidia-nim.md)):

| Key | Type | Default |
| --- | --- | --- |
| `base_url` | string | `"https://integrate.api.nvidia.com/v1"` |
| `api_key` | string | `"None"`; `NVIDIA_API_KEY` wins, or `NVIDIA_INFERENCE_API_KEY` when `base_url` is `https://inference-api.nvidia.com/v1`. Both endpoints raise if the key is unset |
| `default_headers` | dict | none |
| `nvidia_generation_params` | dict | pipecat defaults — same `Settings` cast and `extra` sub-key as vLLM |

Omni / multimodal keys, read by `server.py` to insert a `UserAudioBuffer` (see [Multimodal](../../about/core-concepts/language-models/multimodal.md)):

| Key | Type | Default |
| --- | --- | --- |
| `is_omni_model` | bool | `false` — the switch that inserts the buffer |
| `pre_cache_duration_secs` | float | `0.3` |
| `use_stt_transcript` | bool | `false` |
| `keep_only_last_audio_turn` | bool | `false` |
| `text_prompt_for_audio` | string | none |
| `text_prompt_for_transcript` | string | none — used only when `use_stt_transcript` is true |

## `tts`

| Key | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `type` | `nemo` \| `nvidia` \| `nemotron` | required | `get_tts_service_from_config` (asserted); only `nvidia` selects the hosted service, the others dispatch on `model` |
| `model` | string | required | local types must be `fastpitch-hifigan`, `magpie`, or `kokoro` — the value selects the service class; for `nvidia` it is the NVCF model name |
| `model_config` | path | none | `ConfigManager` — basename selects the file under `tts_configs/` |
| `device` | string | `"cuda"` | local services |
| `main_model_id` | string | per sub-config | primary checkpoint (FastPitch, Magpie, or Kokoro repo id) |
| `sub_model_id` | string or null | per sub-config | HiFi-GAN checkpoint, or the Kokoro voice |
| `speed` | float | `1.0` | Kokoro only (shipped Kokoro config: `1.25`) |
| `language` | string | `"en"` | Magpie local; `"en-US"` for the `nvidia` service |
| `speaker` | string | `"Sofia"` | Magpie local |
| `apply_TN` | bool | `false` | Magpie local — text normalization |
| `think_tokens` | list or null | `null` | skip synthesis between the tokens, e.g. the `<think>` span |
| `ignore_strings` | list or null | `null` | characters stripped before synthesis (shipped: `*` and `<unk>`) |
| `server` | string | `"grpc.nvcf.nvidia.com:443"` | `nvidia` service |
| `function_id` | string | see `default_nvidia.yaml` | `nvidia` service — paired with `model` |
| `voice_id` | string | `"Magpie-Multilingual.EN-US.Aria"` | `nvidia` service |
| `api_key` | string | `"None"`, overridden by `NVIDIA_API_KEY` | `nvidia` service |
| `max_retries` | int | `2` | `nvidia` service — retry a stream that failed before emitting audio; `0` for single-shot |
| `retry_backoff_secs` | float | `0.25` | `nvidia` service |

Text segmentation keys, read by `build_text_aggregator` for the `LLMTextProcessor` placed upstream of TTS.
Only `extra_separator` appears in the shipped YAML files — every `tts_configs/*.yaml` sets it to
`[',', '\n', '.', '?', '!', ';']`, which the merged TTS config hands to `build_text_aggregator`. The other
three are absent, so their defaults apply unless you add them:

| Key | Type | Default |
| --- | --- | --- |
| `use_text_aggregator` | bool | `true` — false drops the processor from the pipeline entirely |
| `extra_separator` | list or null | `null` in code, but every shipped `tts_configs/*.yaml` sets `[',', '\n', '.', '?', '!', ';']` — extra punctuation to split on for lower first-audio latency |
| `min_sentence_length` | int | `5` |
| `use_legacy_eos_detection` | bool | `false` |

## Related

- [Server Configuration](../../build-voice-agents/configure/server-config.md) — editing workflow and precedence rules
- [Model Registry](../../build-voice-agents/configure/model-registry.md) — how `yaml_id` and `reasoning_supported` resolve
- [Prompts](../../build-voice-agents/configure/prompts.md) — `system_prompt` / `system_prompt_suffix` authoring
- [Environment Variables](environment.md) — `SERVER_CONFIG_PATH`, `NVIDIA_API_KEY`, ports
- [Tool Calling](../../build-voice-agents/tools/tool-calling.md) — what `llm.enable_tool_calling` turns on
