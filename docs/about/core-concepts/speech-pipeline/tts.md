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

# Text to Speech

NeMo Labs Voice Agent synthesizes the bot's speech with a local TTS model that runs in the server
process. The service classes live in `nemo_voice_agent/pipecat/services/nemo/tts.py`; the factory
`get_tts_service_from_config` in that file dispatches on `tts.model` and is what
`build_tts` (in `nemo_voice_agent/pipecat/services/nemo/builders.py`) calls.

## Available models

Three local models ship with configs under
`examples/generic_voice_agent/server/server_configs/tts_configs/`. The `tts.model` value is the
dispatch key — it must be exactly one of the three strings below, or the factory raises `ValueError`.

| `tts.model` | Sub-config | Weights | Output sample rate | Voices |
| --- | --- | --- | --- | --- |
| `kokoro` (default) | `kokoro_82M.yaml` | `hexgrad/Kokoro-82M` | 24000 Hz | `sub_model_id`, e.g. `af_heart`, `af_bella`, `am_fenrir`, `am_michael` |
| `fastpitch-hifigan` | `nemo_fastpitch-hifigan.yaml` | `nvidia/tts_en_fastpitch` + `nvidia/tts_hifigan` | 22050 Hz | single voice |
| `magpie` | `magpie_tts_multilingual_357m.yaml` | `nvidia/magpie_tts_multilingual_357m` | 22050 Hz | `speaker`: `Sofia`, `Aria`, `John`, `Jason`, `Leo` |

A fourth option, `tts.type: nvidia`, routes to a hosted NVIDIA Riva/NIM endpoint instead of a local
model — see [NVIDIA NIM endpoints](../../../build-voice-agents/model-serving/nvidia-nim.md).

For local models, `main_model_id` is the primary checkpoint and `sub_model_id` is the secondary one:
the Kokoro voice, the HiFi-GAN vocoder for FastPitch, and `null` for Magpie. Both accept a
HuggingFace/NGC identifier or a local `.nemo` path (FastPitch, HiFi-GAN, and Magpie call
`restore_from` when the string ends in `.nemo`).

## Selecting a model

`tts.model` and `tts.model_config` live in the top-level config
(`examples/generic_voice_agent/server/server_configs/default.yaml`). The sub-YAML named by
`model_config` is merged in *after* the top-level block, so any key it defines wins — see
[Server configuration](../../../build-voice-agents/configure/server-config.md).

```yaml
# server_configs/default.yaml
tts:
  type: nemo
  model: "magpie"
  model_config: "./server_configs/tts_configs/magpie_tts_multilingual_357m.yaml"
  device: "cuda"
```

Then restart the server:

```bash
cd examples/generic_voice_agent/server
python server.py
```

`tts.model_config` short-circuits the registry: `_configure_tts` in
`nemo_voice_agent/utils/config_manager.py` only consults
`examples/generic_voice_agent/server/model_registry.yaml` when `model_config` is unset *and*
`server.use_model_registry` is true. The registry's `tts_models` section lists only
`fastpitch-hifigan` and `hexgrad/Kokoro-82M`; setting `model_config` explicitly, as the shipped
default does, is the reliable path. See [Model registry](../../../build-voice-agents/configure/model-registry.md).

## Shared TTS keys

These are read for every local model.

| Key | Where it is read | Default | Effect |
| --- | --- | --- | --- |
| `type` | `get_tts_service_from_config` | `nemo` | One of `nemo`, `nvidia`, `nemotron`. `nvidia` builds the hosted service; the other two use the local dispatch on `model`. |
| `device` | factory | `cuda` | Torch device for the model. |
| `think_tokens` | `BaseNemoTTSService` | `["<think>", "</think>"]` in all three sub-configs | Must be a list of exactly **two** strings (asserted at construction) or `null`. Text between them is never spoken. |
| `ignore_strings` | `BaseNemoTTSService` and the aggregator | `["*", "<unk>"]` in `default.yaml` | Substrings stripped from the text before synthesis. The aggregator strips them too, falling back to `*` alone when the key is unset. |
| `extra_separator` | `build_text_aggregator` | `[',', '\n', '.', '?', '!', ';']` in all three sub-configs | Punctuation that closes a chunk, so speech starts earlier. Setting it to `null` leaves the aggregator with no punctuation marks at all, so chunking then depends on `use_legacy_eos_detection`. |
| `use_text_aggregator` | `build_text_aggregator` | `true` | Set to `false` to drop the `LLMTextProcessor` stage entirely and let pipecat's plain sentence splitting run. |
| `min_sentence_length` | `build_text_aggregator` | `5` | Chunks shorter than this are held back and merged with the next text. |
| `use_legacy_eos_detection` | `build_text_aggregator` | `false` | Fall back to pipecat's `match_endofsentence` when this repo's punctuation search finds no chunk end. |

The last three have no entry in the shipped YAML files — add them under `tts:` if you need them.

### Chunking and latency

Text aggregation is **not** part of the TTS service. Since pipecat 1.0 it belongs to an
`LLMTextProcessor` that `build_llm_text_processor` inserts between the LLM and the TTS service, built
from `SimpleSegmentedTextAggregator` (`nemo_voice_agent/pipecat/utils/text/simple_text_aggregator.py`).
That aggregator is why `extra_separator` includes `,` — it emits a chunk at the last valid comma so
audio starts before the sentence is finished, while its period/comma heuristics avoid splitting on
decimals (`3.14`), bullet numbering (`1.`), abbreviations (`e.g.`, `Dr.`), and times (`p.m.`).

Passing the aggregator to the TTS service instead would be silently ignored — pipecat drops unknown
constructor kwargs — and segmentation would degrade with no error. If you build a custom pipeline,
keep the processor upstream of TTS; see [Builders](../../../build-voice-agents/extend/pipelines/builders.md).

If TTS playback stutters on the client, raise `transport.audio_out_10ms_chunks` in
`default.yaml` (the shipped value is `8`; pipecat's WebSocket default is `4`).

### Reasoning models

`think_tokens` is what keeps a reasoning model's scratchpad out of the audio: `_handle_think_tokens`
tracks the open/close markers across streamed chunks and returns only the text after the closing
token, so the user hears the answer and not the deliberation. Set it to `null` to think out loud.
When the LLM runs on vLLM with a reasoning parser, the reasoning is already stripped upstream — see
[Reasoning mode](../language-models/reasoning.md).

## Kokoro-specific keys

| Key | Default | Effect |
| --- | --- | --- |
| `speed` | `1.25` in `kokoro_82M.yaml` (`1.0` in code) | Speaking rate multiplier. Must be greater than zero. |
| `sub_model_id` | `af_heart` | Voice id passed to the Kokoro pipeline. |

`KokoroTTSService` preloads both English pipelines (`a` = American, `b` = British) at startup so voice
switches at conversation time do not pay a download cost.

Kokoro is also the canonical component-owned tool provider: `setup_tool_calling` registers six direct
functions the LLM can call mid-conversation — `tool_tts_speak_faster` and `tool_tts_speak_slower`
(each a 15% relative change), `tool_tts_set_speed`, `tool_tts_reset_speed`, `tool_tts_set_voice`
(accent `American English` / `British English` plus gender, mapping to `af_heart`, `am_michael`,
`bf_emma`, `bm_george`), and `tool_tts_reset_voice`. They are only registered when
`llm.enable_tool_calling` is true; `server.py` passes the TTS service in the `tool_mixins` list. See
[Tool calling](../../../build-voice-agents/tools/tool-calling.md). FastPitch-HiFiGAN and Magpie register no tools.

An RTVI context reset also resets the service: Kokoro's `reset()` restores the original speed, voice,
accent, and pipeline, so a new session never inherits the previous caller's "speak faster".

## Magpie-specific keys

| Key | Default | Effect |
| --- | --- | --- |
| `language` | `en` | Language code passed to `do_tts`. |
| `speaker` | `Sofia` | One of `Sofia`, `Aria`, `John`, `Jason`, `Leo`; an unknown name raises `ValueError` at construction. |
| `apply_TN` | `false` | Run the model's text normalization before synthesis. |

Magpie warms up at load time by synthesizing a fixed sentence, so the first real turn is not slowed by
lazy CUDA initialization.

## Licensing note for Kokoro

Kokoro's upstream grapheme-to-phoneme stack falls back to espeak-ng via `phonemizer`, both GPL-3.0,
which this repo excludes. `_espeak_gpl_shim.py` installs no-op stand-ins so `kokoro`/`misaki` import
cleanly, and `_g2p_fallback.py` supplies an Apache-2.0 replacement built on `g2p_en` that maps ARPAbet
to misaki's phoneme inventory. Without that fallback, out-of-vocabulary words would be silently
dropped from the audio. Both modules are in `nemo_voice_agent/pipecat/services/nemo/`.

## Recording synthesized audio

`build_tts` accepts the audio logger built by `build_audio_logger`, so bot audio is captured when
recording is enabled. See [Audio logging](../../../build-voice-agents/configure/audio-logging.md).
