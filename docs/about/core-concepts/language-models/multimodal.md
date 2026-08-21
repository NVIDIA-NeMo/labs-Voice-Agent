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

# Multimodal and Omni Models

NeMo Labs Voice Agent can send the user's **raw audio** to the large language model (LLM), with or without
the automatic speech recognition (ASR) transcript. This behavior supports models that accept audio input.
This page explains the shipped Nemotron Omni configurations, the `llm.is_omni_model` switch, and all
omni-specific keys that
`examples/generic_voice_agent/server/server.py` actually reads.

## Shipped Model and Configurations

The following configurations select the shipped Nemotron Omni model with or without thinking enabled.

| Configuration File | Purpose |
| --- | --- |
| `nemotron_nano_v3_omni.yaml` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`, thinking disabled |
| `nemotron_nano_v3_omni_think.yaml` | Same model with `enable_thinking: True` and `thinking_budget: 2048` |

Both files are under `examples/generic_voice_agent/server/server_configs/llm_configs/`. They set
`type: vllm`, `enable_tool_calling: true`, and `start_vllm_on_init: false`, so you start vLLM separately.
Neither file is listed in `server/model_registry.yaml`, so point `llm.model_config` at the required file.
`llm.enable_reasoning: true` does **not** automatically select the `_think.yaml` variant. That selection
occurs only for models resolved through the registry, as implemented in
`nemo_voice_agent/utils/config_manager.py`.

## Enable an Omni Model

To send raw user audio to the LLM with the shipped Omni configuration, complete the following steps.

1. In `server_configs/default.yaml`, switch the `llm` block to the omni pair (both lines ship commented out):

   ```yaml
   llm:
     model: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
     model_config: "./server_configs/llm_configs/nemotron_nano_v3_omni.yaml"
   ```

   The sub-YAML takes precedence over `default.yaml`. Put omni settings in the sub-YAML or in a top-level
   key that the sub-YAML does not define.

2. Start vLLM with the flags that the configuration expects. This command uses the shipped
   `vllm_server_params` value:

   ```bash
   vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
       --trust-remote-code \
       --tensor-parallel-size 1 \
       --enable-prefix-caching \
       --max-num-seqs 1 \
       --gpu-memory-utilization 0.8 \
       --max-model-len 131072 \
       --video-pruning-rate 0.5 \
       --allowed-local-media-path / \
       --limit-mm-per-prompt '{"video": 999, "image": 999, "audio": 999}' \
       --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
       --enable-auto-tool-choice \
       --tool-call-parser qwen3_coder \
       --reasoning-parser nemotron_v3
   ```

3. Start the server as usual:

   ```bash
   cd examples/generic_voice_agent/server/
   python server.py
   ```

Refer to [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md) for vLLM setup and operation.
Refer to [Reasoning Mode](reasoning.md) for the thinking variant.

## How the Omni Switch Changes the Pipeline

`is_omni_model` is the pipeline switch for direct audio input.

When `llm.is_omni_model` is true, `server.py` constructs a `UserAudioBuffer`
(`nemo_voice_agent/pipecat/services/common.py`) and inserts it into the pipeline **after** the STT,
diarization, and turn-taking stages and **before** the user context aggregator. Nothing else in the
pipeline changes: VAD, ASR, diarization, and turn-taking all still run.

The buffer works per user turn:

- While the user is silent it keeps a ring buffer of the most recent `pre_cache_duration_secs` of input
  audio, so the syllables VAD clipped off at speech onset are not lost.
- While the user is speaking it appends every input audio frame.
- On `UserStoppedSpeakingFrame` it appends one user message to the LLM context containing the whole
  utterance (encoded by Pipecat as base64 WAV) plus the text in `text_prompt_for_audio`, then triggers an
  LLM run.

The buffer **consumes** `TranscriptionFrame` instances instead of forwarding them downstream. With omni
enabled, user turns in the LLM context therefore hold audio rather than ASR text. ASR output still reaches
turn taking upstream of the buffer and drives barge-in and backchannel handling. For details, refer to
[Turn Taking](../speech-pipeline/turn-taking.md).

The buffer is registered in the server's `resettable` list, so the RTVI `reset` action clears any partially
buffered audio and transcript along with the conversation context.

## Configuration Keys

The following keys live under `llm:`. `server.py` reads them only when `is_omni_model` is true:

| Key | Default in `server.py` | Shipped Omni Value | Effect |
| --- | --- | --- | --- |
| `is_omni_model` | `false` | `true` | Master switch that inserts `UserAudioBuffer` |
| `pre_cache_duration_secs` | `0.3` | `0.3` | Seconds of pre-speech audio kept in the ring buffer |
| `use_stt_transcript` | `false` | `false` | Also append the ASR transcript to the audio message |
| `text_prompt_for_audio` | `null` | `"Follow instructions or answer questions in the audio."` | Text that accompanies every audio turn |
| `text_prompt_for_transcript` | `null` | `"Here is the pseudo-transcript of the audio for reference:"` | Preamble before the transcript that is used only when `use_stt_transcript` is true |
| `keep_only_last_audio_turn` | `false` | `false` | Keep at most one audio turn in the context |

Notes on the individual keys:

- **`pre_cache_duration_secs`** is measured against an assumed 16 ms input frame length — the buffer's
  `raw_audio_frame_len_in_secs` argument is not wired to configuration by `server.py`. The comment in
  `common.py` recommends tracking `vad.start_secs`. The shipped omni configurations use `0.3`, while
  `default.yaml` sets `vad.start_secs: 0.1`, which preserves additional pre-roll.
- **`text_prompt_for_audio` and `text_prompt_for_transcript`** fall back to the module constants
  `DEFAULT_TEXT_PROMPT_FOR_AUDIO` and `DEFAULT_TEXT_PROMPT_FOR_TRANSCRIPT` in `common.py` when null. Those
  constants contain the same strings as the shipped configurations.
- **`use_stt_transcript: true`** turns the audio message text into `text_prompt_for_audio`, then
  `text_prompt_for_transcript`, then the utterance transcript. Use it when the model benefits from a
  textual anchor. Leave it false to evaluate the model's own audio understanding.
- **`keep_only_last_audio_turn: true`** rewrites the previous audio turn in the context into a plain-text
  replacement (built from `text_prompt_for_audio`, `text_prompt_for_transcript`, and that turn's ASR
  transcript) before adding the new one, so only the newest turn carries real audio. Set it to `true` for
  endpoints that accept a single audio turn per request. Leave it `false` on vLLM, which accepts multiple
  audio turns. The stand-in text is derived from the ASR transcript regardless of `use_stt_transcript`, so
  keep STT enabled if you use this mode.

Tool calling and reasoning behave as they do for text-only vLLM models. The omni configurations enable
tool calling with the `qwen3_coder` tool parser and `nemotron_v3` reasoning parser, and the shipped TTS
configurations set `think_tokens` so reasoning spans are not spoken. For details, refer to
[Tool Calling](../../../build-voice-agents/tools/tool-calling.md).

## Hosted NVIDIA NIM Omni Models

`llm.is_omni_model` is backend-agnostic. The evaluation harness ships
`evaluation/server_configs/agent_nvidia_omni.yaml`, which uses `llm.type: nvidia` with the same omni keys.
This configuration uses `keep_only_last_audio_turn` because NVIDIA's hosted inference endpoint does not
accept multiple audio turns. In addition,
`evaluation/bot_server.py` defaults `keep_only_last_audio_turn` to `true` when the key is absent, whereas
`server.py` defaults it to `false`. Set the value explicitly rather than relying on the fallback. For related
configuration, refer to [NVIDIA NIM Services](../../../build-voice-agents/model-serving/nvidia-nim.md) and
[Evaluation](../../../evaluate/index.md).

## Troubleshooting

Use the following checks to diagnose common multimodal configuration and context problems.

| Symptom | Check |
| --- | --- |
| Model replies as if it heard nothing | Confirm `is_omni_model: true` is in the *final* merged configuration. The sub-YAML takes precedence, so it can overwrite a value set only in `default.yaml`. The server logs the final LLM configuration at startup. |
| Errors about too many audio inputs | Set `keep_only_last_audio_turn: true`, or raise the audio limit in `--limit-mm-per-prompt`. |
| First word of each utterance is clipped | Increase `pre_cache_duration_secs`. |
| Context grows quickly | Each turn carries a WAV payload. Lowering `llm.max_new_tokens` does not reduce this growth. Use `keep_only_last_audio_turn: true` or reset the context. |

## Related Topics

Use these pages to configure model serving, reasoning, tool use, and evaluation for multimodal agents.

- [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md) — deploy the multimodal model backend.
- [Reasoning Mode](reasoning.md) — select the thinking configuration.
- [Tool Calling](../../../build-voice-agents/tools/tool-calling.md) — configure tools for the vLLM backend.
- [Evaluation Overview](../../../evaluate/index.md) — evaluate multimodal agent behavior.
