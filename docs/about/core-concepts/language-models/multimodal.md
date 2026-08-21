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

NeMo Labs Voice Agent can send the user's **raw audio** to the LLM instead of (or in addition to) the ASR
transcript, for models that accept audio input. This page covers the shipped Nemotron Omni configs, the
`llm.is_omni_model` switch, and every omni-specific key that
`examples/generic_voice_agent/server/server.py` actually reads.

## Shipped Model and Configurations

The following configurations select the shipped Nemotron Omni model with or without thinking enabled.

| File (under `examples/generic_voice_agent/server/server_configs/llm_configs/`) | Purpose |
| --- | --- |
| `nemotron_nano_v3_omni.yaml` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`, thinking disabled |
| `nemotron_nano_v3_omni_think.yaml` | Same model with `enable_thinking: True` and `thinking_budget: 2048` |

Both set `type: vllm`, `enable_tool_calling: true`, and `start_vllm_on_init: false` — you start vLLM
yourself. Neither is listed in `server/model_registry.yaml`, so you must point `llm.model_config` at the
file you want; `llm.enable_reasoning: true` will **not** auto-swap to the `_think.yaml` variant (the swap
only fires for models resolved through the registry, see `nemo_voice_agent/utils/config_manager.py`).

## Enable an Omni Model

To send raw user audio to the LLM with the shipped Omni configuration, complete the following steps.

1. In `server_configs/default.yaml`, switch the `llm` block to the omni pair (both lines ship commented out):

   ```yaml
   llm:
     model: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
     model_config: "./server_configs/llm_configs/nemotron_nano_v3_omni.yaml"
   ```

   Remember that the sub-YAML overrides `default.yaml`, not the other way round — put omni knobs in the
   sub-YAML (or in a top-level key the sub-YAML does not define).

2. Start vLLM with the flags the config expects (this is the shipped `vllm_server_params` value):

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

See [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md) for the general vLLM story and [Reasoning Mode](reasoning.md) for the
thinking variant.

## How `is_omni_model` Changes the Pipeline

When `llm.is_omni_model` is true, `server.py` constructs a `UserAudioBuffer`
(`nemo_voice_agent/pipecat/services/common.py`) and inserts it into the pipeline **after** the STT,
diarization, and turn-taking stages and **before** the user context aggregator. Nothing else in the
pipeline changes: VAD, ASR, diarization, and turn-taking all still run.

The buffer works per user turn:

- While the user is silent it keeps a ring buffer of the most recent `pre_cache_duration_secs` of input
  audio, so the syllables VAD clipped off at speech onset are not lost.
- While the user is speaking it appends every input audio frame.
- On `UserStoppedSpeakingFrame` it appends one user message to the LLM context containing the whole
  utterance (encoded by pipecat as base64 WAV) plus the text in `text_prompt_for_audio`, then triggers an
  LLM run.

`TranscriptionFrame`s are **consumed** by the buffer and not forwarded downstream, so with omni enabled the
user turns in the LLM context hold audio, not ASR text. ASR output still reaches turn-taking (upstream of
the buffer) and is still what drives barge-in and backchannel handling — see [Turn Taking](../speech-pipeline/turn-taking.md).

The buffer is registered in the server's `resettable` list, so the RTVI `reset` action clears any partially
buffered audio and transcript along with the conversation context.

## Configuration Keys

All of these live under `llm:` and are read in `server.py` only when `is_omni_model` is true.

| Key | Default in `server.py` | Value in the shipped omni configs | Effect |
| --- | --- | --- | --- |
| `is_omni_model` | `false` | `true` | Master switch; inserts `UserAudioBuffer` |
| `pre_cache_duration_secs` | `0.3` | `0.3` | Seconds of pre-speech audio kept in the ring buffer |
| `use_stt_transcript` | `false` | `false` | Also append the ASR transcript to the audio message |
| `text_prompt_for_audio` | `null` | `"Follow instructions or answer questions in the audio."` | Text that accompanies every audio turn |
| `text_prompt_for_transcript` | `null` | `"Here is the pseudo-transcript of the audio for reference:"` | Preamble before the transcript; only used when `use_stt_transcript` is true |
| `keep_only_last_audio_turn` | `false` | `false` | Keep at most one audio turn in the context |

Notes on the individual keys:

- **`pre_cache_duration_secs`** is measured against an assumed 16 ms input frame length — the buffer's
  `raw_audio_frame_len_in_secs` argument is not wired to config by `server.py`. The comment in
  `common.py` recommends tracking `vad.start_secs`; the shipped omni configs use `0.3` while
  `default.yaml` sets `vad.start_secs: 0.1`, which simply buys extra pre-roll.
- **`text_prompt_for_audio` / `text_prompt_for_transcript`** fall back to the module constants
  `DEFAULT_TEXT_PROMPT_FOR_AUDIO` and `DEFAULT_TEXT_PROMPT_FOR_TRANSCRIPT` in `common.py` when null; those
  constants carry the same strings the shipped configs spell out.
- **`use_stt_transcript: true`** turns the audio message text into `text_prompt_for_audio`, then
  `text_prompt_for_transcript`, then the utterance transcript. Use it when the model benefits from a
  textual anchor; leave it false to evaluate the model's own audio understanding.
- **`keep_only_last_audio_turn: true`** rewrites the previous audio turn in the context into a plain-text
  stand-in (built from `text_prompt_for_audio`, `text_prompt_for_transcript`, and that turn's ASR
  transcript) before adding the new one, so only the newest turn carries real audio. Set it to `true` for
  endpoints that accept a single audio turn per request; leave it `false` on vLLM, which accepts multiple
  audio turns. The stand-in text is derived from the ASR transcript regardless of `use_stt_transcript`, so
  keep STT enabled if you use this mode.

Tool calling and reasoning behave exactly as they do for text-only vLLM models: the omni configs enable
tool calling with the `qwen3_coder` tool parser and `nemotron_v3` reasoning parser, and the shipped TTS
configs set `think_tokens` so reasoning spans are not spoken. See
[Tool Calling](../../../build-voice-agents/tools/tool-calling.md).

## Hosted NVIDIA NIM Omni Models

`llm.is_omni_model` is backend-agnostic — the evaluation harness ships
`evaluation/server_configs/agent_nvidia_omni.yaml`, which uses `llm.type: nvidia` with the same omni keys.
That config exists because NVIDIA's hosted inference endpoint does not accept multiple audio turns, which
is the case `keep_only_last_audio_turn` was added for. Note that
`evaluation/bot_server.py` defaults `keep_only_last_audio_turn` to `true` when the key is absent, whereas
`server.py` defaults it to `false`; set it explicitly rather than relying on the fallback. See
[NVIDIA NIM Services](../../../build-voice-agents/model-serving/nvidia-nim.md) and [Evaluation](../../../evaluate/index.md).

## Troubleshooting

Use the following checks to diagnose common multimodal configuration and context problems.

| Symptom | Check |
| --- | --- |
| Model replies as if it heard nothing | Confirm `is_omni_model: true` landed in the *final* merged config — the sub-YAML wins, so a value set only in `default.yaml` may be overwritten. The server logs the final LLM config at startup. |
| Errors about too many audio inputs | Set `keep_only_last_audio_turn: true`, or raise the audio limit in `--limit-mm-per-prompt`. |
| First word of each utterance is clipped | Increase `pre_cache_duration_secs`. |
| Context grows very fast | Each turn carries a WAV payload; lower `llm.max_new_tokens` is irrelevant here — use `keep_only_last_audio_turn: true` or reset the context. |

## Related Topics

Use these pages to configure model serving, reasoning, tool use, and evaluation for multimodal agents.

- [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md)
- [Reasoning Mode](reasoning.md)
- [Tool Calling](../../../build-voice-agents/tools/tool-calling.md)
- [Evaluation Overview](../../../evaluate/index.md)
