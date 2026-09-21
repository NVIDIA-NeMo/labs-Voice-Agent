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

# Speech Recognition

NeMo Labs Voice Agent transcribes the user with a
**[cache-aware streaming FastConformer](https://arxiv.org/abs/2312.17279)** automatic speech recognition (ASR)
model that runs locally on the GPU. Audio arrives from the WebSocket transport in 16 ms frames and is buffered
into 80 ms chunks. The model decodes each chunk incrementally against a persistent encoder cache, so partial
text is available before the user stops speaking.

The service lives in `nemo_voice_agent/pipecat/services/nemo/stt.py` (`NemoSTTService`), which wraps the model
loading and cache management in `nemo_voice_agent/pipecat/services/nemo/streaming_asr.py`
(`NemoStreamingASRService`). The pipeline stage is constructed by `build_stt` in
`nemo_voice_agent/pipecat/services/nemo/builders.py`.

## Supported Models

All shipped models are English. Choose based on whether you need end-of-utterance (EOU) prediction or
punctuation — no current model gives you both.

| Model | EOU and EOB Tokens | Punctuation and Capitalization | Use When |
| --- | --- | --- | --- |
| `nvidia/parakeet_realtime_eou_120m-v1` (default) | Yes | No | You want the lowest end-of-turn latency. The model itself signals when the user is done, so the agent does not have to wait out the VAD silence timer. |
| `nvidia/nemotron-speech-streaming-en-0.6b` | No | Yes | Transcript quality matters more than turn latency — for example when the transcript is logged, judged, or fed to a punctuation-sensitive LLM prompt. Turn ends fall back to VAD. |
| `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` | No | No | You need to trade latency against accuracy by selecting a different `att_context_size` lookahead. |
| `stt_en_fastconformer_hybrid_large_streaming_80ms` | No | No | Baseline 80 ms-lookahead hybrid model and the one entry present in `server/model_registry.yaml` under `stt_models`. |

NeMo receives `stt.model` without modification. A plain identifier is resolved with
`ASRModel.from_pretrained()`. A value ending in `.nemo` is treated as a local checkpoint path and loaded with
`ASRModel.restore_from()`.

## The STT Configuration Block

The default `stt` block is in `examples/generic_voice_agent/server/server_configs/default.yaml`:

```yaml
stt:
  type: nemo
  model: "nvidia/parakeet_realtime_eou_120m-v1"
  # model: "nvidia/nemotron-speech-streaming-en-0.6b"
  model_config: "./server_configs/stt_configs/nemo_cache_aware_streaming.yaml"
  device: "cuda"
```

To switch models, uncomment the second `model:` line and comment out the first. Keep `model_config` pointing
at `nemo_cache_aware_streaming.yaml` — both models are cache-aware FastConformers and share those parameters.

`model_config` behaves like the large language model (LLM) configuration. The system uses only its
**basename**, resolved against `server_configs/stt_configs/`. **The sub-YAML overrides the top-level `stt`
block**, not the other way round. The `_configure_stt` function in
`nemo_voice_agent/utils/config_manager.py` logs each override at startup.
If `model_config` is omitted and `server.use_model_registry` is true, the model name is looked up in
`stt_models` in `server/model_registry.yaml` instead. For registry behavior, refer to
[Model registry](../../../build-voice-agents/configure/model-registry.md).

The shipped sub-configuration contains three settings:

```yaml
att_context_size: [70, 1]
frame_len_in_secs: 0.08
audio_chunk_size_in_secs: 0.08
```

## Keys Read by the STT Builder

The following table lists the keys that `get_stt_service_from_config` reads for `type: nemo`. The service
ignores other keys in the block.

| Key | Default | Meaning |
| --- | --- | --- |
| `type` | — | `nemo` (local streaming model), `nemo_speechlm` (offline speech language model, or SpeechLM, behind a vLLM endpoint, used by `default_salm.yaml`), or `nvidia` (hosted NIM). Any other value raises an assertion at startup. For the SpeechLM keys, refer to [Offline SpeechLM transcription](#offline-speechlm-transcription). |
| `model` | — | Hugging Face or NGC model ID, or a path to a local `.nemo` file. |
| `device` | — | Torch device string, such as `cuda` or `cuda:1`. Put ASR on its own GPU if you have available capacity. |
| `att_context_size` | `[70, 1]` | Left and right attention context of the streaming encoder. Larger right context means more lookahead: better accuracy, higher latency. The encoder must support switchable lookaheads, otherwise model load fails with `Model does not support multiple lookaheads`. Check the model card for the pairs a given checkpoint was trained with. |
| `audio_chunk_size_in_secs` | `0.08` | Only used to derive `buffer_size` when that key is absent. |
| `raw_audio_frame_len_in_secs` | `0.016` | Length of one inbound transport frame. |
| `buffer_size` | `audio_chunk_size_in_secs // raw_audio_frame_len_in_secs` (5) | Number of inbound frames accumulated before one inference step. 5 × 16 ms = 80 ms, which matches the FastConformer chunk. |
| `frame_len_in_secs` | `0.08` | Carried on the params object for bookkeeping. The audio actually fed per inference step is `buffer_size × raw_audio_frame_len_in_secs`, so change those two if you need a different cadence. |
| `sample_rate` | `16000` | Input sample rate. The shipped models are 16 kHz. |
| `ignore_eou_eob` | `false` | Strip `EOU` and `EOB` tokens from the hypothesis and fall back to VAD for turn ends. |
| `ttfs_p99_latency` | `null` | P99 seconds from end of speech to final transcript, broadcast to downstream turn-stop strategies. The default is unset because the figure is hardware-dependent. Pipecat then substitutes 1.0 seconds and logs `ttfs_p99_latency not set, using default 1.0s`. Measure it for your deployment before setting it. |

Two constructor arguments are fixed by the builder and are not configurable from YAML: the decoding backend
(`legacy`) and `decoder_type` (`rnnt`, which selects the transducer branch on hybrid checkpoints). Audio
passthrough is on, so frames continue downstream to diarization and turn-taking after transcription.

## EOU Detection and Turn Taking

`parakeet_realtime_eou_120m-v1` emits `EOU` (end of utterance) and `EOB` (end of backchannel) tokens inline in
the hypothesis. `NemoSTTService` keeps a module-level allowlist, `ASR_EOU_MODELS`, and enables EOU-driven
turn taking only for models on it. The flag is derived from the model name and cannot be forced by configuration.

The behavior difference is visible in the frames that the service emits:

- **EOU model.** Every transcript is pushed as an `InterimTranscriptionFrame`. The turn-taking service
  downstream watches for the `EOU` / `EOB` suffix, strips it, and promotes the buffered text to a final
  `TranscriptionFrame`. For downstream behavior, refer to [Turn taking](turn-taking.md).
- **Non-EOU model.** The service uses the model's own `is_final` flag to choose between
  `InterimTranscriptionFrame` and `TranscriptionFrame`, and turn ends come from VAD `stop_secs`. Under
  `turn_taking.type: speech_timeout`, `turn_taking.user_speech_timeout` is added on top of that silence
  window. Refer to [Turn taking](turn-taking.md).

If VAD reports the user stopped speaking while the ASR still considers the utterance incomplete, the service
logs `[EOU missing]` and resets the encoder cache. Occasional lines are normal. Frequent lines indicate that
the EOU head is not firing. Check the model name or set `ignore_eou_eob: true` to make VAD authoritative.

Setting `ignore_eou_eob: true` strips the special tokens from emitted text and disables EOU-driven turn taking,
even for an allowlisted model. The evaluation harness uses this setting to make both bots' turn ends purely
VAD-driven, as shown in `evaluation/server_configs/agent.yaml`:

```yaml
stt:
  type: nemo
  model: "nvidia/parakeet_realtime_eou_120m-v1"
  device: "cuda"
  att_context_size: [70, 1]
  frame_len_in_secs: 0.08
  buffer_size: 5
  ignore_eou_eob: true
```

## Offline SpeechLM Transcription

Set `type: nemo_speechlm` to transcribe with an offline SpeechLM that an external vLLM
endpoint serves, instead of a local streaming checkpoint. `NemoSpeechLMSTTService` extends Pipecat's
`SegmentedSTTService`, so it transcribes a complete utterance after VAD reports the end of speech rather than
decoding incrementally. Start the endpoint yourself, and set `model` to the server's `--served-model-name`. An
example ships as `server_configs/default_salm.yaml`, which pairs the backend with
`turn_taking.type: speech_timeout` and a longer `vad.stop_secs` of 1.2 seconds.

The following table lists the keys that `get_stt_service_from_config` reads for `type: nemo_speechlm`. None of
the streaming keys in the previous table apply, and `device` is not read because no model loads locally.

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | — | Served model name. It must match the `--served-model-name` of the vLLM endpoint. |
| `base_url` | `http://localhost:8000/v1` | OpenAI-compatible endpoint of the SpeechLM server. `default_salm.yaml` uses port 5002. |
| `sample_rate` | `16000` | Input sample rate of the audio segment sent for transcription. |
| `generation_kwargs` | `{"chat_template_kwargs": {"enable_thinking": false}}` | Sampling parameters, such as `max_tokens`, `temperature`, and `top_p`. Your values merge over the default and travel to the endpoint in `extra_body`, so vLLM-only keys arrive untouched. Setting your own `chat_template_kwargs` replaces the default wholesale, so repeat `enable_thinking: false`. Without that flag the model answers the prompt as a chatbot instead of transcribing. |
| `max_tokens_per_sec` | `null` | Ceiling on the decode budget, proportional to the duration of the audio. Refer to [Limit the decode budget by audio duration](#limit-the-decode-budget-by-audio-duration). |
| `system_prompt` | `You are a helpful assistant. /no_think` | System message sent with every request. |
| `user_prompt` | `NemoSpeechLMSTTService.DEFAULT_USER_PROMPT` | Instruction sent after the audio. The default asks for a verbatim transcript that preserves named entities, numbers, and acronyms, and that omits accidental repetitions. |
| `api_key` | none | API key for the endpoint. |
| `api_key_env_var` | none | Name of an environment variable to read the API key from when `api_key` is unset. |

### Limit the Decode Budget by Audio Duration

A SpeechLM can fall into a repetition hallucination: from a few seconds of audio it decodes thousands of tokens
of fabricated prose that has nothing to do with what the user said. That text then lands in the agent's context
as a user turn. In one evaluation run, two user turns of roughly 40 words each were transcribed as 9,370 and
9,252 words, while no turn in the equivalent streaming FastConformer runs exceeded about 300 words.

A flat `generation_kwargs.max_tokens` cannot separate that failure from a genuinely long utterance, because
both ask for many tokens. A bound proportional to the duration of the audio can. Set `max_tokens_per_sec` to a
token rate, and each request carries `min(max_tokens, audio_duration_seconds * max_tokens_per_sec)` in place of
the flat value:

```yaml
stt:
  type: nemo_speechlm
  model: "nemotron-transcribe"
  generation_kwargs:
    max_tokens: 2048
  max_tokens_per_sec: 20
```

With that configuration, 3 seconds of audio gets a budget of 60 tokens and 10 seconds gets 200 tokens, while
the configured 2,048 still bounds anything longer than about 102 seconds.

Size the rate against real speech. English runs at roughly 2.5 words per second, and tokens outnumber words by
about 1.3 times, so genuine speech needs only 3 to 4 tokens per second. A value of 20 to 25 leaves a margin of
five to eight times over natural speech, and it cuts a 10-second hallucination from thousands of tokens to a
few hundred.

Four properties of the cap are worth knowing before you set it:

- **The default is off.** `max_tokens_per_sec` is unset in every shipped configuration. An unset value
  preserves the flat `max_tokens` behavior exactly.
- **It only lowers the budget.** The derived value is capped by the configured `max_tokens`, so it never raises
  the budget above the value you set. When `generation_kwargs` omits `max_tokens`, the derived value is sent on
  its own.
- **Short utterances keep a floor.** The derived budget never falls below
  `NemoSpeechLMSTTService.MIN_TOKEN_BUDGET`, which is 16 tokens, so a one-word reply is not clipped.
- **Unparseable audio falls back.** When a segment cannot be read as WAV, the service sends the flat budget
  rather than guess a cap from a length it cannot trust.

With `server.log_level: DEBUG`, the service logs one line each time the cap binds, such as
`Capping max_tokens 2048 -> 200 for 10.00s of audio (max_tokens_per_sec=20)`.

## Hosted ASR

Set `type: nvidia` to call a hosted NVIDIA endpoint instead of loading a local checkpoint. This path uses
Pipecat's `NvidiaSTTService` and reads `model`, `function_id`, `server`, `language`, and `sample_rate`. It also
reads an API key from the `NVIDIA_API_KEY` environment variable and falls back to `api_key` in the block.
`model` and `function_id` are a matched pair that addresses one deployment. Change both or neither. An example
configuration ships as `server_configs/default_nvidia.yaml`. For endpoint setup, refer to
[NVIDIA NIM](../../../build-voice-agents/model-serving/nvidia-nim.md).

## Verify Your Setup

Start the server and grep the log for the resolved configuration — `_configure_stt` prints the merged block,
and the service prints the model it loaded:

```bash
grep -E "Final STT config|Initialized NeMo STT|has_turn_taking" examples/generic_voice_agent/server/bot_server.log
```

A `Setting has_turn_taking to True` line confirms that the EOU path is active. With
`server.log_level: DEBUG`, the log also includes one line per non-empty chunk with the inference time and
running transcript. EOU and EOB latency and probability appear when the tokens fire. Use these measurements
to confirm that ASR inference is faster than the 80 ms of audio consumed per step.

## Related Topics

Use these pages to understand the stages that act on ASR output and to configure other speech services.

- [Turn taking](turn-taking.md) — how EOU tokens, VAD, and backchannel phrases combine into turn ends.
- [Diarization](diarization.md) — the optional speaker-tagging stage that sits after ASR.
- [Server configuration](../../../build-voice-agents/configure/server-config.md) — configuration layering and the full file layout.
- [Audio logging](../../../build-voice-agents/configure/audio-logging.md) — capturing the per-turn audio and transcripts referenced above.
- [Troubleshooting](../../../troubleshooting/index.md) — diagnose recognition and latency problems.
