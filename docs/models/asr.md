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

NeMo Labs Voice Agent transcribes the user with a **cache-aware streaming FastConformer** model that runs
locally on the GPU. Audio arrives from the WebSocket transport in 16 ms frames, is buffered into 80 ms chunks,
and each chunk is decoded incrementally against a persistent encoder cache — so partial text is available
while the user is still talking, not only after they stop.

The service lives in `nemo_voice_agent/pipecat/services/nemo/stt.py` (`NemoSTTService`), which wraps the model
loading and cache management in `nemo_voice_agent/pipecat/services/nemo/streaming_asr.py`
(`NemoStreamingASRService`). The pipeline stage is constructed by `build_stt` in
`nemo_voice_agent/pipecat/services/nemo/builders.py`.

## Supported models

All shipped models are English. Pick based on whether you need end-of-utterance (EOU) prediction or
punctuation — no current model gives you both.

| Model | EOU / EOB tokens | Punctuation + capitalization | Use when |
| --- | --- | --- | --- |
| `nvidia/parakeet_realtime_eou_120m-v1` (default) | Yes | No | You want the lowest end-of-turn latency. The model itself signals when the user is done, so the agent does not have to wait out the VAD silence timer. |
| `nvidia/nemotron-speech-streaming-en-0.6b` | No | Yes | Transcript quality matters more than turn latency — for example when the transcript is logged, judged, or fed to a punctuation-sensitive LLM prompt. Turn ends fall back to VAD. |
| `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` | No | No | You need to trade latency against accuracy by selecting a different lookahead (see `att_context_size` below). |
| `stt_en_fastconformer_hybrid_large_streaming_80ms` | No | No | Baseline 80 ms-lookahead hybrid model; the one entry present in `server/model_registry.yaml` under `stt_models`. |

`stt.model` is passed straight to NeMo. A plain identifier is resolved with
`ASRModel.from_pretrained()`; a value ending in `.nemo` is treated as a local checkpoint path and loaded with
`ASRModel.restore_from()`.

## The `stt` config block

The default block in `examples/generic_voice_agent/server/server_configs/default.yaml`:

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

`model_config` behaves exactly like the LLM one: only its **basename** is used, resolved against
`server_configs/stt_configs/`, and **the sub-YAML overrides the top-level `stt` block**, not the other way
round (see `_configure_stt` in `nemo_voice_agent/utils/config_manager.py`; each override is logged at startup).
If `model_config` is omitted and `server.use_model_registry` is true, the model name is looked up in
`stt_models` in `server/model_registry.yaml` instead — see [Model registry](../configure/model-registry.md).

The shipped sub-config is small:

```yaml
att_context_size: [70, 1]
frame_len_in_secs: 0.08
audio_chunk_size_in_secs: 0.08
```

## Keys read by the STT builder

These are the keys `get_stt_service_from_config` actually reads for `type: nemo`. Anything else in the block is
ignored.

| Key | Default | Meaning |
| --- | --- | --- |
| `type` | — | `nemo` (local model) or `nvidia` (hosted NIM). Any other value raises an assertion at startup. |
| `model` | — | HuggingFace/NGC model id, or a path to a local `.nemo` file. |
| `device` | — | Torch device string, e.g. `cuda`, `cuda:1`. Put ASR on its own GPU if you have spares. |
| `att_context_size` | `[70, 1]` | Left and right attention context of the streaming encoder. Larger right context means more lookahead: better accuracy, higher latency. The encoder must support switchable lookaheads, otherwise model load fails with `Model does not support multiple lookaheads`. Check the model card for the pairs a given checkpoint was trained with. |
| `audio_chunk_size_in_secs` | `0.08` | Only used to derive `buffer_size` when that key is absent. |
| `raw_audio_frame_len_in_secs` | `0.016` | Length of one inbound transport frame. |
| `buffer_size` | `audio_chunk_size_in_secs // raw_audio_frame_len_in_secs` (5) | Number of inbound frames accumulated before one inference step. 5 × 16 ms = 80 ms, which matches the FastConformer chunk. |
| `frame_len_in_secs` | `0.08` | Carried on the params object for bookkeeping. The audio actually fed per inference step is `buffer_size × raw_audio_frame_len_in_secs`, so change those two if you need a different cadence. |
| `sample_rate` | `16000` | Input sample rate. The shipped models are 16 kHz. |
| `ignore_eou_eob` | `false` | Strip `EOU`/`EOB` tokens from the hypothesis and fall back to VAD for turn ends. See below. |
| `ttfs_p99_latency` | `null` | P99 seconds from end of speech to final transcript, broadcast to downstream turn-stop strategies. Unset by default because the figure is hardware-dependent; measure it for your deployment before setting it. |

Two constructor arguments are fixed by the builder and are not configurable from YAML: the decoding backend
(`legacy`) and `decoder_type` (`rnnt`, which selects the transducer branch on hybrid checkpoints). Audio
passthrough is on, so frames continue downstream to diarization and turn-taking after transcription.

## EOU detection and turn-taking

`parakeet_realtime_eou_120m-v1` emits `EOU` (end of utterance) and `EOB` (end of backchannel) tokens inline in
the hypothesis. `NemoSTTService` keeps a module-level allowlist, `ASR_EOU_MODELS`, and enables EOU-driven
turn-taking only for models on it — the flag is derived from the model name and cannot be forced from config.

The behaviour difference is visible in the frames the service emits:

- **EOU model.** Every transcript is pushed as an `InterimTranscriptionFrame`. The turn-taking service
  downstream watches for the `EOU` / `EOB` suffix, strips it, and promotes the buffered text to a final
  `TranscriptionFrame`. See [Turn taking](./turn-taking.md).
- **Non-EOU model.** The service uses the model's own `is_final` flag to choose between
  `InterimTranscriptionFrame` and `TranscriptionFrame`, and turn ends come from VAD `stop_secs`.

If VAD reports the user stopped speaking while the ASR still believes they are mid-utterance, the service logs
`[EOU missing]` and resets the encoder cache. Occasional lines are normal; a flood of them means the EOU head
is not firing — check the model name, or set `ignore_eou_eob: true` to make VAD authoritative.

Setting `ignore_eou_eob: true` does two things: it strips the special tokens from the emitted text, and it
forces EOU-driven turn-taking off even for an allowlisted model. The evaluation harness uses this to make both
bots' turn ends purely VAD-driven — see `evaluation/server_configs/agent.yaml`:

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

## Hosted ASR

Set `type: nvidia` to call a hosted NVIDIA endpoint instead of loading a local checkpoint. That path uses
pipecat's `NvidiaSTTService` and reads a different set of keys — `model`, `function_id`, `server`, `language`,
`sample_rate`, plus an API key from the `NVIDIA_API_KEY` environment variable (falling back to `api_key` in the
block). `model` and `function_id` are a matched pair addressing one specific deployment; change both or
neither. A worked config ships as `server_configs/default_nvidia.yaml`. See
[NVIDIA NIM](./nvidia-nim.md).

## Verifying your setup

Start the server and grep the log for the resolved configuration — `_configure_stt` prints the merged block,
and the service prints the model it loaded:

```bash
grep -E "Final STT config|Initialized NeMo STT|has_turn_taking" examples/generic_voice_agent/server/bot_server.log
```

A `Setting has_turn_taking to True` line confirms the EOU path is active. With `server.log_level: DEBUG` you also
get one line per non-empty chunk containing the inference time and the running transcript, plus EOU/EOB
latency and probability when the tokens fire — useful for confirming that ASR inference is comfortably faster
than the 80 ms of audio it consumes per step.

## Related pages

- [Turn taking](./turn-taking.md) — how EOU tokens, VAD, and backchannel phrases combine into turn ends.
- [Diarization](./diarization.md) — the optional speaker-tagging stage that sits after ASR.
- [Server configuration](../configure/server-config.md) — config layering and the full file layout.
- [Audio logging](../configure/audio-logging.md) — capturing the per-turn audio and transcripts referenced above.
- [Troubleshooting](../get-started/troubleshooting.md).
