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

# Speaker Diarization

NeMo Labs Voice Agent uses streaming Sortformer to identify which speaker is talking during each user
turn. In a multi-person conversation, the large language model (LLM) receives a speaker identity with
every utterance.
Diarization is on by default in `server_configs/default.yaml`.

## How Speaker Diarization Works

The diarization service (`nemo_voice_agent/pipecat/services/nemo/diar.py`) sits immediately after STT in
the pipeline and before turn-taking. It is a pass-through stage: it never produces transcripts, only
speaker labels.

1. While VAD reports the user is speaking, the service buffers incoming audio frames. Raw frames arrive
   at 16 ms. The service groups them into `frame_len_in_secs` chunks (80 ms by default, or five raw
   frames) before invoking the model.
2. Inference runs in a background thread (`asyncio.to_thread`), so the audio pipeline is not blocked.
   `NeMoStreamingDiarService` in `streaming_diar.py` keeps a persistent Sortformer streaming state plus a
   speaker cache across chunks, and returns a per-frame speaker-probability array.
3. Probabilities are thresholded at `diar.threshold`, and the speaker holding the most frames in the
   chunk becomes the *dominant speaker*.
4. When the dominant speaker changes, a `DiarResultFrame` is pushed downstream. The turn-taking service
   consumes it and prepends a tag such as `<speaker_0>` to the user's transcript text, so the LLM
   sees the speaker identity inline in the conversation context.
5. On `VADUserStoppedSpeakingFrame`, the current speaker and the audio buffer are cleared, so each user
   turn starts a fresh attribution.

The stock system prompt in `default.yaml` tells the LLM to use speaker tags for speaker identification
without echoing them. If you replace the system prompt, retain that instruction. For prompt configuration,
refer to [Prompts](../../../build-voice-agents/configure/prompts.md).

## Supported Models

The following models are available for local and hosted speaker diarization.

| Model | Notes |
| --- | --- |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | Shipped default with improved accuracy over v2. |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Previous release. |

Both are four-speaker streaming Sortformer checkpoints pulled from Hugging Face on first start. A local
`.nemo` file path also works — `build_diarizer` calls `SortformerEncLabelModel.restore_from` when the
configured `model` ends in `.nemo`, and `from_pretrained` otherwise.

## Configuration

Diarization is configured under the top-level `diar:` block of
`examples/generic_voice_agent/server/server_configs/default.yaml`.

```yaml
diar:
  type: nemo
  enabled: true          # set to false to disable
  model: "nvidia/diar_streaming_sortformer_4spk-v2.1"
  device: "cuda"
  threshold: 0.5         # lower = more sensitive speaker detection
  frame_len_in_secs: 0.08
```

The following keys control the local diarization service:

| Key | Default | Effect |
| --- | --- | --- |
| `enabled` | `true` | When `false`, `build_diar` returns `None` and the stage is left out of the pipeline entirely. |
| `type` | `nemo` | Only the NeMo Sortformer backend is implemented today. |
| `model` | `nvidia/diar_streaming_sortformer_4spk-v2.1` | Hugging Face model ID or path to a local `.nemo` file. |
| `threshold` | `0.5` | Probability above which a speaker counts as present in a frame. Lower values increase sensitivity. |
| `frame_len_in_secs` | `0.08` | Sortformer frame length. Change this value only when you use a different architecture. |

Two things to know about `device`:

- The builder passes `config_manager.STT_DEVICE` to the diarization service, so the diarizer follows
  `stt.device`, not `diar.device`. Change `stt.device` to move both services to another device.
- The underlying `DiarizationConfig` device is set from that same value, so ASR and diarization always
  share a device. The shipped value is `cuda`.

Remember the configuration precedence rule described in
[Server configuration](../../../build-voice-agents/configure/server-config.md): a model sub-YAML
referenced by `model_config:` takes precedence over matching keys in `default.yaml`.

## Limits

The following limits come from the service implementation and the shipped diarization configuration.

| Limit | Value | Why |
| --- | --- | --- |
| Speakers per User Turn | 1 | Only the dominant speaker of a turn is reported. Overlapping speech within one turn collapses to a single tag. |
| Speakers per Conversation | 4 | `DiarizationConfig.max_num_speakers` is 4, matching the `4spk` checkpoints. |
| Turn Boundary | VAD | Speaker state is cleared on `VADUserStoppedSpeakingFrame`, so attribution is per VAD-delimited turn. |

Different turns can come from different speakers. This is the supported multi-speaker mode. Splitting a
single turn between two people is not supported.

## Accuracy Considerations

Account for the following behavior when you decide whether diarization fits your conversation environment.

- The diarization model is not robust to noise. In a noisy room, it can drop or confuse speakers. Use a
  noise-cancelling microphone or a quiet environment.
- It works best when the voices are clearly distinct. The model is more likely to merge or swap
  similar-sounding speakers and accents with limited representation in the training data.
- Speaker identity is not stable across a reset. The client's **Reset** button sends the RTVI `reset`
  client message and calls `reset()` on the diarization service. This action clears the Sortformer
  streaming state and speaker cache, so speaker numbering restarts.

## Disable Diarization

Disable diarization when you have a single known user, limited latency or VRAM capacity, or mislabeled speakers:

```yaml
diar:
  enabled: false
```

With `enabled: false`, the runtime changes as follows:

- `build_diar` returns `None`, and `server.py` omits the stage from the pipeline — no model is loaded and
  no GPU memory is used.
- `config_manager.USE_DIAR` becomes `False`, which is forwarded to the turn-taking service as
  `use_diar=False`, so no speaker tags are added to transcripts.

Several shipped configurations disable diarization:

| Configuration | Reason |
| --- | --- |
| `server_configs/default_nvidia.yaml` | Hosted NIM path. There is no diarization NIM yet, so the local model is disabled. For endpoint details, refer to [NVIDIA NIM](../../../build-voice-agents/model-serving/nvidia-nim.md). |
| `evaluation/server_configs/agent.yaml`, `evaluation/server_configs/user.yaml` | The evaluation harness is a two-bot, single-voice-per-side setup, so speaker tags add nothing. |

## Related Topics

Use these pages to understand the adjacent recognition and turn-taking stages or to change diarization settings.

- [Speech Recognition](asr.md) — the STT stage that feeds diarization.
- [Turn Taking](turn-taking.md) — consumes `DiarResultFrame` and writes the speaker tag into the transcript.
- [Server configuration](../../../build-voice-agents/configure/server-config.md) — full configuration layout and merge rules.
- [Troubleshooting](../../../troubleshooting/index.md) — what to check when speakers are mislabeled.
