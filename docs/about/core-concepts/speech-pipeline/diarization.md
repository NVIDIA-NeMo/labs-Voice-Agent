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

NeMo Labs Voice Agent uses [streaming Sortformer](https://arxiv.org/abs/2507.18446) to identify which
speaker is talking during each user turn. In a multi-person conversation, the large language model (LLM)
receives a speaker identity inline with each user utterance that diarization can attribute.
Diarization is on by default in `server_configs/default.yaml`.

## How Speaker Diarization Works

The diarization service, `NemoDiarService` in `nemo_voice_agent/pipecat/services/nemo/diar.py`, sits
immediately after STT in the pipeline and before turn-taking. It is a pass-through stage: it never produces
transcripts, only speaker evidence. The turn-taking service turns that evidence into speaker tags.

1. While VAD reports the user is speaking, the service buffers incoming audio frames. Raw frames arrive
   at 16 ms. The service groups them into `frame_len_in_secs` chunks (80 ms by default, or five raw
   frames) and records the VAD segment that each chunk belongs to. A VAD segment runs from a
   `VADUserStartedSpeakingFrame` to the next `VADUserStoppedSpeakingFrame`.
2. Inference runs in a background worker thread (`asyncio.to_thread`), so the audio pipeline is not blocked.
   `NeMoStreamingDiarService` in `streaming_diar.py` keeps a persistent Sortformer streaming state plus a
   speaker cache across chunks. For each chunk, it returns speaker probabilities for an output window of six
   80 ms frames, which describes roughly the last 400 ms of audio.
3. For each chunk, the service reports one speaker: the speaker with the highest probability above
   `diar.threshold` in the newest frame of the output window. Consecutive windows overlap by about 320 ms,
   so the newest frame is the only audio that no earlier result covered. A chunk with no speaker above the
   threshold produces no evidence.
4. Each chunk with a reported speaker becomes one `DiarResultFrame`, so the service pushes up to about 12.5
   frames per second of user speech. The frame carries the speaker index in `diar_result`. Its `segment_id`
   field holds the Pipecat `id` of the `VADUserStartedSpeakingFrame` that opened the segment. Each frame is
   one vote, not a label for a whole utterance.
5. `NeMoTurnTakingService` collects the votes and chooses a tag when it emits a final or interim transcript.
   It uses only the evidence for that utterance and applies the following rules in order:
   - It picks the speaker with the most votes since the previous final transcript or suppressed backchannel
     in the same VAD segment. A tie goes to the speaker of the most recent vote.
   - With no votes, it picks the latest speaker heard in the same VAD segment. This rule covers the same
     speaker continuing after an `<EOU>` token flushed the previous utterance.
   - With no evidence in the VAD segment, it emits the transcript without a tag.

A tagged transcript has the form `<speaker_0> text`, with one space after the tag, so the LLM sees the
speaker identity inline in the conversation context.

Speaker evidence never crosses a VAD segment boundary. The following mechanisms enforce this:

- The diarization service drops a result whose VAD segment has ended. Its worker thread also skips queued
  audio from an ended segment, which bounds the backlog when inference runs slower than real time.
- `NeMoTurnTakingService` also drops a `DiarResultFrame` from an ended segment. Pipecat processes VAD frames,
  which are system frames, ahead of queued data frames such as `DiarResultFrame`. A frame that the
  diarization service pushed earlier can therefore arrive after its segment has ended.
- On `VADUserStoppedSpeakingFrame`, the turn-taking service clears all speaker evidence.
- At the first chunk of each VAD segment, the worker thread clears the model's audio and feature buffer, so
  audio from the previous segment stays out of the new output window. The speaker cache is kept, so speaker
  numbering stays stable across turns.

Because each transcript takes its tag from its own evidence, a second speaker who starts talking before the
first speaker's `<EOU>` arrives adds only a few votes to the first speaker's utterance. That transcript keeps
the first speaker's tag as long as the first speaker holds most of its votes. After the `<EOU>`, the second
speaker's next transcript starts with fresh votes. A transcript can also arrive without a tag, for example
after a very short utterance or when inference falls behind real time. The service leaves such a transcript
untagged instead of reusing a speaker from an earlier VAD segment.

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
| Speakers per Transcript | 1 | Each final or interim transcript carries at most one tag, chosen by the votes for that utterance. Overlapping speech within one utterance collapses to a single tag. |
| Speakers per Conversation | 4 | `DiarizationConfig.max_num_speakers` is 4, matching the `4spk` checkpoints. |
| Evidence Boundary | VAD segment | Speaker evidence is cleared when a VAD segment starts or stops, and results from an ended segment are dropped. A transcript never takes its tag from an earlier VAD segment. |

Consecutive transcripts can come from different speakers, including two transcripts in one VAD segment that
an `<EOU>` token separates. This is the supported multi-speaker mode. Splitting a single transcript between
two people is not supported.

## Accuracy Considerations

Account for the following behavior when you decide whether diarization fits your conversation environment.

- The diarization model is not robust to noise. In a noisy room, it can drop or confuse speakers. Use a
  noise-cancelling microphone or a quiet environment.
- It works best when the voices are clearly distinct. The model is more likely to merge or swap
  similar-sounding speakers and accents with limited representation in the training data.
- Tagging depends on inference keeping up with real time. Each 80 ms chunk must be diarized before the
  transcript that it belongs to is emitted. On a CPU or a heavily loaded GPU, inference can fall behind.
  A transcript inside one VAD segment can then carry the wrong tag or no tag. The service does not correct
  a transcript after it is emitted.
- Speaker identity is not stable across a reset. The client's **Reset** button sends the RTVI `reset`
  client message and calls `reset()` on the diarization service. The service drops results that are still
  in flight. Before it processes the next chunk, its worker thread clears the Sortformer streaming state and
  speaker cache, so speaker numbering restarts.

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
- [Turn Taking](turn-taking.md) — collects `DiarResultFrame` votes and tags each transcript when it is emitted.
- [Server configuration](../../../build-voice-agents/configure/server-config.md) — full configuration layout and merge rules.
- [Troubleshooting](../../../troubleshooting/index.md) — what to check when speakers are mislabeled or untagged.
