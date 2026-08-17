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
turn, so a multi-person conversation reaches the LLM with the speaker attached to every utterance.
Diarization is on by default in `server_configs/default.yaml`.

## How it works

The diarization service (`nemo_voice_agent/pipecat/services/nemo/diar.py`) sits immediately after STT in
the pipeline and before turn-taking. It is a pass-through stage: it never produces transcripts, only
speaker labels.

1. While VAD reports the user is speaking, the service buffers incoming audio frames. Raw frames arrive
   at 16 ms; the service groups them into `frame_len_in_secs` chunks (80 ms by default, i.e. 5 raw
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

The stock system prompt in `default.yaml` already tells the LLM how to interpret speaker tags (use them
to identify the speaker, never echo them back). If you replace the system prompt, carry that instruction
over — see [Prompts](../configure/prompts.md).

## Supported models

| Model | Notes |
| --- | --- |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | Shipped default; improved accuracy over v2. |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Previous release. |

Both are 4-speaker streaming Sortformer checkpoints pulled from Hugging Face on first start. A local
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

| Key | Default | Effect |
| --- | --- | --- |
| `enabled` | `true` | When `false`, `build_diar` returns `None` and the stage is left out of the pipeline entirely. |
| `type` | `nemo` | Only the NeMo Sortformer backend is implemented today. |
| `model` | `nvidia/diar_streaming_sortformer_4spk-v2.1` | Hugging Face model id or path to a local `.nemo` file. |
| `threshold` | `0.5` | Probability above which a speaker counts as present in a frame. Lower values increase sensitivity. |
| `frame_len_in_secs` | `0.08` | Sortformer frame length. Leave alone unless you swap in a different architecture. |

Two things to know about `device`:

- The builder passes `config_manager.STT_DEVICE` to the diarization service, so the diarizer follows
  `stt.device`, not `diar.device`. Change `stt.device` if you need to move both off `cuda`.
- The underlying `DiarizationConfig` device is set from that same value, so ASR and diarization always
  share a device.

Remember the config precedence rule described in [Server configuration](../configure/server-config.md):
a model sub-YAML referenced by `model_config:` overrides the matching keys in `default.yaml`, not the
other way round.

## Limits

| Limit | Value | Why |
| --- | --- | --- |
| Speakers per user turn | 1 | Only the dominant speaker of a turn is reported; overlapping speech within one turn collapses to a single tag. |
| Speakers per conversation | 4 | `DiarizationConfig.max_num_speakers` is 4, matching the `4spk` checkpoints. |
| Turn boundary | VAD | Speaker state is cleared on `VADUserStoppedSpeakingFrame`, so attribution is per VAD-delimited turn. |

Different turns can come from different speakers; that is the supported multi-speaker mode. Splitting a
single turn between two people is not supported.

## Accuracy caveats

- The diarization model is not noise-robust. In a noisy room it can drop or confuse speakers; use a
  noise-cancelling microphone or a quiet environment.
- It works best when the voices are clearly distinct. Similar-sounding speakers, and some accents that
  are thinly represented in the training data, are more likely to be merged or swapped.
- Speaker identity is not stable across a reset. Resetting the conversation (the client's **Reset**
  button, which sends the RTVI `reset` client message) calls `reset()` on the diarization service, which
  clears the Sortformer streaming state and speaker cache. Speaker numbering restarts from scratch.

## Disabling diarization

Turn it off when you have a single known user, when you are latency- or VRAM-constrained, or when the
model is mislabeling speakers:

```yaml
diar:
  enabled: false
```

With `enabled: false`:

- `build_diar` returns `None`, and `server.py` omits the stage from the pipeline — no model is loaded and
  no GPU memory is used.
- `config_manager.USE_DIAR` becomes `False`, which is forwarded to the turn-taking service as
  `use_diar=False`, so no speaker tags are added to transcripts.

Several shipped configs already ship with it off:

| Config | Reason |
| --- | --- |
| `server_configs/default_nvidia.yaml` | Hosted NIM path — see [NVIDIA NIM](./nvidia-nim.md); there is no diarization NIM yet, so the local model is disabled. |
| `evaluation/server_configs/agent.yaml`, `evaluation/server_configs/user.yaml` | The evaluation harness is a two-bot, single-voice-per-side setup, so speaker tags add nothing. |

## Related pages

- [Speech Recognition](./asr.md) — the STT stage that feeds diarization.
- [Turn Taking](./turn-taking.md) — consumes `DiarResultFrame` and writes the speaker tag into the transcript.
- [Server configuration](../configure/server-config.md) — full config layout and merge rules.
- [Troubleshooting](../get-started/troubleshooting.md) — what to check when speakers are mislabeled.
