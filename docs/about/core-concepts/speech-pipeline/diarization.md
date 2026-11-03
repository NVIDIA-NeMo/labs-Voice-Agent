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

NeMo Labs Voice Agent uses [streaming Sortformer](https://arxiv.org/abs/2507.18446) models to identify which
speaker is talking during each user turn. In a multi-person conversation, the large language model (LLM)
receives a speaker identity inline with each user utterance that diarization can attribute.
Diarization is on by default in `server_configs/default.yaml`. The shipped model,
`nvidia/Nemotron-3-Diarization`, tells up to eight speakers apart. The four-speaker
`nvidia/diar_streaming_sortformer_4spk-v2.1` model remains available as an alternative.

## How Speaker Diarization Works

The diarization service, `NemoDiarService` in `nemo_voice_agent/pipecat/services/nemo/diar.py`, sits
immediately after STT in the pipeline and before turn-taking. It is a pass-through stage: it never produces
transcripts, only speaker evidence. The turn-taking service turns that evidence into speaker tags.

1. While VAD reports the user is speaking, the service buffers incoming audio frames. Raw frames arrive
   at 16 ms. The service groups them into `frame_len_in_secs` chunks (80 ms by default, or five raw
   frames) and records the VAD segment that each chunk belongs to. A VAD segment runs from a
   `VADUserStartedSpeakingFrame` to the next `VADUserStoppedSpeakingFrame`.
2. Inference runs in a background worker thread (`asyncio.to_thread`), so the audio pipeline is not blocked.
   The model wrapper in `streaming_diar.py` keeps a persistent Sortformer streaming state plus a speaker
   cache across chunks. With the shipped model, `NeMoChunkedStreamingDiarService` runs chunked streaming:
   each model step scores three 80 ms frames with one frame of look-ahead and returns one row for each frame
   that no earlier result covered. A four-speaker model such as `nvidia/diar_streaming_sortformer_4spk-v2.1`
   runs the sliding-window path in `NeMoStreamingDiarService` instead: for each 80 ms chunk, it returns
   speaker probabilities for an output window of six 80 ms frames, which describes roughly the last 400 ms of
   audio. For the difference, refer to [Streaming Modes](#streaming-modes).
3. For each new frame, the service reports at most one speaker: the speaker with the highest probability
   above `diar.threshold`. In chunked streaming, every returned row is a new frame. In the sliding window, only
   the newest frame of the output window is new, because consecutive windows overlap by about 320 ms. A frame
   with no speaker above the threshold produces no evidence.
4. Each 80 ms frame with a reported speaker becomes one `DiarResultFrame`, so the service pushes up to about
   12.5 frames per second of user speech. With the shipped model, these frames arrive in bursts of up to
   three, one burst per chunk. The frame carries the speaker index in `diar_result`. Its `segment_id`
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

The following models are available for local speaker diarization.

| Model | Speakers | Streaming Mode | Notes |
| --- | --- | --- | --- |
| `nvidia/Nemotron-3-Diarization` | 8 | Chunked | Shipped default. High-resolution checkpoint with a rotary position embedding (RoPE) transformer encoder. Requires the NeMo build that `pyproject.toml` pins. |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | 4 | Sliding window | Previous default, with improved accuracy over v2. |
| `nvidia/diar_streaming_sortformer_4spk-v2` | 4 | Sliding window | Previous release. |

Every model is pulled from Hugging Face on first start. A local `.nemo` file path also works:
`build_diarizer` calls `SortformerEncLabelModel.restore_from` when the configured `model` ends in `.nemo`,
and `from_pretrained` otherwise. The service reads the number of speakers from the checkpoint
(`sortformer_modules.n_spk`), so the same code serves four-speaker and eight-speaker models.

`nvidia/Nemotron-3-Diarization` predicts one row for each 10 ms feature frame. The service sets the model's
`output_subsampling_factor` to the encoder's subsampling factor, which is 8 for both shipped models. The model
then pools its output to one row per 80 ms frame, the resolution that the rest of the service works in.

At startup, `build_diarizer` checks that `frame_len_in_secs` equals the model's encoder frame, which is the
preprocessor window stride times the encoder subsampling factor (80 ms for both shipped models). Any other
value stops the server with a `ValueError` that names the expected value.

After the model loads, the service runs about one second of silence through it and then resets its state.
This warm-up adds a few seconds to server startup. Without it, the first forward pass in the process takes
2–4 seconds while CUDA initializes, so the speaker results for the first user turn arrive after that turn
ends and are dropped.

### NeMo Version for Nemotron-3-Diarization

No NeMo release on PyPI can load `nvidia/Nemotron-3-Diarization` yet. NeMo 2.7.3, the version that this
project pinned before, does not export `nemo.collections.asr.modules.TransformerEncoder`. NeMo 3.0.0 exports
that class but does not support the RoPE attention or the 10 ms high-resolution outputs that the checkpoint
uses. `pyproject.toml` therefore installs `nemo-toolkit[asr,tts]` from a pinned commit of the
[NeMo Speech repository](https://github.com/NVIDIA-NeMo/Speech).

An environment that you created before that change still has NeMo 2.7.3. Because
`nvidia/Nemotron-3-Diarization` is the shipped default, the server fails at startup there with the default
configuration and reports `Error locating target 'nemo.collections.asr.modules.TransformerEncoder'`. Run
`bash install.sh` or `uv sync` again to update the environment. Until you can, set `diar.model` to
`nvidia/diar_streaming_sortformer_4spk-v2.1`, as described in
[Switch to a Four-Speaker Model](#switch-to-a-four-speaker-model). For details, refer to the matching entry in
[Troubleshooting](../../../troubleshooting/index.md).

## Streaming Modes

The service runs one of two streaming implementations from `streaming_diar.py`. The `diar.chunk_len` key
selects between them: when it is unset, the service uses the sliding window, and when it is set, the service
uses chunked streaming.

| Aspect | Sliding Window | Chunked |
| --- | --- | --- |
| Class | `NeMoStreamingDiarService` | `NeMoChunkedStreamingDiarService` |
| Selected when | `diar.chunk_len` is unset | `diar.chunk_len` is set |
| Model step | One step for every 80 ms of audio. Each step rescores the most recent frames, and only the newest frame counts as a vote. | One step for every `chunk_len` frames. Each step scores a chunk with up to one frame of left context and `chunk_right_context` frames of look-ahead. Each frame is scored once, so it enters the first-in, first-out (FIFO) queue only once. |
| Vote timing | One vote every 80 ms. | `chunk_len` votes arrive together after the chunk and its look-ahead have arrived. |
| Used by | The four-speaker models | `nvidia/Nemotron-3-Diarization`, the shipped default |

With the shipped Nemotron-3-Diarization settings, `chunk_len: 3` and `chunk_right_context: 1`, the first
burst of three votes arrives after 320 ms of audio in a VAD segment. Later bursts arrive every 240 ms. Model
inference time comes on top of these delays. A vote can therefore lag its audio by up to
`(chunk_len + chunk_right_context) × 80 ms`, which is 320 ms for these settings. A vote that arrives after a
final transcript is emitted counts toward the next transcript in the same VAD segment.

Chunk positions restart at the first chunk of each VAD segment. Audio at the end of a segment that does not
fill a chunk and its look-ahead is never scored. With the shipped settings, that audio is at most 240 ms long
and falls inside the `vad.stop_secs` silence that ends the segment.

A high-resolution checkpoint such as `nvidia/Nemotron-3-Diarization` tends to merge speakers in the
sliding-window path. If you load it without `chunk_len`, the service logs a warning at startup that
recommends chunked streaming. You can also run the four-speaker models in chunked mode by setting
`chunk_len` and `chunk_right_context` in the `diar` block.

## Model Streaming Settings

Each diarization model has a sub-configuration file under
`examples/generic_voice_agent/server/server_configs/diar_configs/` that holds its streaming settings.
`ConfigManager` merges that file into the `diar` block in the same way that it merges the STT, LLM, and TTS
sub-configurations. It merges a file only when `diar.enabled` is true. It selects the file in the following
order:

1. If `diar.model_config` is set, `ConfigManager` uses its file name and loads that file from `diar_configs/`.
   The directory part of the value is ignored.
2. Otherwise, if `server.use_model_registry` is true, `ConfigManager` looks up `diar.model` in the
   `diar_models` section of `examples/generic_voice_agent/server/model_registry.yaml`. A model matches an
   entry by its full ID or by the repository name after the slash. For a `.nemo` path, the file name without
   the extension is matched, so `/models/Nemotron-3-Diarization.nemo` matches `nvidia/Nemotron-3-Diarization`.
3. If no entry matches, `ConfigManager` logs a warning and uses the `diar` block of the server
   configuration as it is.

Keys in the sub-configuration override the same keys in the `diar` block, as for the other components.
`ConfigManager` logs one `Diarization config field ... is overridden` line for each replaced key, followed by
a `Final diarization config:` line. A missing sub-configuration file raises `FileNotFoundError` at startup.

The shipped files contain the following settings:

| File | Models | Settings |
| --- | --- | --- |
| `diar_streaming_sortformer_4spk.yaml` | `nvidia/diar_streaming_sortformer_4spk-v2.1`, `nvidia/diar_streaming_sortformer_4spk-v2` | None. The file holds only comments, so the `diar` block values apply and the service uses the sliding window. |
| `nemotron_3_diarization.yaml` | `nvidia/Nemotron-3-Diarization` | `spkcache_len: 264`, `fifo_len: 264`, `spkcache_update_period: 222`, `chunk_len: 3`, and `chunk_right_context: 1`. These values match the ultra-low latency configuration on the model card, which has a 0.32 second input buffer. |

Because the sub-configuration wins, setting `chunk_len` in the `diar` block of `default.yaml` has no effect
on `nvidia/Nemotron-3-Diarization`. To change its streaming settings, edit `nemotron_3_diarization.yaml`, or
copy it to a new file in `diar_configs/` and point `diar.model_config` at the copy.

The shipped `default.yaml` sets `server.use_model_registry: true`, so the registry selects
`nemotron_3_diarization.yaml` for the default model. If your configuration sets `use_model_registry: false`,
set `diar.model_config: "./server_configs/diar_configs/nemotron_3_diarization.yaml"` to keep the chunked
settings. Use the same key when your local `.nemo` file has a different name than
`Nemotron-3-Diarization.nemo`.

With the shipped configuration, the startup log contains the following line:

```
Diarization service initialized on device: cuda (NeMoChunkedStreamingDiarService, 8 speakers)
```

If the line names `NeMoStreamingDiarService` for `nvidia/Nemotron-3-Diarization`, the chunked settings were
not merged. The service also logs a warning that recommends chunked streaming. Check the
`Final diarization config:` line in the same log.

At load, the service passes the final values to NeMo's streaming-parameter check. The check rejects invalid
values, such as a speaker cache shorter than the model's per-speaker minimum. It also warns when
`spkcache_update_period` falls outside the range from `chunk_len` to `fifo_len + chunk_len`.

## Configuration

Diarization is configured under the top-level `diar:` block of
`examples/generic_voice_agent/server/server_configs/default.yaml`.

```yaml
diar:
  type: nemo
  enabled: true # set to false to disable
  model: "nvidia/Nemotron-3-Diarization"  # 8 speakers; its streaming settings come from diar_configs/ via model_registry.yaml
  # model: "nvidia/diar_streaming_sortformer_4spk-v2.1"  # 4 speakers
  device: "cuda"
  threshold: 0.5  # lower = more sensitive speaker detection
  frame_len_in_secs: 0.08  # must equal the model's encoder frame (80 ms for both models above)
```

The following keys control the local diarization service. The streaming keys are lengths in 80 ms model
frames. The shipped `default.yaml` sets none of them. For the shipped `nvidia/Nemotron-3-Diarization`, their
values come from `nemotron_3_diarization.yaml`, as [Model Streaming Settings](#model-streaming-settings)
describes. For the streaming keys, the table lists the code defaults. They apply when neither the `diar` block
nor the model's sub-configuration sets a key, as for the four-speaker models.

| Key | Default | Effect |
| --- | --- | --- |
| `enabled` | `true` | When `false`, `build_diar` returns `None` and the stage is left out of the pipeline entirely. |
| `type` | `nemo` | Only the NeMo Sortformer backend is implemented today. |
| `model` | `nvidia/Nemotron-3-Diarization` | Hugging Face model ID or path to a local `.nemo` file. |
| `model_config` | Unset | File name of a sub-configuration in `diar_configs/`. When unset, the model registry selects the file. |
| `threshold` | `0.5` | Probability above which a speaker counts as present in a frame. Lower values increase sensitivity. |
| `frame_len_in_secs` | `0.08` | Length of one model frame. It must equal the model's encoder frame (80 ms for both shipped models), and the service raises a `ValueError` at startup otherwise. |
| `spkcache_len` | `188` | Number of frames in the speaker cache. |
| `fifo_len` | `188` | Number of recent frames in the FIFO queue. |
| `spkcache_update_period` | The checkpoint's value | Number of frames that move from the FIFO queue into the speaker cache in each update. |
| `chunk_len` | Unset | Number of frames in each chunk. Setting it selects chunked streaming. When it is unset, the service uses the sliding window. |
| `chunk_right_context` | `7` | Number of look-ahead frames after each chunk. It applies only to chunked streaming. |

Two things to know about `device`:

- The builder passes `config_manager.STT_DEVICE` to the diarization service, so the diarizer follows
  `stt.device`, not `diar.device`. Change `stt.device` to move both services to another device.
- The underlying `DiarizationConfig` device is set from that same value, so ASR and diarization always
  share a device. The shipped value is `cuda`.

Remember the configuration precedence rule described in
[Server configuration](../../../build-voice-agents/configure/server-config.md): a model sub-YAML takes
precedence over matching keys in `default.yaml`. For diarization, the sub-YAML comes from `diar.model_config`
or from the model registry.

## Switch to a Four-Speaker Model

Complete the following steps to replace the shipped eight-speaker model with
`nvidia/diar_streaming_sortformer_4spk-v2.1`. This model also loads with NeMo 2.7.3, so you can use it in an
environment that you have not updated yet.

1. Set `diar.model` in your server configuration. In `default.yaml`, comment out the
   `nvidia/Nemotron-3-Diarization` line and uncomment the line below it:

   ```yaml
   diar:
     type: nemo
     enabled: true
     model: "nvidia/diar_streaming_sortformer_4spk-v2.1"
     device: "cuda"
     threshold: 0.5
     frame_len_in_secs: 0.08
   ```

   The registry maps this model to `diar_streaming_sortformer_4spk.yaml`, which sets no keys, so the service
   uses the sliding window with the code defaults. If you set `chunk_len` in the `diar` block, remove it,
   because it selects chunked streaming for any model. Also remove a `diar.model_config` that names
   `nemotron_3_diarization.yaml`.

2. Start the server and check the startup log for the following line:

   ```
   Diarization service initialized on device: cuda (NeMoStreamingDiarService, 4 speakers)
   ```

To return to the shipped model, set `diar.model` back to `nvidia/Nemotron-3-Diarization`. That model needs the
pinned NeMo build. For details, refer to
[NeMo Version for Nemotron-3-Diarization](#nemo-version-for-nemotron-3-diarization).

## Limits

The following limits come from the service implementation and the shipped diarization configuration.

| Limit | Value | Why |
| --- | --- | --- |
| Speakers per Transcript | 1 | Each final or interim transcript carries at most one tag, chosen by the votes for that utterance. Overlapping speech within one utterance collapses to a single tag. |
| Speakers per Conversation | 8 with the shipped model, 4 with a `4spk` model | The service reads the speaker count from the checkpoint (`sortformer_modules.n_spk`). The shipped `nvidia/Nemotron-3-Diarization` has 8 speakers, and the `4spk` checkpoints have 4. |
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
  a transcript after it is emitted. The shipped model uses chunked streaming, which also waits for each
  chunk's look-ahead, so a vote can lag its audio by up to 320 ms plus inference time. Refer to
  [Streaming Modes](#streaming-modes).
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
- [Model Registry](../../../build-voice-agents/configure/model-registry.md) — how `diar.model` selects a file in `diar_configs/`.
- [Troubleshooting](../../../troubleshooting/index.md) — what to check when speakers are mislabeled or untagged.
