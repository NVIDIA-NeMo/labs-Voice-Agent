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

# Turn Taking and Backchannels

Turn taking decides when the user's turn ends, when the bot may speak, and which user utterances are
allowed to interrupt the bot. NeMo Labs Voice Agent combines two signals: Silero VAD (energy/speech
activity) and the ASR model's end-of-utterance tokens. Backchannel suppression sits on top so short
acknowledgements like "uh-huh" do not cut the bot off mid-sentence.

## How Turn Detection Works

Three pipeline stages cooperate:

| Stage | Component | Emits |
| --- | --- | --- |
| VAD | `VADProcessor` (pipecat), built by `build_vad_processor` right after `transport.input()` | `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame` |
| ASR | `NemoSTTService` in `nemo_voice_agent/pipecat/services/nemo/stt.py` | transcript text, with `<EOU>` / `<EOB>` tokens appended when the model supports them |
| Turn taking | `NeMoTurnTakingService` in `nemo_voice_agent/pipecat/services/nemo/turn_taking.py` | `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`, `InterruptionFrame`, final/interim transcription frames |

Since pipecat 1.0 exactly one component may emit user-turn frames. `build_context_and_aggregators` in
`nemo_voice_agent/pipecat/services/nemo/builders.py` enforces that: when a turn-taking service exists it
selects `ExternalUserTurnStrategies` so the LLM user aggregator stays quiet, and when it is `None` it falls
back to VAD-driven strategies in the aggregator. See [Builders](../../../build-voice-agents/extend/pipelines/builders.md).

`<EOU>` (end of utterance) means the user finished a turn; `<EOB>` (end of backchannel) means the ASR model
itself flagged the segment as a backchannel. Only ASR models listed in `ASR_EOU_MODELS` in `stt.py` produce
them — the shipped default `nvidia/parakeet_realtime_eou_120m-v1` does. Set `stt.ignore_eou_eob: true` to
strip both tokens and fall back to VAD-only boundaries. See [ASR](asr.md).

## VAD Settings

Configured under the top-level `vad:` block; `ConfigManager.get_vad_params()` maps them onto pipecat's
`VADParams`. Shipped values in `server_configs/default.yaml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `vad.type` | `silero` | Analyzer type. |
| `vad.confidence` | `0.6` | Speech-probability threshold. |
| `vad.start_secs` | `0.1` | Minimum speech duration before a start-speaking event fires. |
| `vad.min_volume` | `0.4` | Minimum audio volume; frames below this are never speech. |
| `vad.stop_secs` | `1.2` | Silence required before a stop-speaking event fires. |

`vad.stop_secs` is the main latency knob when the ASR model does not supply `<EOU>`: lower values respond
faster but cut users off in mid-sentence pauses. With an EOU-capable ASR model the end of turn usually
arrives from `<EOU>` first, and VAD stop acts as the fallback — the log line
`[EOU missing] STT failed to detect end of utterance before VAD detected user stopped speaking` in
`bot_server.log` tells you the fallback fired.

The analyzer runs at `transport.audio_in_sample_rate` (falling back to the config-wide sample rate). VAD is
not optional on the websocket path: `build_vad_analyzer` always returns an analyzer.

## Turn-Taking Settings

The following settings control turn finalization, backchannel filtering, and response timing.

| Key | Default | Meaning |
| --- | --- | --- |
| `turn_taking.enabled` | `true` when the key is absent | `false` makes `build_turn_taking` return `None`, dropping the processor from the pipeline. |
| `turn_taking.backchannel_phrases_path` | `"./backchannel_phrases.yaml"` | YAML file path, inline list, or `null`. See below. |
| `turn_taking.max_buffer_size` | `2` | Number of completed words that may accumulate mid-utterance before the bot is interrupted and an interim transcript is pushed downstream. Lower interrupts sooner. |
| `turn_taking.bot_stop_delay` | `0.5` | Seconds to keep treating the bot as "still speaking" after `BotStoppedSpeakingFrame`, covering audio still buffered on the client. `0` flips the flag immediately. |

`bot_stop_delay` matters for backchannels: suppression only applies while the bot is considered to be
speaking, so a value that is too small lets a trailing "okay" from the user start a new LLM turn.

## Backchannel Suppression

The config key is **`turn_taking.backchannel_phrases_path`** (not `backchannel_phrases`). Despite the name
it accepts three forms, resolved by `ConfigManager._resolve_backchannel_phrases` and validated in
`NeMoTurnTakingService._load_backchannel_phrases`:

```yaml
turn_taking:
  # 1. Path to a YAML file containing a flat list of phrases (shipped default).
  backchannel_phrases_path: "./backchannel_phrases.yaml"

  # 2. Inline list — no file needed.
  # backchannel_phrases_path: ["uh huh", "mhmm", "right", "okay"]

  # 3. null / empty — disable backchanneling entirely; any speech interrupts the bot.
  # backchannel_phrases_path: null
```

The shipped file is `examples/generic_voice_agent/server/backchannel_phrases.yaml` (78 phrases:
`uh huh`, `mhmm`, `okay`, `right`, `sure`, `i see`, and so on). A relative path is tried against the current
working directory first and then against the server base directory, so the default works whether you launch
from the repo root or from the server directory. A path that exists at neither location raises
`FileNotFoundError` naming both attempts, rather than failing later inside the service.

Matching rules, from `clean_text` / `is_backchannel` in `turn_taking.py`:

- Comparison is case-insensitive and whitespace-normalized.
- A trailing `<EOU>` or `<EOB>` token and any leading `<speaker_N>` tag are stripped first.
- Every character except `a`-`z`, apostrophes, and whitespace is **deleted, not replaced by a space** — so
  `uh-huh` normalizes to `uhhuh`, which is why the shipped file lists both `uh huh` and `uh-huh`.
- The match is exact against the normalized set; there is no substring or fuzzy matching.
- Only English is supported — `clean_text` raises `ValueError` for any other `Language`.

A phrase is suppressed **only while the bot is speaking**. Then the text is pushed *upstream* as a
transcription frame wrapped in parentheses, for example `(uh huh)`, so it reaches the client transcript and
the audio logger without entering the LLM context or triggering an interruption. When the bot is silent the
same phrase is treated as ordinary user speech and starts a normal turn.

Anything longer than the phrase list still interrupts: once `max_buffer_size` completed words accumulate,
`NeMoTurnTakingService` emits `UserStartedSpeakingFrame` plus `InterruptionFrame` without waiting for
`<EOU>`.

To confirm the list loaded, grep the server log:

```bash
grep -E "backchannel phrases|Backchannel detected" examples/generic_voice_agent/server/bot_server.log
```

`Loading backchannel phrases from file:` / `Using backchannel phrases from list:` appears at startup, and
`Backchannel detected:` appears each time a phrase is suppressed. When the audio logger is enabled the
suppressed segment is tagged `is_backchannel` — see [Audio logging](../../../build-voice-agents/configure/audio-logging.md).

## Disable Turn Taking

To let VAD drive turn boundaries without the turn-taking service, update the configuration as follows.

```yaml
turn_taking:
  enabled: false
```

`build_turn_taking` then returns `None`, the processor is left out of the pipeline, and turn boundaries come
from VAD alone through the LLM user aggregator's turn strategies. This is what
`server_configs/default_nvidia.yaml` ships with, since the hosted ASR path does not emit `<EOU>` / `<EOB>`.
Expect turn ends to be governed entirely by `vad.stop_secs` in that mode, and note that backchannel
suppression is inactive because the component that implements it is gone.

The two evaluation configs (`evaluation/server_configs/agent.yaml` and `user.yaml`) take the opposite tack:
turn taking stays enabled but `backchannel_phrases_path: null`, `max_buffer_size: 0`, and
`bot_stop_delay: 0.0` — the bot-to-bot harness wants the fastest, most deterministic barge-in possible.

## Diarization Interaction

When `diar.enabled` is true, `NeMoTurnTakingService` receives `DiarResultFrame` and prefixes the buffered
utterance with a `<speaker_N>` tag, which is stripped again before backchannel matching. A buffer that
contains only a speaker tag is never pushed downstream. See [Diarization](diarization.md).

## Related Topics

Use these pages to understand the recognition and diarization signals that influence turn boundaries.

- [ASR](asr.md) — EOU/EOB-capable models and `stt.ignore_eou_eob`.
- [Server configuration](../../../build-voice-agents/configure/server-config.md) — how `default.yaml` and model sub-YAMLs merge.
- [Config schema](../../../reference/runtime/config-schema.md) — full key list.
- [Troubleshooting](../../../troubleshooting/index.md) — barge-in and latency symptoms.
