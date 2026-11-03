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

Turn taking decides when the user's turn ends, when the bot can speak, and which user utterances can
interrupt the bot. The `turn_taking.type` key selects the mechanism that makes those decisions. Under
`type: nemo`, NeMo Labs Voice Agent combines two signals: Silero voice activity detection (VAD) and
automatic speech recognition (ASR) end-of-utterance tokens. Backchannel suppression prevents short
acknowledgments such as "uh-huh" from interrupting the bot mid-sentence.

## Turn-Taking Modes

`turn_taking.type` accepts two values. `ConfigManager._configure_turn_taking` validates the value at startup,
and any other value raises an assertion that lists the two it accepts.

| Value | What Ends the Turn | Choose It When |
| --- | --- | --- |
| `nemo` | `NeMoTurnTakingService` reads ASR `EOU` and `EOB` tokens alongside VAD frames and emits the user-turn frames itself. `build_turn_taking` returns the service, and the pipeline gains a turn-taking stage. | You run a local EOU-capable ASR model, or you want backchannel suppression, diarization-aware buffering, or early interruption through `max_buffer_size`. |
| `speech_timeout` | Pipecat's `VADUserTurnStartStrategy` and `SpeechTimeoutUserTurnStopStrategy` run inside the user aggregator. With `stt.type: nemo_speechlm` and a non-omni LLM, the stop strategy is the `SegmentedSTTUserTurnStopStrategy` subclass instead. Refer to Multi-Segment Turns. `build_turn_taking` returns `None`, so the pipeline has no turn-taking stage. | Your ASR model emits no `EOU` token, such as the hosted `nvidia` path, or you want plain silence-based endpointing. |

The default is `nemo` when the key is absent. The boolean `turn_taking.enabled` key that earlier releases used
is gone. `ConfigManager` logs a warning that names the replacement and then ignores the old key, so a config
that relied on `enabled: false` runs NeMo turn taking until you migrate it. Migrate `enabled: true` to
`type: nemo` and `enabled: false` to `type: speech_timeout`.

No shipped configuration pairs `stt.type: nemo` with `type: speech_timeout`. The stop strategy ends a turn
only after at least one final transcript arrives, and `NemoSTTService` pushes interim transcripts whenever it
defers finalization to `NeMoTurnTakingService`, which the shipped `nvidia/parakeet_realtime_eou_120m-v1` model
selects. Pair `speech_timeout` with an ASR backend that emits final transcripts itself, such as `nvidia` or
`nemo_speechlm`.

## How Turn Detection Works

Under `type: nemo`, three pipeline stages cooperate:

| Stage | Component | Emits |
| --- | --- | --- |
| VAD | `VADProcessor` (Pipecat), built by `build_vad_processor` immediately after `transport.input()` | `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame` |
| ASR | `NemoSTTService` in `nemo_voice_agent/pipecat/services/nemo/stt.py` | Transcript text, with `<EOU>` and `<EOB>` tokens appended when the model supports them |
| Turn Taking | `NeMoTurnTakingService` in `nemo_voice_agent/pipecat/services/nemo/turn_taking.py` | `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`, `InterruptionFrame`, and final or interim transcription frames |

Since Pipecat 1.0, exactly one component can emit user-turn frames. `build_context_and_aggregators` in
`nemo_voice_agent/pipecat/services/nemo/builders.py` enforces this ownership. When a turn-taking service
exists, the function selects `ExternalUserTurnStrategies`, so the large language model (LLM) user aggregator
does not emit those frames. When the service is `None`, the aggregator uses the VAD-driven strategies of the
`speech_timeout` mode. For the construction logic, refer to
[Builders](../../../build-voice-agents/extend/pipelines/builders.md).

`<EOU>` (end of utterance) means the user finished a turn. `<EOB>` (end of backchannel) means the ASR model
flagged the segment as a backchannel. Only ASR models listed in `ASR_EOU_MODELS` in `stt.py` produce these
tokens. The shipped default `nvidia/parakeet_realtime_eou_120m-v1` does. Set `stt.ignore_eou_eob: true` to
strip both tokens and fall back to VAD-only boundaries. For model details, refer to [ASR](asr.md).

## VAD Settings

Configure VAD under the top-level `vad:` block. `ConfigManager.get_vad_params()` maps these settings onto
Pipecat's `VADParams`. `server_configs/default.yaml` ships the following values:

| Key | Default | Meaning |
| --- | --- | --- |
| `vad.type` | `silero` | Analyzer type. |
| `vad.confidence` | `0.6` | Speech-probability threshold. |
| `vad.start_secs` | `0.1` | Minimum speech duration before a start-speaking event fires. |
| `vad.min_volume` | `0.2` | Minimum audio volume. Frames below this value are never speech. |
| `vad.stop_secs` | `1.2` | Silence required before a stop-speaking event fires. |

`vad.stop_secs` is the main latency setting when the ASR model does not supply `<EOU>`. Lower values respond
faster but can cut off a user during a mid-sentence pause. With an EOU-capable ASR model, the end of turn
usually arrives from `<EOU>` first, and VAD stop acts as the fallback. Under `type: speech_timeout`,
`vad.stop_secs` is only the first half of the wait, as described in End-of-Turn Latency. The log line
`[EOU missing] STT failed to detect end of utterance before VAD detected user stopped speaking` in
`bot_server.log` tells you the fallback fired.

The analyzer runs at `transport.audio_in_sample_rate` and falls back to the configuration-wide sample rate.
VAD is not optional on the WebSocket path: `build_vad_analyzer` always returns an analyzer.

## Turn-Taking Settings

The following settings control turn finalization, backchannel filtering, and response timing.

| Key | Default | Meaning |
| --- | --- | --- |
| `turn_taking.type` | `nemo` when the key is absent | `nemo` or `speech_timeout`. Refer to Turn-Taking Modes. |
| `turn_taking.backchannel_phrases_path` | `"./backchannel_phrases.yaml"` | YAML file path, inline list, or `null`. Refer to Backchannel Suppression. |
| `turn_taking.max_buffer_size` | `2` | Number of completed words that can accumulate mid-utterance before the bot is interrupted and an interim transcript is pushed downstream. Lower interrupts sooner. |
| `turn_taking.bot_stop_delay` | `0.5` | Seconds to keep treating the bot as "still speaking" after `BotStoppedSpeakingFrame`, covering audio still buffered on the client. `0` flips the flag immediately. |
| `turn_taking.user_speech_timeout` | `0.6` | Seconds that `SpeechTimeoutUserTurnStopStrategy` waits after the VAD reports the end of speech, on top of `vad.stop_secs`. `ConfigManager` reads it for either type, but only `type: speech_timeout` applies it. |
| `turn_taking.user_turn_stop_timeout` | `null` (off) | Seconds without speech or transcript activity before Pipecat's watchdog ends an open user turn early. `null` or `0` disables the watchdog. Applies to both types. Refer to User-Turn Watchdog. |

The `backchannel_phrases_path`, `max_buffer_size`, and `bot_stop_delay` keys belong to the NeMo service. Under
`type: speech_timeout`, `ConfigManager` replaces them with an empty phrases path, a `max_buffer_size` of 0, and
a `bot_stop_delay` of 0.0, and nothing reads them. It does not validate them either, so a
`backchannel_phrases_path` pointing at a missing file is harmless on this path.

`bot_stop_delay` affects backchannels because the service applies suppression only while it considers the
bot to be speaking. A value that is too small lets a trailing "okay" from the user start a new LLM turn.

## End-of-Turn Latency

The two modes finish a turn at different moments, and the difference is easy to miss.

Under `type: nemo`, `NeMoTurnTakingService` pushes `UserStoppedSpeakingFrame` as soon as it accepts the end of
the turn, either from an `<EOU>` token or from the VAD stop frame. End-of-turn latency is therefore
`vad.stop_secs` on the VAD path, and `turn_taking.user_speech_timeout` has no effect.

Under `type: speech_timeout`, the two waits are additive. The VAD emits its stop frame after `vad.stop_secs`
of silence, and only then does `SpeechTimeoutUserTurnStopStrategy` start its own timer:

```text
end of speech → vad.stop_secs → turn_taking.user_speech_timeout → user turn ends
```

With the default `user_speech_timeout` of 0.6 seconds, `vad.stop_secs: 0.8` produces an end-of-turn latency of
at least 1.4 seconds, not 0.8 seconds. The shipped configurations allow for the addition: `default_nvidia.yaml`
selects this mode and pairs it with `vad.stop_secs: 0.2`. Lower `turn_taking.user_speech_timeout` to shorten
the trailing wait, or raise it to give the user more room to resume speaking after a pause.

A second timer runs in parallel and can extend the wait. As a safety net for slow transcripts, the same
strategy also waits `max(0, stt.ttfs_p99_latency - vad.stop_secs)`, and it ends the turn only once both timers
have elapsed and at least one transcript has arrived. The shipped configurations leave `stt.ttfs_p99_latency`
unset, so Pipecat substitutes 1.0 seconds and logs `ttfs_p99_latency not set, using default 1.0s`. The wait
after the VAD stop frame is therefore:

```text
max(turn_taking.user_speech_timeout, max(0, stt.ttfs_p99_latency - vad.stop_secs))
```

A transcript marked finalized cancels the second timer, so the turn ends as soon as that transcript has
arrived and `turn_taking.user_speech_timeout` has elapsed. `NemoSpeechLMSTTService` always marks its
transcripts that way, because Pipecat's `SegmentedSTTService` returns one transcript per segment and sets the
flag in `push_frame`. `NemoSTTService` sets the flag on a final transcript that it emits itself. It emits one
only when it is not deferring finalization to `NeMoTurnTakingService`, and the shipped
`nvidia/parakeet_realtime_eou_120m-v1` model always defers, so the flag does not reach this strategy. Pipecat's
hosted `NvidiaSTTService` sets the flag directly on the final transcripts that it returns. Two shipped
configurations show both ends of the range:

| Configuration | `vad.stop_secs` | Post-VAD wait | End-of-turn latency |
| --- | --- | --- | --- |
| `default_nvidia.yaml` (`stt.type: nvidia`) | `0.2` | 0.6 s, or up to 0.8 s until the final transcript lands | 0.8 to 1.0 s |
| `default_salm.yaml` (`stt.type: nemo_speechlm`) | `1.2` | 0.6 s, since `1.0 - 1.2` clamps the second timer to 0, or longer until the last segment's transcription returns | 1.8 s or more |

To stop the safety net from dominating, measure the latency of your deployment and set
`stt.ttfs_p99_latency`. `get_stt_service_from_config` reads that key only for `stt.type: nemo`, so on the
`nvidia` and `nemo_speechlm` backends the 1.0-second substitute is the only value available. For the key
itself, refer to [ASR](asr.md).

## Multi-Segment Turns

`NemoSpeechLMSTTService`, the `stt.type: nemo_speechlm` backend, transcribes each VAD segment only after that
segment's `VADUserStoppedSpeakingFrame`. A mid-utterance pause longer than `vad.stop_secs` splits one user turn
into two or more segments, so the last transcript arrives after the last VAD stop frame of the turn.

Pipecat's `SpeechTimeoutUserTurnStopStrategy` can end such a turn before the last segment is transcribed. Its
transcript check is per turn, so the first segment's transcript already satisfies it. Its safety-net timer is
zero on `default_salm.yaml`, as End-of-Turn Latency shows, and even a nonzero timer can be skipped. VAD frames
are system frames, so the second segment's VAD start frame can overtake the first transcript on its way to the
user aggregator. The first transcript then sets the finalized flag after that VAD start frame has cleared it, and
the stale flag skips the timer at the last VAD stop. The aggregator keeps the late text and adds it to the start
of the next user message.

`build_context_and_aggregators` therefore uses `SegmentedSTTUserTurnStopStrategy` from
`nemo_voice_agent.pipecat.turns` when `turn_taking.type` is `speech_timeout`, `stt.type` is `nemo_speechlm`,
and `llm.is_omni_model` is false. The shipped `default_salm.yaml` meets these conditions. No configuration key
turns the strategy on or off, and every other combination keeps Pipecat's strategy.

The strategy subclasses `SpeechTimeoutUserTurnStopStrategy`, so the timers from End-of-Turn Latency still
apply. It also counts pending segments. Each `VADUserStoppedSpeakingFrame` adds one, and each
`STTSegmentDoneFrame` removes one, with a floor of zero. `NemoSpeechLMSTTService` pushes this data frame after
every segment. It follows the `TranscriptionFrame` when the transcription has text, and it arrives alone when
the transcription is empty or the request fails. The count therefore does not depend on whether VAD frames
overtake transcripts. The user aggregator passes `STTSegmentDoneFrame` downstream like any other data frame
that it does not handle.

While any segment is pending, the strategy holds a turn that would otherwise end, and it logs
`holding the user turn for` at the `DEBUG` level. The `STTSegmentDoneFrame` that clears the last pending segment
ends a held turn, so the user message carries the text of every segment. An empty or failed segment clears its
count as soon as its request returns, so it does not delay the end of the turn. A turn whose only segment comes
back empty still stays open until the user speaks again, as described in User-Turn Watchdog. Otherwise, turn
timing does not change, except that a turn now waits for the transcription of its last segment.

The `pending_transcript_timeout` argument, 10 seconds by default, is a backstop for a transcription that does
not return in time. It counts from the latest VAD stop frame, and each new VAD stop frame restarts it. When it
expires, the strategy logs a `WARNING` that ends in `giving up on them` and clears the pending count. A held turn
then ends without the missing segment's text. If that text arrives later, it starts the next user message, and the
segment's late `STTSegmentDoneFrame` is ignored, so it cannot clear a newer segment. A turn with no transcript yet
keeps waiting, because Pipecat's transcript check still applies. The builder passes no
`pending_transcript_timeout`, and no configuration key sets it. A custom pipeline can pass its own value.
Refer to
[Building Your Own Pipeline](../../../build-voice-agents/extend/pipelines/custom-pipeline.md#stop-strategy-for-offline-speechlm-transcription).

## User-Turn Watchdog

Pipecat's LLM user aggregator has a watchdog that can end a user turn early. When
`turn_taking.user_turn_stop_timeout` is a positive number, the watchdog ends an open user turn after that many
seconds pass with no VAD frame, user-turn frame, or transcript. It does not fire while the user is speaking.
`build_context_and_aggregators` passes the value to `LLMUserAggregatorParams` under both turn-taking types.

The watchdog is off by default. Pipecat's own default is 5.0 seconds, but `ConfigManager` resolves an absent
key, `null`, or `0` to `None`, which disables the watchdog. No shipped configuration sets the key. `0` maps to
`None` because Pipecat would otherwise run the watchdog in a tight loop that uses a full CPU core. That loop
ends every user turn as soon as the user stops speaking, without waiting for the transcript or
`turn_taking.user_speech_timeout`. A negative number, a
boolean, or a quoted string fails an assertion at startup. With the watchdog off, a user turn ends only when
its stop strategy fires.

The default protects slow transcripts. `NemoSpeechLMSTTService` starts transcribing only after the VAD stop
frame, so its reply can take several seconds. With a 5-second watchdog, a reply slower than that arrived after
the watchdog had already ended the empty turn. The user aggregator then kept the late text and added it to the
start of the next user message. In an evaluation run, the `update_system_prompt` action clears that text at the
start of each scenario. A transcript that arrives after that reset still starts the next scenario's first message.
Turning the watchdog off is enough for a turn with one VAD segment. For a turn that a mid-utterance pause
splits into several segments, refer to Multi-Segment Turns.

For a non-omni LLM, the cost is that a turn without a transcript no longer closes on its own. Both stop strategies
wait for transcript text: `ExternalUserTurnStopStrategy` under `type: nemo` and `SpeechTimeoutUserTurnStopStrategy`
or its `SegmentedSTTUserTurnStopStrategy` subclass under `type: speech_timeout`. When no transcript arrives, the
turn stays open until the user speaks again and that speech produces a transcript. For example,
`NemoSpeechLMSTTService` can receive an empty transcription, or its request can fail. The LLM receives nothing for
the lost utterance with or without the watchdog, because the aggregator does not push an empty user message.

Two related cases follow from turning the watchdog off:

- Under `type: nemo` with `backchannel_phrases_path: null`, as in the shipped evaluation `agent.yaml` and
  `user.yaml`, a VAD segment with no ASR text also leaves the turn open. `NeMoTurnTakingService` starts the
  turn at the VAD start frame but ends it only after transcript text arrives. A cough or background noise
  therefore keeps the turn open until the next transcribed utterance closes it.
- Under `type: speech_timeout` with a non-omni LLM, Pipecat's user turn controller ignores new turn starts
  while such a turn is open. The user's next speech therefore does not broadcast an interruption until a
  transcript closes the open turn. The bot is normally silent then, because the opening turn already interrupted it. It
  can be speaking only if something else started it, such as `talk_first` on a reconnect or the evaluation
  bridge's opening `send-text` message. Under `type: nemo`, interruptions do not depend on the user turn
  controller, because `NeMoTurnTakingService` pushes them itself. The watchdog setting therefore does not change
  its barge-in behavior.

To bound such turns, restore a finite watchdog. Set a value longer than your slowest transcription reply, for
example 30 seconds:

```yaml
turn_taking:
  user_turn_stop_timeout: 30.0  # seconds; null or 0 disables the watchdog
```

When the watchdog fires, the user aggregator logs `User stopped speaking (strategy: None)` at the `DEBUG`
level. Search `bot_server.log` and the rotated `bot_server.<timestamp>.log` files for that line.

### Omni Models

An omni LLM hears the audio itself, so its turns do not wait for a transcript. When `llm.is_omni_model` is
true, `UserAudioBuffer` consumes every `TranscriptionFrame` before the user aggregator. A stop strategy that
waits for text would never fire. `build_context_and_aggregators` therefore sets `wait_for_transcript=False`
on the stop strategy under both turn-taking types:

- Under `type: speech_timeout`, the turn ends `vad.stop_secs` plus `turn_taking.user_speech_timeout` after
  the user stops speaking. The safety-net timer from End-of-Turn Latency can extend that wait, because no
  finalized transcript cancels it. `UserAudioBuffer` then adds the audio message and runs the LLM. With
  `evaluation/server_configs/agent_nvidia_omni.yaml`, a turn ends 1.0 second after the user stops speaking.
  The builder keeps Pipecat's `SpeechTimeoutUserTurnStopStrategy` even with `stt.type: nemo_speechlm`,
  because the turn has no transcript for `SegmentedSTTUserTurnStopStrategy` to wait for.
- Under `type: nemo`, the `UserStoppedSpeakingFrame` from `NeMoTurnTakingService` drives `UserAudioBuffer`,
  and the aggregator turn closes on the same frame. Because the turn is already closed, a finite watchdog does
  not send a second `UserStoppedSpeakingFrame` after each utterance. Such a frame would add a duplicate audio
  message and a second LLM run.

Because the turn does not wait for a transcript, a transcript that arrives after the turn ends stays in the
buffer for the next turn. With `llm.use_stt_transcript: true`, the next turn's audio message carries it.
Every shipped omni configuration sets that key to `false`. The `keep_only_last_audio_turn` stand-in text reads
the same buffer, so a late transcript also moves into the next turn's stand-in. For the buffer itself, refer
to [Multimodal and Omni Models](../language-models/multimodal.md#how-the-omni-switch-changes-the-pipeline).

## Backchannel Suppression

The configuration key is **`turn_taking.backchannel_phrases_path`** (not `backchannel_phrases`). Despite the
name, it accepts three forms. `ConfigManager._resolve_backchannel_phrases` resolves the value, and
`NeMoTurnTakingService._load_backchannel_phrases` validates it:

```yaml
turn_taking:
  # 1. Path to a YAML file containing a flat list of phrases (shipped default).
  backchannel_phrases_path: "./backchannel_phrases.yaml"

  # 2. Inline list — no file needed.
  # backchannel_phrases_path: ["uh huh", "mhmm", "right", "okay"]

  # 3. null / empty — disable backchanneling entirely; any speech interrupts the bot.
  # backchannel_phrases_path: null
```

The shipped file is `examples/generic_voice_agent/server/backchannel_phrases.yaml` and contains 78 phrases,
including `uh huh`, `mhmm`, `okay`, `right`, `sure`, and `i see`. The resolver checks a relative path against
the current working directory and then the server base directory. The default therefore works from the
repository root or the server directory. A path that exists in neither location raises `FileNotFoundError`
and names both attempts.

The `clean_text` and `is_backchannel` functions in `turn_taking.py` apply the following matching rules:

- Comparison is case-insensitive and whitespace-normalized.
- A trailing `<EOU>` or `<EOB>` token and any leading `<speaker_N>` tag are stripped first.
- Every character except `a`-`z`, apostrophes, and whitespace is **deleted, not replaced by a space**.
  Therefore, `uh-huh` normalizes to `uhhuh`, and the shipped file lists both `uh huh` and `uh-huh`.
- The match is exact against the normalized set. There is no substring or fuzzy matching.
- Only English is supported — `clean_text` raises `ValueError` for any other `Language`.

A phrase is suppressed **only while the bot is speaking**. The service pushes the text upstream as a
transcription frame wrapped in parentheses, for example `(uh huh)`. It reaches the client transcript and
audio logger without entering the LLM context or triggering an interruption. When the bot is silent the
same phrase is treated as ordinary user speech and starts a normal turn.

Anything longer than the phrase list still interrupts. After `max_buffer_size` completed words accumulate,
`NeMoTurnTakingService` emits `UserStartedSpeakingFrame` plus `InterruptionFrame` without waiting for
`<EOU>`.

To confirm the list loaded, grep the server log:

```bash
grep -E "backchannel phrases|Backchannel detected" examples/generic_voice_agent/server/bot_server.log
```

`Loading backchannel phrases from file:` or `Using backchannel phrases from list:` appears at startup, and
`Backchannel detected:` appears each time the service suppresses a phrase. When audio logging is enabled, the
suppressed segment is tagged `is_backchannel`. For details, refer to
[Audio logging](../../../build-voice-agents/configure/audio-logging.md).

## Use the Speech Timeout Mode

To let VAD drive turn boundaries without the NeMo turn-taking service, set the type and tune the timeout.

```yaml
turn_taking:
  type: speech_timeout
  user_speech_timeout: 0.6
```

`build_turn_taking` then returns `None`, and the server omits the processor from the pipeline. Turn boundaries
come from VAD through the LLM user aggregator's turn strategies, which `build_context_and_aggregators` sets to
`VADUserTurnStartStrategy` and `SpeechTimeoutUserTurnStopStrategy`. With `stt.type: nemo_speechlm` and a
non-omni LLM, the stop strategy is `SegmentedSTTUserTurnStopStrategy`, as described in Multi-Segment Turns.
Backchannel suppression, `max_buffer_size`, and `bot_stop_delay` are inactive. The shipped
`server_configs/default_nvidia.yaml` uses this mode because the hosted ASR path does not emit `<EOU>` or
`<EOB>`, and it pairs the mode with a short `vad.stop_secs` of 0.2 seconds.

In contrast, the two evaluation configurations (`evaluation/server_configs/agent.yaml` and `user.yaml`) keep
the NeMo mode. They set `backchannel_phrases_path: null`, `max_buffer_size: 0`, and `bot_stop_delay: 0.0` to
provide fast, deterministic barge-in for the bot-to-bot harness.

## Diarization Interaction

When `diar.enabled` is true, `NeMoTurnTakingService` receives `DiarResultFrame` and prefixes the buffered
utterance with a `<speaker_N>` tag, which is stripped again before backchannel matching. A buffer that
contains only a speaker tag is never pushed downstream. For speaker attribution details, refer to
[Diarization](diarization.md).

## Related Topics

Use these pages to understand the recognition and diarization signals that influence turn boundaries.

- [ASR](asr.md) — EOU- and EOB-capable models and `stt.ignore_eou_eob`.
- [Server configuration](../../../build-voice-agents/configure/server-config.md) — how `default.yaml` and model sub-YAMLs merge.
- [Config schema](../../../reference/runtime/config-schema.md) — full key list.
- [Troubleshooting](../../../troubleshooting/index.md) — barge-in and latency symptoms.
