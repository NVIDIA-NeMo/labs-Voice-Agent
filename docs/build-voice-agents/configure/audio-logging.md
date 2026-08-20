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

# Audio Capture & Logging

NeMo Labs Voice Agent writes two independent kinds of artifact: per-session **audio + transcript
capture** (off by default) and the **rotating text log** the pipeline emits at runtime. This page
covers how to turn on the first and how to read the second.

## Enable audio capture

Audio capture is controlled by two keys in the `transport:` block of the top-level server config
(`examples/generic_voice_agent/server/server_configs/default.yaml`, and the same block in
`default_nvidia.yaml`):

```yaml
transport:
  audio_out_10ms_chunks: 8
  record_audio_data: true          # default: false
  audio_log_dir: "./audio_logs"    # default: "./audio_logs"
```

| Key | Default | Effect |
| --- | --- | --- |
| `transport.record_audio_data` | `false` | Master switch. When false, no capture code runs at all. |
| `transport.audio_log_dir` | `"./audio_logs"` | Base directory, resolved relative to the server's working directory. |

`build_audio_logger` in `nemo_voice_agent/pipecat/services/nemo/builders.py` reads those two keys.
When `record_audio_data` is false it returns `None`, and every downstream service (STT, TTS,
turn-taking, diarization) is constructed with `audio_logger=None`, so capture costs nothing.

The session ID is generated once at **server start**, not per client connection:
`session_YYYYMMDD_HHMMSS`. Because the server keeps one pipeline alive across reconnects (see
[Server Config](server-config.md)), a client that disconnects and reconnects keeps appending to the
same session directory.

`audio_logs/`, `audio_logs_user/`, and `audio_logs_agent/` are gitignored.

## What lands on disk

```text
audio_logs/
└── session_20260806_141200/
    ├── user/
    │   ├── 00001_141207.wav
    │   ├── 00001_141207.json
    │   └── ...
    ├── agent/
    │   ├── 00001_141203.wav
    │   ├── 00001_141203.json
    │   └── ...
    ├── session_metadata.json
    └── conversation_stereo.wav
```

| Artifact | Contents |
| --- | --- |
| `user/NNNNN_HHMMSS.wav` | One user turn, mono 16-bit at 16 kHz, sliced out of the continuous input buffer by the turn's start/end times. |
| `user/NNNNN_HHMMSS.json` | Metadata + the final transcription for that turn. |
| `agent/NNNNN_HHMMSS.wav` | One synthesized TTS segment (mono 16-bit) at the TTS service's output sample rate. A single agent turn is usually several segments. |
| `agent/NNNNN_HHMMSS.json` | Metadata + the text that was synthesized. |
| `session_metadata.json` | Rolling index of every entry; rewritten after each save and again at finalize. |
| `conversation_stereo.wav` | Whole-session mixdown, left channel = agent, right channel = user, 16 kHz. Written only at finalize. |

The `NNNNN` prefix is a per-speaker counter (users and agents count separately), and `HHMMSS` is
wall-clock time at save. User turns get an 0.8 s pre-roll prepended, clamped so a turn never
overlaps the previous entry's end time.

### Metadata fields

Both sides share `base_name`, `counter`, `turn_index`, `speaker`, `timestamp` (ISO 8601),
`start_time` / `end_time` (float seconds from the first audio frame of the session),
`audio_file`, `sample_rate`, `num_channels`, and `audio_duration_sec`. Beyond that:

| Field | Side | Meaning |
| --- | --- | --- |
| `transcription` | user | Final ASR text for the turn. |
| `is_backchannel` | user | True when turn-taking classified the utterance as a backchannel, so it did not interrupt the bot. See [Turn Taking](../../about/core-concepts/speech-pipeline/turn-taking.md). |
| `num_audio_chunks`, `num_transcription_chunks` | user | How many streaming chunks were merged into this turn. |
| `model`, `backend` | user | ASR model name and backend that produced the transcription. |
| `text` | agent | The text handed to TTS for this segment. |
| `model` | agent | TTS model name. |
| `cutoff_time` | agent | `null` when the segment played to completion; a float (seconds from session start) when the user interrupted. Interruption stamps the same value on **every** segment of the current turn and zeroes the agent channel of the stereo mix after that point. |

`session_metadata.json` holds `session_id`, `start_time`, `last_updated`, a flat `user_entries`
list, and an `agent_entries` list where each element is itself the list of segments for one agent
turn. `finalize_session` adds `end_time`, `total_user_entries`, `total_agent_segments`, and
`total_agent_turns`.

## How capture is wired

`AudioLogger` (`nemo_voice_agent/pipecat/services/nemo/audio_logger.py`) is a plain object passed
into the services by the builders; each service pushes data into it at the right moment.

| Source | What it contributes |
| --- | --- |
| STT service | Stamps the session's first-audio timestamp, appends **every** input chunk to the continuous user buffer (not VAD-gated, so the user channel includes silence), and stages the turn's audio + transcription. |
| Turn-taking service | Marks backchannel turns, records when the bot actually starts speaking, sets `cutoff_time` on interruption, and advances the turn index for user turns. |
| TTS service | Calls `log_agent_audio` once per synthesized segment and advances the turn index for agent turns. |
| `RTVIAudioLoggerObserver` | A pipeline observer that flushes the staged user turn to disk when a `TranscriptionFrame` is pushed. It is added to the task's observer list unconditionally and no-ops when the logger is `None`. |
| `run_bot_websocket_server` | Calls `finalize_session` on client disconnect, on session timeout, and on pipeline shutdown — this is what writes `conversation_stereo.wav`. |

Two limitations worth knowing:

- `build_audio_logger` passes only the directory, session ID, and enabled flag. `AudioLogger`'s
  other constructor arguments (user sample rate, pre-roll seconds, rounding precision) are **not**
  exposed through YAML, so their defaults always apply. Change them by constructing `AudioLogger`
  yourself — see [Builders](../extend/pipelines/builders.md).
- The `AudioLogger` docstring records a known issue with `conversation_stereo.wav`: the two channels
  need roughly a -0.8 s offset applied to sound in sync. The per-turn WAV files are unaffected.

Quick check after a session:

```bash
ls audio_logs/session_*/
python -m json.tool audio_logs/session_*/session_metadata.json | head -40
```

## Server log file

Logging is configured by `setup_logging` in `nemo_voice_agent/utils/misc.py`, which installs a
colorized stderr sink plus a file sink with `rotation="1 day"` at `DEBUG` level. Once a day loguru
rolls the active file aside into a timestamped sibling, so when debugging a failure check the
**newest** `bot_server.*.log` and not only `bot_server.log`.

`setup_rotating_log` wraps that with a rename-or-delete step for a pre-existing file: it either
removes the old log or renames it to `bot_server.<YYYYmmdd_HHMMSS>.log` before installing the sinks.

| Key | Default | Read by |
| --- | --- | --- |
| `server.log_file` | `"bot_server.log"` | `evaluation/bot_server.py` |
| `server.log_level` | `"DEBUG"` | `evaluation/bot_server.py` |
| `server.create_new_log` | `false` | `evaluation/bot_server.py` (roll the existing log aside at startup) |
| `server.overwrite_existing_log` | `false` | `evaluation/bot_server.py` (delete instead of rename) |

**Gotcha:** the example server (`examples/generic_voice_agent/server/server.py`) calls
`setup_logging()` with no arguments, so it always writes `bot_server.log` at `DEBUG` regardless of
the `server.log_file` / `server.log_level` values in `default.yaml`. Those keys take effect for the
evaluation bot servers, which resolve them through `resolve_log_file_path` and `setup_rotating_log`.
The call is repeated after service construction because model libraries reconfigure loguru during
import.

## Evaluation runs

The evaluation role configs (`evaluation/server_configs/agent.yaml` and `user.yaml`) also ship with
`record_audio_data: false`, and give each role its own `audio_log_dir` (`./audio_logs_agent`,
`./audio_logs_user`) so the two bots do not collide. Independently of this, the bridge writes
`conversation_log.wav` (stereo), `conversation_log.txt`, and `conversation_log.seglst.json` into each
scenario's output directory — see [Evaluation Results](../../evaluate/run-evaluations/results.md).

## Related pages

- [Server Config](server-config.md) — the rest of the top-level YAML.
- [Configuration](index.md) — how `default.yaml` and the model sub-configs merge.
- [Config Schema](../../reference/runtime/config-schema.md) — full key reference.
- [Troubleshooting](../../troubleshooting/index.md) — reading the logs when a session misbehaves.
