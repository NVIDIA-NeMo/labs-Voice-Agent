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

# Speech Pipeline

The speech pipeline turns live microphone audio into text, decides when the user has finished speaking, and
turns the assistant response back into audio. The stages stream data rather than waiting for an entire turn.
They include voice activity detection (VAD), automatic speech recognition (ASR), speaker diarization, turn
taking, large language model (LLM) inference, and text-to-speech (TTS).

```text
WebSocket audio -> VAD -> ASR -> diarization -> turn taking -> LLM -> TTS -> WebSocket audio
```

Optional stages are omitted by the builders when their configuration disables them. For the complete
runtime ordering and the frames exchanged between stages, refer to [Architecture](../../architecture.md).

## Core Concepts

The following pages explain the stages that interpret incoming audio and produce spoken output.

| Concept | What It Explains |
| --- | --- |
| [Speech recognition](asr.md) | Streaming transcription, model selection, and end-of-utterance tokens. |
| [Speaker diarization](diarization.md) | Assigning speaker identities across user turns. |
| [Text to speech](tts.md) | Streaming synthesis, voices, and text aggregation. |
| [Turn taking](turn-taking.md) | VAD, end-of-utterance signals, interruptions, and backchannels. |

## Related Topics

Use these pages to experience the pipeline, change its configuration, or understand the complete architecture.

- [Quickstart](../../../get-started/quickstart.md) — run the complete pipeline.
- [Configure](../../../build-voice-agents/configure/index.md) — select models and tune pipeline behavior.
