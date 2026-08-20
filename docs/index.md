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

# NeMo Voice Agent

NeMo Voice Agent is an open-source Python framework for building and evaluating real-time voice agents. It
combines streaming speech recognition, turn taking, language models, and text-to-speech services in a Pipecat
pipeline with a browser client and a spoken-conversation evaluation harness.

## Skip Ahead

| I want to | Start here |
| --- | --- |
| Understand the project | [Overview](about/overview.md) |
| Run the default voice agent | [Choose your path](get-started/choose-your-path.md) |
| Configure or extend an agent | [Build Voice Agents](build-voice-agents/index.md) |
| Measure an agent | [Evaluate Voice Agents](evaluate/index.md) |
| Resolve a problem | [Troubleshooting](troubleshooting/index.md) |

## Use Cases

- Run a local voice pipeline with configurable speech, language, and text-to-speech models.
- Connect supported stages to hosted NVIDIA endpoints.
- Add application tools or custom frame processors.
- Evaluate an in-repository or external voice agent through live audio conversations.

## Core Concepts

- [Speech pipeline](about/core-concepts/speech-pipeline/overview.md) covers recognition, diarization, turn
  taking, and synthesis.
- [Language models](about/core-concepts/language-models/overview.md) covers backends, reasoning, and
  multimodal execution.
- [Architecture](about/architecture.md) shows how the pipeline stages and control messages fit together.

## Core Components

| Component | Role |
| --- | --- |
| Browser client | Captures microphone audio, opens the WebSocket session, and plays assistant audio. |
| Voice-agent server | Loads configuration, builds the Pipecat pipeline, and serves one client connection. |
| Model services | Provide ASR, LLM, diarization, turn-taking, and TTS behavior. |
| Evaluation harness | Connects a simulated user and agent, records artifacts, and scores scenario outcomes. |

See [Release Notes](about/release-notes.md) for dated project updates.
