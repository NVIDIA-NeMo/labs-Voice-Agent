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

# Overview

NeMo Voice Agent is an open-source Python framework for building and evaluating real-time voice agents.
It assembles streaming speech recognition, turn taking, a language model, and text-to-speech services in a
Pipecat pipeline that you can configure or extend.

## What you can do

- Build a browser-based voice agent with local models or hosted NVIDIA endpoints.
- Configure speech, language, and text-to-speech models through YAML.
- Add tools, custom frame processors, or a different pipeline assembly.
- Evaluate voice agents through spoken, two-bot scenarios with deterministic and LLM-judged signals.

## Learn how the system works

- [Architecture](architecture.md) follows audio and control messages through the runtime pipeline.
- [Speech pipeline](core-concepts/speech-pipeline/overview.md) explains recognition, diarization, synthesis,
  and turn taking.
- [Language models](core-concepts/language-models/overview.md) explains the LLM backends, reasoning, and
  multimodal execution path.

## Choose a journey

- [Choose your path](../get-started/choose-your-path.md) if this is your first visit.
- [Build voice agents](../build-voice-agents/index.md) to configure models, add tools, or extend the pipeline.
- [Evaluate voice agents](../evaluate/index.md) to run scenarios and interpret scores.
- [Reference](../reference/index.md) for configuration, runtime, CLI, and metrics lookup.
