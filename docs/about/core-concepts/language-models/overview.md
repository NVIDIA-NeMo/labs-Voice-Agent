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

# Language Models

The language-model stage receives the aggregated conversation context and streams the assistant response to
text-to-speech. NeMo Voice Agent can run a HuggingFace model in process, call a vLLM server, or use a hosted
NVIDIA endpoint.

## Core concepts

The following pages explain how language models generate, reason about, and consume voice-agent context.

| Concept | What it explains |
| --- | --- |
| [LLM backends](llm.md) | Backend selection, configuration layers, and supported tool-calling paths. |
| [Reasoning mode](reasoning.md) | Thinking configurations and how reasoning text is kept out of speech. |
| [Multimodal and omni models](multimodal.md) | Direct audio input and the additional user-audio buffer stage. |

Model concepts describe runtime behavior. For deployment and endpoint setup, continue to
[Model serving](../../../build-voice-agents/model-serving/index.md).

## Related Topics

Use these pages to configure model behavior, serve a backend, or connect models to application tools.

- [System prompts](../../../build-voice-agents/configure/prompts.md) to control assistant behavior.
- [Tools](../../../build-voice-agents/tools/index.md) to let supported backends call functions.
