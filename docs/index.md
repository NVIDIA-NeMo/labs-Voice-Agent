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

# NeMo Labs Voice Agent

NeMo Labs Voice Agent is an open-source real-time voice-agent pipeline built on
Pipecat and NVIDIA NeMo speech models.

## ✨ Key Features

- Open-source, local deployment, and flexible customization.
- Talk to most LLMs from vLLM/HuggingFace with configurable prompts.
- Streaming speech recognition with low latency and end-of-utterance detection.
- Low latency TTS for fast audio response generation.
- Speaker diarization up to 4 speakers in different user turns.
- WebSocket server for easy deployment.
- Tool calling for LLMs to use external tools and adjust its own behavior.
- Voice-agent evaluation harness with deterministic + LLM-judged scoring, and 328 scenarios across 4 primary benchmark domains (eva_airline, tau2_airline, tau2_retail, tau2_telecom).
