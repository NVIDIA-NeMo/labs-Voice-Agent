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

NeMo Voice Agent is an open-source framework to build and evaluate voice agents, with support on a wide range of ASR/LLM/TTS models. 

- No API keys required to get started! Happy hacking!

## ✨ Key Features

- Open-source, local/remote deployment, and flexible customization.
- Talk to LLMs from vLLM/HuggingFace with configurable prompts and tools.
- Streaming speech recognition with low latency and end-of-utterance detection.
- Low latency TTS for fast audio response generation.
- Speaker diarization up to 4 speakers in different user turns.
- Voice-agent evaluation harness with deterministic + LLM-judged scoring, and 328 scenarios across 4 primary benchmark domains.

## 📅 Recent Updates
- 2026-08-06: [NVIDIA-NeMo/labs-Voice-Agent](https://github.com/NVIDIA-NeMo/labs-Voice-Agent/tree/main) is opensource, graduating from a sub-package from [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/v3.0.0/examples/voice_agent).
- 2026-06-13: Voice-agent evaluation harness shipped — three primary benchmark domains ported (tau2-bench airline 50 / retail 114 / telecom 114 dual-side), plus eva_airline (50) from ServiceNow/eva. Per-scenario `success_signals` scoring. See [📊 Evaluation](./evaluate/index.md).
- 2026-05-15: Added support for [Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4). See [🤖 Multimodal / Omni](./models/multimodal.md).
- 2026-01-26: Added support for [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) LLM model, and support for [magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) TTS model.
- 2025-12-31: Added examples for [tool calling](./features/tool-calling.md), such as changing the speaking speed, switching between male/female voices and British/American accents (with [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)), and getting the current weather of a city (with `python_weather`). Diarization model is updated to [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) with improved performance.
- 2025-11-14: Added support for joint ASR and EOU detection with [Parakeet-realtime-eou-120m](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) model.
- 2025-10-10: Added support for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS model.
- 2025-10-03: Add support for serving LLM with vLLM and auto-switch between vLLM and HuggingFace, add [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) as default LLM.
- 2025-09-05: First draft of NeMo Voice Agent, supporting ASR+EOU->/LLM->TTS pipeline.