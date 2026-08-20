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

# Release Notes

Dated feature updates for NeMo Labs Voice Agent, newest first. This page is the documentation site's
canonical history of the milestones listed below.

There are no per-entry version tags: the package version is a single value declared in
`nemo_voice_agent/package_info.py` and exposed as `nemo_voice_agent.package_info.__version__`.
For changes made after the most recent entry below, read the repository commit history.

## Summary

| Date | Highlights |
| --- | --- |
| 2026-08-06 | Graduated to a standalone repository |
| 2026-06-13 | Voice-agent evaluation harness with four benchmark domains |
| 2026-05-15 | Omni (multimodal) LLM support |
| 2026-01-26 | Nemotron-3-Nano-30B-A3B-BF16 LLM and Magpie multilingual TTS |
| 2025-12-31 | Tool-calling examples; diarization model upgrade |
| 2025-11-14 | Joint ASR and end-of-utterance detection |
| 2025-10-10 | Kokoro-82M TTS |
| 2025-10-03 | vLLM serving with automatic HuggingFace fallback |
| 2025-09-05 | First release |

## 2026-08-06 — Standalone repository

NeMo Labs Voice Agent graduated out of
[NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/voice_agent), where it had
lived as a sub-package, and is now developed in its own repository. The Python package name
(`nemo_voice_agent`) and every import path are unchanged.

## 2026-06-13 — Evaluation harness

Shipped the two-bot evaluation harness: a simulated user bot talks to the agent under test through a
bridge that shuttles audio between two WebSocket Pipecat servers and scores the outcome.

Four benchmark domains ported:

| Domain | Scenarios | Upstream source |
| --- | --- | --- |
| `eva_airline` | 50 | ServiceNow `eva` |
| `tau2_airline` | 50 | tau2-bench airline |
| `tau2_retail` | 114 | tau2-bench retail |
| `tau2_telecom` | 114 | tau2-bench telecom (dual-side: separate agent-side and user-side DBs) |

`tau2_telecom` is registered twice — a second `tau2_telecom_workflow` family shares the same 114
upstream tasks and differs only in which policy file is rendered into the agent prompt.

Scoring is per-scenario: each scenario declares a `success_signals` whitelist over the six signals
`is_action_match`, `db_state_match`, `db_state_assertion`, `nl_assertion`, `judge_passed`, and
`clean_exit`.

Learn more: [Evaluation overview](../evaluate/index.md) ·
[Benchmarks](../evaluate/understand-scoring/benchmarks.md) · [Scoring](../evaluate/understand-scoring/scoring.md) ·
[eva_airline](../evaluate/domain-guides/eva-airline.md) ·
[tau2_airline](../evaluate/domain-guides/tau2-airline.md) ·
[tau2_retail](../evaluate/domain-guides/tau2-retail.md) ·
[tau2_telecom](../evaluate/domain-guides/tau2-telecom.md)

## 2026-05-15 — Omni (multimodal) LLM support

Added support for `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`, served through the
`llm_configs/nemotron_nano_v3_omni.yaml` model config (a `_think.yaml` sibling enables reasoning).
Omni models consume user audio directly, so the pipeline inserts a user audio buffer stage that
non-omni pipelines omit.

Learn more: [Multimodal models](core-concepts/language-models/multimodal.md) · [Reasoning](core-concepts/language-models/reasoning.md)

## 2026-01-26 — Nemotron-3-Nano-30B-A3B and Magpie TTS

- Added support for the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` LLM. The
  `llm_configs/nemotron_nano_v3.yaml` model config covers this family; the NVFP4 variant is the
  shipped default and needs a GPU with FP4 support.
- Added support for the `nvidia/magpie_tts_multilingual_357m` TTS model, via
  `tts_configs/magpie_tts_multilingual_357m.yaml`.

Learn more: [LLM backends](core-concepts/language-models/llm.md) · [TTS](core-concepts/speech-pipeline/tts.md) ·
[Model registry](../build-voice-agents/configure/model-registry.md)

## 2025-12-31 — Tool calling examples and diarization upgrade

- Added tool-calling examples. The Kokoro TTS service registers six direct functions that let the
  LLM change how the agent sounds mid-conversation: `tool_tts_speak_faster`, `tool_tts_speak_slower`,
  `tool_tts_set_speed`, `tool_tts_reset_speed`, `tool_tts_set_voice` (American or British accent,
  male or female voice), and `tool_tts_reset_voice`. A standalone example function,
  `tool_get_city_weather`, looks up current weather with the `python_weather` package.
- Updated the diarization model to `nvidia/diar_streaming_sortformer_4spk-v2.1`, which handles up to
  four speakers across user turns.

Learn more: [Tool calling](../build-voice-agents/tools/tool-calling.md) · [Custom tools](../build-voice-agents/tools/custom-tools.md) ·
[Diarization](core-concepts/speech-pipeline/diarization.md)

## 2025-11-14 — Joint ASR and end-of-utterance detection

Added support for `nvidia/parakeet_realtime_eou_120m-v1`, which predicts end-of-utterance jointly
with transcription. It is the shipped default STT model and is optimized for lowest latency; it does
not emit punctuation or capitalization.

Learn more: [ASR](core-concepts/speech-pipeline/asr.md) · [Turn taking](core-concepts/speech-pipeline/turn-taking.md)

## 2025-10-10 — Kokoro-82M TTS

Added support for the `hexgrad/Kokoro-82M` TTS model, configured through
`tts_configs/kokoro_82M.yaml`.

Learn more: [TTS](core-concepts/speech-pipeline/tts.md)

## 2025-10-03 — vLLM serving and automatic backend selection

- Added vLLM as an LLM serving backend alongside HuggingFace.
- Added automatic backend selection: with `llm.type: auto`, the server tries vLLM first and falls
  back to HuggingFace, so parameters for both backends must be valid.
- Made `nvidia/NVIDIA-Nemotron-Nano-9B-v2` the default LLM at the time. (The current default is the
  Nemotron-3-Nano-30B-A3B family — see the 2026-01-26 entry.)

Learn more: [vLLM backend](../build-voice-agents/model-serving/vllm.md) · [LLM backends](core-concepts/language-models/llm.md)

## 2025-09-05 — First release

First release of NeMo Labs Voice Agent: a real-time speech-to-speech pipeline built on Pipecat, with
streaming ASR, speaker diarization, LLM, and TTS behind a WebSocket server, plus a browser client.
Shipped at the time as a sub-package of
[NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/voice_agent); see the
2026-08-06 entry for the move to a standalone repository.

Learn more: [Architecture](architecture.md) ·
[Quickstart](../get-started/quickstart.md)

## Upcoming

Work called out as next up in the repository `README.md`:

- Accuracy and robustness improvements to the ASR models.
- Combined ASR and speaker diarization to handle overlapping speech.
- More evaluation domains and scenarios.
