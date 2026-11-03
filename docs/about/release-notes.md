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

# Release Notes for NeMo Voice Agent

Dated feature updates for NeMo Labs Voice Agent appear newest first. This page is the documentation site's
canonical history of automatic speech recognition (ASR), large language model (LLM), text-to-speech (TTS),
and evaluation milestones.

There are no per-entry version tags: the package version is a single value declared in
`nemo_voice_agent/package_info.py` and exposed as `nemo_voice_agent.package_info.__version__`.
For changes after the most recent entry, refer to the repository commit history.

## Release History

The following table summarizes the dated project milestones documented on this page.

| Date | Highlights |
| --- | --- |
| 2026-11-02 | Nemotron-3.5-Transcribe-30B-A4B offline speech recognition; named turn-taking modes; evaluation auto-retry and conversation-end policy |
| 2026-09-09 | `tau2_airline` scores NL assertions on 24 scenarios; task 39 gold-action correction |
| 2026-09-02 | Nemotron-3.5-Lightning-30B-A3B-NVFP4 is the new shipped default LLM; added Qwen3.6-35B-A3B and Qwen3.8-27B configs |
| 2026-08-06 | Graduated to a standalone repository |
| 2026-06-13 | Voice-agent evaluation harness with four benchmark domains |
| 2026-05-15 | Omni (multimodal) LLM support |
| 2026-01-26 | Nemotron-3-Nano-30B-A3B-BF16 LLM and Magpie multilingual TTS |
| 2025-12-31 | Tool-calling examples and diarization model upgrade |
| 2025-11-14 | Joint ASR and end-of-utterance detection |
| 2025-10-10 | Kokoro-82M TTS |
| 2025-10-03 | vLLM serving with automatic Hugging Face fallback |
| 2025-09-05 | First release |

## 2026-11-02 — Nemotron-3.5-Transcribe Offline Speech Recognition

NeMo Labs Voice Agent now supports
[NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
for speech-to-text (STT). Nemotron-3.5-Transcribe is an open-weight, offline speech and audio model that
transcribes 40 languages. It is a Mamba-2 and Transformer hybrid mixture-of-experts (MoE) model with 34 billion
total parameters and about 4.3 billion active parameters, including its audio encoder.

The new `stt.type: nemo_speechlm` backend, `NemoSpeechLMSTTService`, sends each utterance to an
OpenAI-compatible vLLM endpoint that serves the model. Because the model is offline, the backend transcribes a
complete utterance after voice activity detection (VAD) reports the end of speech, instead of streaming partial
text.

To try it with the example server, complete the following steps:

1. Start the vLLM server for the agent LLM as
   [Start vLLM Yourself](../build-voice-agents/model-serving/vllm.md#start-vllm-yourself-shipped-default)
   describes. `default_salm.yaml` uses the same Nemotron-3.5-Lightning LLM at `http://localhost:8000/v1`.
2. Serve Nemotron-3.5-Transcribe with vLLM as the Quick Start on the model card describes, in an environment with
   the vLLM version and NeMo Speech plugin that the card lists. Add `--port 5002` to the Quick Start command.
   `default_salm.yaml` expects the server at `http://localhost:5002/v1` under the Quick Start's served model
   name, `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`, and the agent LLM already uses vLLM's default port
   8000. Alternatively, set `stt.model` and `stt.base_url` to match your server.
3. From `examples/generic_voice_agent/server/`, run
   `SERVER_CONFIG_PATH=./server_configs/default_salm.yaml python server.py`.

The Quick Start reserves 94 percent of GPU memory and the agent LLM reserves 80 percent, so run the two vLLM
servers on separate GPUs or lower their `--gpu-memory-utilization` values.

`default_salm.yaml` also sets the following values for offline transcription:

- `stt.max_tokens_per_sec: 20` caps the decode budget in proportion to the duration of the audio. On 10 seconds
  of audio, a repetition hallucination stops at 200 tokens instead of running to thousands.
- `stt.generation_kwargs.chat_template_kwargs.enable_thinking: false` keeps the model transcribing instead of
  answering the prompt as a chatbot would. The backend sends this flag by default, but a `chat_template_kwargs`
  block of your own replaces that default, so keep the flag in it.
- `vad.stop_secs: 1.2` with `turn_taking.type: speech_timeout` waits for a longer silence before transcribing,
  so each request usually carries a whole utterance rather than fragments.

This release also reworks turn taking, largely to support offline transcription:

- **Named turn-taking modes.** `turn_taking.type` replaces the boolean `turn_taking.enabled`. The `nemo` default
  runs the NeMo turn-taking service, and `speech_timeout` lets VAD end the user turn through Pipecat's
  `SpeechTimeoutUserTurnStopStrategy`. `ConfigManager` ignores a leftover `enabled` key and logs a warning, so a
  configuration with `enabled: false` now runs `nemo`. Replace it with `type: speech_timeout`.
- **Configurable speech timeout.** `turn_taking.user_speech_timeout`, 0.6 seconds by default, sets the extra
  silence that `speech_timeout` waits after `vad.stop_secs`. Pipecat already waited 0.6 seconds, so the key alone
  changes no timing. To shorten end-of-turn latency, the hosted `speech_timeout` configurations lower
  `vad.stop_secs`: `default_nvidia.yaml` from 1.0 to 0.2 seconds, `agent_nvidia.yaml` from 1.0 to 0.4,
  `agent_nvidia_omni.yaml` from 0.8 to 0.4, and `user_nvidia.yaml` from 1.5 to 0.9.
- **User-turn watchdog off by default.** Pipecat ends an open user turn after 5 seconds without activity, which
  sent a slow offline transcript to the next user turn. `turn_taking.user_turn_stop_timeout` now leaves that
  watchdog off in every configuration unless you set it to a positive number of seconds. As a trade-off, a
  non-omni turn whose transcription is empty or fails stays open until the user's next transcribed speech. Set
  the key to 5.0 to restore the previous behavior.
- **Multi-segment turns.** With `nemo_speechlm`, `speech_timeout`, and a non-omni LLM, a user turn no longer
  ends while one of its VAD segments is still being transcribed. The turn gives up on a segment 10 seconds after
  the latest VAD stop. A custom pipeline that uses `NemoSpeechLMSTTService` needs
  `SegmentedSTTUserTurnStopStrategy`, and a custom processor between the STT service and the user aggregator
  must forward `STTSegmentDoneFrame`.
- **Faster omni turns.** With an omni LLM under `speech_timeout`, the user turn now ends after `vad.stop_secs`
  plus `user_speech_timeout` instead of after the 5-second watchdog. The stop strategy no longer waits for a
  transcript that the omni pipeline consumes.

The evaluation harness has the following changes:

- `--conversation-end-policy` selects how the `clean_exit` signal is scored. The `tool-only` default still
  requires the agent to call the end-conversation tool. `valid-terminal-state` also accepts a successful end
  that the simulator reports, which then stops the scenario, or an inactivity timeout after a final user turn.
- `--inactivity-timeout`, 120 seconds by default, stops a scenario after that long without meaningful activity
  from either bot.
- The runner now re-runs a scenario that recorded zero conversation turns, once by default, instead of leaving
  it for a `--resume` run. `--no-auto-resume-on-stale` turns this off, `--max-auto-retries` caps the re-runs,
  and `--auto-retry-backoff-secs` sets the wait before each one. `--auto-resume-on-insufficient-turns` also
  re-runs scenarios below `--min-agent-turns`. The runner keeps each discarded attempt at
  `<scenario>.killed.autoretry<N>.<timestamp>/`, and `metrics.json` records `auto_retry_count`.
- `run_evaluation.py` now defaults `--min-agent-turns` to 2, down from 3. `check_resume.py` defaults to the same
  2 instead of 0, so it now flags stalled scenarios by default. It also lists scenarios that completed after an
  automatic retry.

The shipped model configurations have the following changes:

- The Nemotron-3.5-Lightning configurations, `nemotron_3.5_lightning.yaml` and its `_think.yaml` sibling, now
  sample with `temperature: 1.0` and no `top_k` limit, in place of 0.6 and 20.
- The hosted `default_nvidia.yaml` and `agent_nvidia.yaml` configurations, and the `llm.type: nvidia` fallback
  model, now use `nvidia/nemotron-3.5-lightning-30b-a3b` instead of `nvidia/nemotron-3-nano-30b-a3b`.
  `default_nvidia.yaml` passes `reasoning_budget` in place of `thinking_token_budget`.
- `default.yaml` lowers `vad.min_volume` from 0.4 to 0.2, so the default server detects quieter speech.

This release also includes the following fixes:

- A sentence that ends on a finished decimal, such as `It costs $3.14.`, now goes to text-to-speech (TTS) on
  its own instead of merging with the next sentence.
- The `reset` and `update_system_prompt` Real-Time Voice Interface (RTVI) actions now clear pending text from
  the user and assistant aggregators. Previously, the reset never ran, so an evaluation scenario could begin
  with text left over from the previous scenario.
- Logs and evaluation artifacts redact more API keys. `api_key` and `*_api_key` values in configuration log
  lines, `--api-key` in `llm.vllm_server_params` and in the captured vLLM output, and the judge key in
  `run_args.json` now appear as `<redacted>`. `Authorization` headers in `default_headers` are still logged.
  Services still receive the real keys.
- With `llm.start_vllm_on_init: true` and a vLLM API key, the readiness check now sends the key. Startup no
  longer waits the full hour for the server and then fails.

Learn more: [ASR](core-concepts/speech-pipeline/asr.md#offline-speechlm-transcription) ·
[Turn taking](core-concepts/speech-pipeline/turn-taking.md) ·
[Evaluation CLI](../reference/evaluation/eval-cli.md) ·
[Automatic in-run retry](../evaluate/run-evaluations/resume.md#automatic-in-run-retry) ·
[Scoring](../evaluate/understand-scoring/scoring.md) ·
[Security](../resources/contribute/security.md#handling-credentials) ·
[Serving with vLLM](../build-voice-agents/model-serving/vllm.md)

## 2026-09-09 — tau2_airline NL-Assertion Scoring and Task 39 Gold-Action Correction

The `tau2_airline` evaluation domain now scores an `NL_ASSERTION` signal on 24 of its 50 scenarios. Those
24 have no database-mutating reference action, so their expected database equals the seeded one and
`DB_STATE_MATCH` passed any agent that changed nothing — including one that greeted the caller and
immediately ended the call. Each now carries curated natural-language assertions, 84 in total and two to
nine per scenario, which the large language model (LLM) judge scores one verdict at a time. A scenario
passes only when the agent actually states the refusal or the fact under test. The remaining 26 scenarios
keep `(DB_STATE_MATCH, CLEAN_EXIT)` and are unaffected.

Unlike `tau2_retail`, which reads `evaluation_criteria.nl_assertions` from the upstream task file, airline
adopts a curated subset declared in `ADOPTED_NL_ASSERTIONS`. Upstream never scores its own airline
assertions, because all 50 tasks carry a `reward_basis` of `DB` and `COMMUNICATE` that excludes
`NL_ASSERTION`. An audit of all 123 upstream assertions found several that contradict the domain policy or
database, alongside many that only restate the reference actions.

Task 39's reference action `39_10` is also dropped before gold replay. It cancels a reservation that the
domain's own policy does not permit cancelling, which made the expected database reward a policy-violating
agent and penalize a correct one. That task's score is therefore intentionally no longer comparable to the
upstream published number; emptying the `GOLD_ACTION_DROPS` constant restores the previous behavior. Both
fixture files stay byte-identical to upstream, so the corrections live in code and re-imports cannot
overwrite them.

Learn more: [tau2_airline](../evaluate/domain-guides/tau2-airline.md) ·
[Scoring](../evaluate/understand-scoring/scoring.md) ·
[Benchmarks](../evaluate/understand-scoring/benchmarks.md)

## 2026-09-02 — Nemotron-3.5-Lightning Default LLM and New Qwen3.6-35B-A3B / Qwen3.8-27B Configs

The example server's shipped default LLM changed from `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` to
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, a Mamba-2 and MoE hybrid with attention, 30 billion
total and 3 billion active parameters, NVFP4-quantized, with a context length of up to 1,000,000 tokens.
`default.yaml` now points `llm.model` and `llm.model_config` at
`llm_configs/nemotron_3.5_lightning.yaml` (and its `_think.yaml` reasoning sibling), which serve the model
through vLLM with the same `--tool-call-parser qwen3_coder` and `--reasoning-parser nemotron_v3` flags as
the previous default. The `nemotron_nano_v3.yaml` and `nemotron_nano_v3_think.yaml` configurations remain
available and supported; they are no longer the default.

This release also added two new Qwen LLM configs, both served through vLLM with
`--tool-call-parser qwen3_coder --reasoning-parser qwen3` and tool calling enabled:
`llm_configs/qwen3.6-35B-A3B.yaml` for `Qwen/Qwen3.6-35B-A3B` (a 35B-total/3B-active MoE model with a
262,144-token native context) and `llm_configs/qwen3.8-27B.yaml` for `Qwen/Qwen3.8-27B` (a 27B dense hybrid
Gated DeltaNet and Gated Attention model, also with a 262,144-token native context). Both ship `_think.yaml`
reasoning siblings and are registered in `model_registry.yaml` with `reasoning_supported: true`.

Learn more: [LLM backends](core-concepts/language-models/llm.md) · [Serving with vLLM](../build-voice-agents/model-serving/vllm.md)

## 2026-08-06 — Standalone Repository

NeMo Labs Voice Agent graduated out of
[NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/voice_agent), where it had
lived as a sub-package. Development now occurs in its own repository. The Python package name
(`nemo_voice_agent`) and every import path are unchanged.

## 2026-06-13 — Evaluation Harness

This milestone shipped the two-bot evaluation harness. A simulated user bot talks to the agent under
test through a bridge that transfers audio between two WebSocket Pipecat servers and scores the outcome.

The release ported four benchmark domains:

| Domain | Scenarios | Upstream Source |
| --- | --- | --- |
| `eva_airline` | 50 | ServiceNow `eva` |
| `tau2_airline` | 50 | tau2-bench airline |
| `tau2_retail` | 114 | tau2-bench retail |
| `tau2_telecom` | 114 | tau2-bench telecom (dual-side: separate agent-side and user-side databases) |

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

## 2026-05-15 — Omni and Multimodal LLM Support

Added support for `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`, served through the
`llm_configs/nemotron_nano_v3_omni.yaml` model config (a `_think.yaml` sibling enables reasoning).
Omni models consume user audio directly, so the pipeline inserts a user audio buffer stage that
non-omni pipelines omit.

Learn more: [Multimodal models](core-concepts/language-models/multimodal.md) · [Reasoning](core-concepts/language-models/reasoning.md)

## 2026-01-26 — Nemotron-3-Nano-30B-A3B and Magpie TTS

This milestone added the following language and speech model support.

- Added support for the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` LLM. The
  `llm_configs/nemotron_nano_v3.yaml` model configuration covers this family. The NVFP4 variant needs a
  GPU with FP4 support and was the shipped default until the 2026-09-02 change to Nemotron-3.5-Lightning.
- Added support for the `nvidia/magpie_tts_multilingual_357m` TTS model. Its configuration is
  `tts_configs/magpie_tts_multilingual_357m.yaml`.

Learn more: [LLM backends](core-concepts/language-models/llm.md) · [TTS](core-concepts/speech-pipeline/tts.md) ·
[Model registry](../build-voice-agents/configure/model-registry.md)

## 2025-12-31 — Tool-Calling Examples and Diarization Upgrade

This milestone expanded tool-calling examples and updated the default diarization model.

- Added tool-calling examples. The Kokoro TTS service lets the LLM change how the agent sounds
  mid-conversation. It registers `tool_tts_speak_faster`, `tool_tts_speak_slower`,
  `tool_tts_set_speed`, `tool_tts_reset_speed`, `tool_tts_set_voice` (American or British accent,
  male or female voice), and `tool_tts_reset_voice`. A standalone example function,
  `tool_get_city_weather`, looks up current weather with the `python_weather` package.
- Updated the diarization model to `nvidia/diar_streaming_sortformer_4spk-v2.1`, which handles up to
  four speakers across user turns.

Learn more: [Tool calling](../build-voice-agents/tools/tool-calling.md) · [Custom tools](../build-voice-agents/tools/custom-tools.md) ·
[Diarization](core-concepts/speech-pipeline/diarization.md)

## 2025-11-14 — Joint ASR and End-of-Utterance Detection

Added support for `nvidia/parakeet_realtime_eou_120m-v1`, which predicts end-of-utterance jointly
with transcription. It is the shipped default STT model and is optimized for lowest latency. It does
not emit punctuation or capitalization.

Learn more: [ASR](core-concepts/speech-pipeline/asr.md) · [Turn taking](core-concepts/speech-pipeline/turn-taking.md)

## 2025-10-10 — Kokoro-82M TTS

Added support for the `hexgrad/Kokoro-82M` TTS model, configured through
`tts_configs/kokoro_82M.yaml`.

Learn more: [TTS](core-concepts/speech-pipeline/tts.md)

## 2025-10-03 — vLLM Serving and Automatic Backend Selection

This milestone added the following serving and backend-selection capabilities.

- Added vLLM as an LLM serving backend alongside Hugging Face.
- Added automatic backend selection: with `llm.type: auto`, the server tries vLLM first and falls
  back to Hugging Face, so parameters for both backends must be valid.
- Made `nvidia/NVIDIA-Nemotron-Nano-9B-v2` the default LLM at the time. The current default is the
  Nemotron-3-Nano-30B-A3B family, as described in the 2026-01-26 entry.

Learn more: [vLLM backend](../build-voice-agents/model-serving/vllm.md) · [LLM backends](core-concepts/language-models/llm.md)

## 2025-09-05 — First Release

First release of NeMo Labs Voice Agent: a real-time speech-to-speech pipeline built on Pipecat, with
streaming ASR, speaker diarization, LLM, and TTS behind a WebSocket server, plus a browser client.
Shipped at the time as a sub-package of
[NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/voice_agent). Refer to the
2026-08-06 entry for the move to a standalone repository.

Learn more: [Architecture](architecture.md) ·
[Quickstart](../get-started/quickstart.md)

## Upcoming Work

The repository `README.md` lists the following planned work:

- Accuracy and robustness improvements to the ASR models.
- Combined ASR and speaker diarization to handle overlapping speech.
- More evaluation domains and scenarios.

## Related Topics

Use these pages to begin with the current project, understand its architecture, or review its evaluation capabilities.

- [Quickstart](../get-started/quickstart.md) — run the current default configuration.
- [Architecture overview](architecture.md) — understand the runtime pipeline and services.
- [Evaluation overview](../evaluate/index.md) — understand the evaluation workflow and scoring signals.
