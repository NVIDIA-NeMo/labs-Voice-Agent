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

# NVIDIA NIM and Riva

NeMo Labs Voice Agent can run automatic speech recognition (ASR), the large language model (LLM), and
text-to-speech (TTS) against hosted NVIDIA endpoints instead of local GPU models. This configuration-only
change sets `stt.type`, `llm.type`, and `tts.type` to `nvidia`. The builders in
`nemo_voice_agent/pipecat/services/nemo/` construct Pipecat's NVIDIA services from the same YAML blocks.

## Prerequisites

Before you connect the agent to hosted NVIDIA endpoints, complete the following preparation:

1. Install NeMo Labs Voice Agent by following the [Installation](../../get-started/installation.md) guide.
2. Obtain the NVIDIA API key required by the endpoint you plan to use.
3. Choose the demo-server or evaluation configuration for your run.

## Run It

`server_configs/default_nvidia.yaml` is the ready-made top-level config. All three component types are
`nvidia`, and both `diar.enabled` and `turn_taking.enabled` are `false`. There is no diarization NIM, so voice
activity detection (VAD) alone drives turn boundaries.

```bash
export NVIDIA_API_KEY="nvapi-..."
cd examples/generic_voice_agent/server
SERVER_CONFIG_PATH=./server_configs/default_nvidia.yaml python server.py
```

`SERVER_CONFIG_PATH` is resolved against the current working directory, so `cd` first. The browser
client is unchanged. Refer to [Quickstart](../../get-started/quickstart.md). Unlike the default vLLM path,
this config has no local model server to start in a second terminal.

The two-bot eval harness has matching configs (`evaluation/server_configs/agent_nvidia.yaml`,
`user_nvidia.yaml`, `agent_nvidia_omni.yaml`); refer to
[Evaluation Quickstart](../../evaluate/run-evaluations/quickstart.md).

## Credentials

Hosted NVIDIA services read credentials from environment variables rather than the server YAML.

| Variable | When It Is Used | Behavior if Missing |
| --- | --- | --- |
| `NVIDIA_API_KEY` | STT, TTS, and the LLM whenever `llm.base_url` is `https://integrate.api.nvidia.com/v1` | The LLM builder raises `ValueError` at startup. STT/TTS get the literal string `"None"` and fail later during the gRPC handshake. |
| `NVIDIA_INFERENCE_API_KEY` | The LLM only, and only when `llm.base_url` is `https://inference-api.nvidia.com/v1` | The LLM builder raises `ValueError` at startup. |

Each component also accepts an `api_key` key in its YAML block, but the environment variable wins: the
builders read `os.getenv("NVIDIA_API_KEY", config.get("api_key", "None"))`. `server.py` calls
`load_dotenv(override=True)`, so a `.env` file found from the server directory upward is applied and
overrides variables already exported in the shell.

## STT Keys

Read by `get_stt_service_from_config` in `nemo_voice_agent/pipecat/services/nemo/stt.py`, which builds
Pipecat's `NvidiaSTTService`.

| Key | Default in the Builder | Notes |
| --- | --- | --- |
| `type` | — | Must be `nemo` or `nvidia`; anything else fails an assertion at startup. |
| `model` | `nemotron-asr-streaming` | Model name sent in the NVCF function map. |
| `function_id` | the `nemotron-asr-streaming` UUID | Addresses one specific deployment. `model` and `function_id` are a matched pair — change both or neither. |
| `server` | `grpc.nvcf.nvidia.com:443` | gRPC endpoint. Point it at your own Riva host for a self-hosted NIM. |
| `sample_rate` | `16000` | Must match what the transport feeds the pipeline. |
| `api_key` | `"None"` | Fallback for `NVIDIA_API_KEY`. |

Transient stream drops need no configuration: Pipecat's `NvidiaSTTService` reconnects on gRPC errors
itself, and defers the reconnect until the user stops speaking.

## LLM Keys

Read by `get_llm_service_from_config` in `nemo_voice_agent/pipecat/services/nemo/llm.py`, which builds
Pipecat's `NvidiaLLMService` (an OpenAI-compatible client).

| Key | Value in `default_nvidia.yaml` | Notes |
| --- | --- | --- |
| `type` | `nvidia` | One of `auto`, `hf`, `vllm`, `nvidia`. |
| `model` | `nvidia/nemotron-3-nano-30b-a3b` | Hosted model id, not a local checkpoint path. |
| `base_url` | `https://integrate.api.nvidia.com/v1` | Also selects which API-key variable is required, as described above. |
| `default_headers` | unset | Optional dict of extra HTTP headers. |
| `nvidia_generation_params` | inline block | Cast into Pipecat's OpenAI settings object. Holds `temperature`, `top_p`, `max_completion_tokens`, `frequency_penalty`, `presence_penalty`, `seed`, and an `extra` dict for model-specific fields. |
| `function_call_timeout_secs` | `10.0` | Seconds to wait for a tool call before giving up. `null` restores Pipecat's unbounded default. |
| `enable_tool_calling` | `true` | Described in [Tool Calling](#tool-calling). |
| `enable_reasoning` | `false` | Described in [Reasoning](#reasoning). |

`system_prompt`, `system_role`, and `system_prompt_suffix` behave exactly as on the local backends —
refer to [Prompts](../configure/prompts.md).

## TTS Keys

Read by `get_tts_service_from_config` in `nemo_voice_agent/pipecat/services/nemo/tts.py`, which builds
`ResilientNvidiaTTSService` from `nemo_voice_agent/pipecat/services/nvidia/tts.py` — a thin subclass of
Pipecat's `NvidiaTTSService`.

| Key | Default in the Builder | Notes |
| --- | --- | --- |
| `type` | — | `nemo`, `nvidia`, or `nemotron`. |
| `model` | `magpie_tts_ensemble-Magpie-Multilingual` | Paired with `function_id`, same rule as STT. |
| `function_id` | the Magpie multilingual UUID | |
| `voice_id` | `Magpie-Multilingual.EN-US.Aria` | Voice name within the model. |
| `server` | `grpc.nvcf.nvidia.com:443` | gRPC endpoint. |
| `api_key` | `"None"` | Fallback for `NVIDIA_API_KEY`. |
| `max_retries` | `2` | Extra attempts after a synthesis stream fails. Set to `0` for Pipecat's single-shot behavior. |
| `retry_backoff_secs` | `0.25` | Base delay, doubled per retry. |

Why the subclass exists: upstream treats every synthesis exception as terminal, so the NVCF cold-start
failure `DEADLINE_EXCEEDED: failed to establish link to worker` silently drops a whole bot turn. The
subclass replays the buffered text and retries — but **only when the attempt produced no audio**, since
re-running mid-utterance would splice a duplicate prefix into the speech. The output sample rate on this
path is fixed at 22050 Hz by the builder; `tts.sample_rate` is not consulted.

## Tool Calling

Tool calling works on this backend. `default_nvidia.yaml` already sets `llm.enable_tool_calling: true`,
which is the only thing `server.py` checks before registering tools — there is no backend gate.

One difference from the local default: component-owned tools come from services that mix in
`ToolCallingMixin`, and the NVIDIA TTS service does not. The Kokoro-only tools ("speak faster", "switch
accent") are therefore absent, `register_direct_tools_to_llm` logs `is not a ToolCallingMixin, skipping`,
and only the direct function `tool_get_city_weather` is registered. Refer to
[Tool Calling](../tools/tool-calling.md) and [Custom Tools](../tools/custom-tools.md).

## Reasoning

There is no `_think.yaml` swap on this path — the config interpolates the switch straight into the
request body, so flipping one boolean is enough:

```yaml
llm:
  enable_reasoning: false
  nvidia_generation_params:
    extra:
      extra_body:
        chat_template_kwargs:
          enable_thinking: ${llm.enable_reasoning}
        thinking_token_budget: 3000
```

Reasoning text does not reach TTS. Pipecat's `NvidiaLLMService` pulls `reasoning_content` from the streaming
delta and emits it as thought frames instead of spoken text. For models that emit leading `<think>` spans
inline, the service strips those spans. Refer to
[Reasoning Mode](../../about/core-concepts/language-models/reasoning.md).

## Self-Hosted NIM and Riva

The same three blocks target a NIM you host yourself: set `llm.base_url` to your endpoint's `/v1` URL,
and both `stt.server` and `tts.server` to your Riva host and gRPC port. Consider two limits before you try a
plaintext local deployment:

- The builders do not forward Pipecat's `use_ssl` flag, which defaults to `True`. A Riva server without
  TLS cannot be reached by YAML alone; it needs a builder change. Refer to
  [Builders](../extend/pipelines/builders.md).
- `stt.language` and `tts.language` are read from YAML and passed to the constructor, but Pipecat 1.6
  takes the language from its settings object instead, so the value is discarded and both services stay
  on `en-US`.

## Gotchas

Use these symptoms to distinguish credential, endpoint, and tool-calling configuration problems.

| Symptom | Cause |
| --- | --- |
| Startup logs a not-in-registry warning for the STT, LLM, and TTS model | None of the hosted model IDs appear in `model_registry.yaml`, and no block sets `model_config`. No sub-config is merged, so every key comes from `default_nvidia.yaml`. This warning is expected; refer to [Model Registry](../configure/model-registry.md). |
| gRPC auth failures although a key is exported | The key is read at service construction; a typo yields the literal `"None"` for STT/TTS, which only fails at connection time. |
| The model mentions a summary tool it cannot call | The shipped `llm.system_prompt_suffix` ends with an instruction about `SendScenarioSummaryTool`, an eval-harness tool that the example server does not register. Trim that sentence for non-eval use. |
| A changed `model` returns errors for an unrelated model | `function_id` still points at the old deployment. |

## Related

Continue with the concept or configuration reference that matches the backend you are using:

- [LLM Backends](../../about/core-concepts/language-models/llm.md) — the `auto`, `hf`, `vllm`, `nvidia` selection.
- [ASR](../../about/core-concepts/speech-pipeline/asr.md) and [TTS](../../about/core-concepts/speech-pipeline/tts.md) — the local counterparts of these blocks.
- [Server Configuration](../configure/server-config.md) — every top-level block.
- [Troubleshooting](../../troubleshooting/index.md).
