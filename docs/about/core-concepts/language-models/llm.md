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

# LLM Backends

NeMo Labs Voice Agent can drive its large language model (LLM) stage from four backends, selected by the
`llm.type` key. `get_llm_service_from_config` in `nemo_voice_agent/pipecat/services/nemo/llm.py` builds the
backend. The pipeline calls this function through `build_llm` in
`nemo_voice_agent/pipecat/services/nemo/builders.py`.

## Choose a Backend

Choose the backend that matches where the model runs and the capabilities your agent requires.

| `llm.type` | Runs Where | Requirements | Tool Calling |
| --- | --- | --- | --- |
| `hf` | In the server process, using `transformers` `AutoModelForCausalLM` | A local GPU (`llm.device`) and enough VRAM for `llm.dtype` | Not supported — the HF path streams raw text and never emits tool calls |
| `vllm` | An OpenAI-compatible vLLM server at `llm.base_url` | vLLM running (either started for you, or started by you) | Supported when `vllm_server_params` includes a tool-call parser |
| `nvidia` | A hosted NVIDIA endpoint | `NVIDIA_API_KEY` in the environment | Supported |
| `auto` | Resolves to `vllm` or `hf` at startup | Same as whichever it picks | Depends on the resolved backend |

### How Auto Resolves

The `auto` value follows a model-support probe to choose a backend.

`auto` tries to construct a vLLM `ModelConfig` for `llm.model` with `trust_remote_code=True`. If that
succeeds, the backend becomes `vllm`. If it raises, the server logs the reason and falls back to `hf`.
This is a *model-support* probe only. It does not check that a vLLM server is reachable. If you know
your model is vLLM-supported, set `type: vllm` explicitly rather than relying on the probe.

## Where the LLM Configuration Lives

`llm:` in `examples/generic_voice_agent/server/server_configs/default.yaml` holds the top-level block.
Each entry in `server_configs/llm_configs/` is a model sub-configuration that `ConfigManager` merges after it.

Two configuration precedence rules affect these files. For more context, refer to
[Configuration Model](../../../build-voice-agents/configure/index.md):

- **The sub-YAML wins.** Keys from the model configuration overwrite the top-level `llm:` block, not the
  reverse. `default.yaml` sets `type: auto`, but `nemotron_nano_v3.yaml` sets `type: vllm`, so the
  shipped default is vLLM. Editing `type` in `default.yaml` alone has no effect.
- **`model_config` short-circuits the registry.** When `llm.model_config` is set, only its basename is
  used and the file is loaded from `server_configs/llm_configs/`. Registry lookup (and with it the
  `_think.yaml` swap described in [Reasoning Mode](reasoning.md)) is skipped. For lookup behavior, refer to
  [Model Registry](../../../build-voice-agents/configure/model-registry.md).

If `llm.model_config` is unset and `llm.model` is absent from `model_registry.yaml`, the server logs a
warning and merges no sub-configuration. Your top-level `llm:` block must then be self-contained. To
run an arbitrary Hugging Face checkpoint, point `model_config` at `llm_configs/hf_llm_generic.yaml`
instead of relying on a fallback.

## Shipped Model Configurations

All paths are relative to `examples/generic_voice_agent/server/server_configs/llm_configs/`.

| Configuration File | `llm.model` | Backend | Tool Calling | Starts vLLM for You |
| --- | --- | --- | --- | --- |
| `nemotron_nano_v3.yaml` (default) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `vllm` | Yes | No |
| `nemotron_nano_v3_think.yaml` | Same model, reasoning enabled | `vllm` | Yes | No |
| `nemotron_nano_v3_omni.yaml` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `vllm` | Yes | No |
| `nemotron_nano_v3_omni_think.yaml` | Same model, reasoning enabled | `vllm` | Yes | No |
| `nemotron_nano_v2.yaml` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `vllm` | Yes | Yes |
| `qwen3-8B.yaml`, `qwen3-8B_think.yaml` | `Qwen/Qwen3-8B` | Inherits `auto` | No | Yes |
| `qwen2.5-7B.yaml` | `Qwen/Qwen2.5-7B-Instruct` | Inherits `auto` | No | Yes |
| `llama3.1-8B-instruct.yaml` | `meta-llama/Llama-3.1-8B-Instruct` | Inherits `auto` | No | Yes |
| `hf_llm_generic.yaml` | Model that you set | Inherits `auto` | No | Yes |

The omni configurations also set `is_omni_model: true`, which inserts the audio-buffer stage. For details,
refer to [Multimodal Models](multimodal.md). `llama3.1-8B-instruct.yaml` requires `HF_TOKEN` for gated access.

### The Shipped Default

`default.yaml` ships `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` with
`model_config: ./server_configs/llm_configs/nemotron_nano_v3.yaml`. That file sets
`start_vllm_on_init: false`, so `python server.py` cannot reach an LLM on its own. Start vLLM first in a
separate terminal with the flags from that file's `vllm_server_params`:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
    --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Wait until vLLM reports it is serving on `http://localhost:8000`, then start the agent server. To have
the agent launch vLLM instead, set `start_vllm_on_init: true` in the model configuration. For flag details,
refer to [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md).

## Parameter Precedence

Generation settings are duplicated on purpose: `llm.temperature`, `llm.top_k`, `llm.top_p`,
`llm.min_p`, and `llm.max_new_tokens` are the user-facing settings, and the per-backend blocks reference
them with OmegaConf interpolation so you only edit one place. Only the block matching the active
backend is read.

| Backend | Configuration Keys | Destination |
| --- | --- | --- |
| `hf` | `model`, `device`, `dtype`, `generation_kwargs`, `apply_chat_template_kwargs`, `reasoning_budget` | `model.generate()` and `tokenizer.apply_chat_template()` |
| `vllm` | `model`, `api_key`, `base_url`, `organization`, `project`, `default_headers`, `dtype`, `start_vllm_on_init`, `vllm_server_params`, `vllm_generation_params` | the `vllm serve` command line and the OpenAI chat-completions request |
| `nvidia` | `model`, `api_key`, `base_url`, `default_headers`, `nvidia_generation_params` | the hosted OpenAI-compatible request |
| All | `function_call_timeout_secs` | The Pipecat `LLMService` (defaults to `10.0` here, not Pipecat's `None`) |

Notes on the individual blocks:

- **`generation_kwargs` and `apply_chat_template_kwargs` (`hf` only).** Free-form dictionaries forwarded verbatim.
  `apply_chat_template_kwargs.tokenize` is forced to `False` — a `True` value is dropped with a warning.
- **`vllm_server_params` (`vllm` only).** A single flag string appended to `vllm serve <model>`. If it
  does not already mention `dtype`, the server prepends `--dtype <llm.dtype>` for you. Keep
  `--max-num-seqs 1`: the agent serves one connection at a time.
- **`vllm_generation_params` and `nvidia_generation_params`.** Cast into the Pipecat OpenAI settings
  object, so the keys must be ones that object accepts. The shipped set is `frequency_penalty`,
  `presence_penalty`, `seed`, `temperature`, `top_k`, `top_p`, `max_completion_tokens`, and `extra`.
  Anything model-specific goes under `extra` — for example `extra.extra_body.chat_template_kwargs` is
  where the Nemotron configurations set `enable_thinking`. `extra` must be a mapping or `null`. Any other
  type raises at startup.

## Switch Models

Point both keys at the model and its configuration in `default.yaml`:

```yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
  model_config: "./server_configs/llm_configs/nemotron_nano_v2.yaml"
```

Then run the server as usual:

```bash
cd examples/generic_voice_agent/server && python server.py
```

To use a configuration that is not `default.yaml`, set `SERVER_CONFIG_PATH`. The path is resolved against the
current working directory, so `cd` first:

```bash
cd examples/generic_voice_agent/server
SERVER_CONFIG_PATH=./server_configs/default_nvidia.yaml python server.py
```

## Hosted NVIDIA Endpoints

The `nvidia` backend connects the LLM stage to a hosted NVIDIA endpoint.

`server_configs/default_nvidia.yaml` is the provided example: `llm.type: nvidia`,
`base_url: https://integrate.api.nvidia.com/v1`, `model: nvidia/nemotron-3-nano-30b-a3b`. It reads
`NVIDIA_API_KEY` from the environment and raises at startup if the key is missing. Pointing `base_url`
at `https://inference-api.nvidia.com/v1` switches the expected variable to `NVIDIA_INFERENCE_API_KEY`.
Unlike the local backends, this configuration carries its generation settings inline under
`llm.nvidia_generation_params` rather than in a `llm_configs/` sub-file. For endpoint details, refer to
[NVIDIA NIM and Riva](../../../build-voice-agents/model-serving/nvidia-nim.md).

## Related Topics

Use these pages to serve models, enable advanced response behavior, and configure prompts or tools.

- [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md) and [vLLM Plugins](../../../build-voice-agents/model-serving/vllm-plugins.md) — deploy a local backend and configure plugins.
- [Reasoning Mode](reasoning.md) — configure `enable_reasoning` and the `_think.yaml` files.
- [Tool Calling](../../../build-voice-agents/tools/tool-calling.md) — configure `enable_tool_calling` and parser requirements.
- [Prompts](../../../build-voice-agents/configure/prompts.md) — configure `system_prompt`, `system_role`, and `system_prompt_suffix`.
