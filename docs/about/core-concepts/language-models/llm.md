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

NeMo Labs Voice Agent can drive its LLM stage from four different backends, selected by the `llm.type`
key. The backend is built by `get_llm_service_from_config` in
`nemo_voice_agent/pipecat/services/nemo/llm.py`, which the pipeline reaches through `build_llm` in
`nemo_voice_agent/pipecat/services/nemo/builders.py`.

## Choose a Backend

Choose the backend that matches where the model runs and the capabilities your agent requires.

| `llm.type` | Runs where | What it needs | Tool calling |
| --- | --- | --- | --- |
| `hf` | In the server process, via `transformers` `AutoModelForCausalLM` | A local GPU (`llm.device`) and enough VRAM for `llm.dtype` | Not supported — the HF path streams raw text and never emits tool calls |
| `vllm` | An OpenAI-compatible vLLM server at `llm.base_url` | vLLM running (either started for you, or started by you) | Supported when `vllm_server_params` includes a tool-call parser |
| `nvidia` | A hosted NVIDIA endpoint | `NVIDIA_API_KEY` in the environment | Supported |
| `auto` | Resolves to `vllm` or `hf` at startup | Same as whichever it picks | Depends on the resolved backend |

### How `auto` Resolves

`auto` tries to construct a vLLM `ModelConfig` for `llm.model` with `trust_remote_code=True`. If that
succeeds, the backend becomes `vllm`; if it raises, the server logs the reason and falls back to `hf`.
This is a *model-support* probe only — it does not check that a vLLM server is reachable. If you know
your model is vLLM-supported, set `type: vllm` explicitly rather than relying on the probe.

## Where the LLM Configuration Lives

`llm:` in `examples/generic_voice_agent/server/server_configs/default.yaml` holds the top-level block.
Each entry in `server_configs/llm_configs/` is a model sub-config that gets merged in on top of it.

Two things trip people up, both covered in detail on [Configuration Model](../../../build-voice-agents/configure/index.md):

- **The sub-YAML wins.** Keys from the model config overwrite the top-level `llm:` block, not the other
  way round. `default.yaml` sets `type: auto`, but `nemotron_nano_v3.yaml` sets `type: vllm`, so the
  shipped default is vLLM. Editing `type` in `default.yaml` alone has no effect.
- **`model_config` short-circuits the registry.** When `llm.model_config` is set, only its basename is
  used and the file is loaded from `server_configs/llm_configs/`. Registry lookup (and with it the
  `_think.yaml` swap described in [Reasoning Mode](reasoning.md)) is skipped. See
  [Model Registry](../../../build-voice-agents/configure/model-registry.md).

If `llm.model_config` is unset and `llm.model` is absent from `model_registry.yaml`, the server logs a
warning and merges no sub-config at all — your top-level `llm:` block must then be self-contained. To
run an arbitrary HuggingFace checkpoint, point `model_config` at `llm_configs/hf_llm_generic.yaml`
instead of relying on a fallback.

## Shipped Model Configurations

All paths are relative to `examples/generic_voice_agent/server/server_configs/llm_configs/`.

| Config file | `llm.model` | Backend | Tool calling | Starts vLLM for you |
| --- | --- | --- | --- | --- |
| `nemotron_nano_v3.yaml` (default) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `vllm` | yes | no |
| `nemotron_nano_v3_think.yaml` | same, reasoning enabled | `vllm` | yes | no |
| `nemotron_nano_v3_omni.yaml` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `vllm` | yes | no |
| `nemotron_nano_v3_omni_think.yaml` | same, reasoning enabled | `vllm` | yes | no |
| `nemotron_nano_v2.yaml` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `vllm` | yes | yes |
| `qwen3-8B.yaml` / `qwen3-8B_think.yaml` | `Qwen/Qwen3-8B` | inherits `auto` | no | yes |
| `qwen2.5-7B.yaml` | `Qwen/Qwen2.5-7B-Instruct` | inherits `auto` | no | yes |
| `llama3.1-8B-instruct.yaml` | `meta-llama/Llama-3.1-8B-Instruct` | inherits `auto` | no | yes |
| `hf_llm_generic.yaml` | whatever you set | inherits `auto` | no | yes |

The omni configs additionally set `is_omni_model: true`, which inserts the audio-buffer stage — see
[Multimodal Models](multimodal.md). `llama3.1-8B-instruct.yaml` requires `HF_TOKEN` for gated access.

### The Shipped Default

`default.yaml` ships `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` with
`model_config: ./server_configs/llm_configs/nemotron_nano_v3.yaml`. That file sets
`start_vllm_on_init: false`, so `python server.py` on its own will fail to reach an LLM — start vLLM
first, in its own terminal, with the flags from that file's `vllm_server_params`:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
    --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Wait until vLLM reports it is serving on `http://localhost:8000`, then start the agent server. To have
the agent launch vLLM instead, set `start_vllm_on_init: true` in the model config. Details and flag
reference: [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md).

## Parameter Precedence

Generation settings are duplicated on purpose: `llm.temperature`, `llm.top_k`, `llm.top_p`,
`llm.min_p`, and `llm.max_new_tokens` are the human-facing knobs, and the per-backend blocks reference
them with OmegaConf interpolation so you only edit one place. Only the block matching the active
backend is read.

| Backend | Config keys it reads | Passed to |
| --- | --- | --- |
| `hf` | `model`, `device`, `dtype`, `generation_kwargs`, `apply_chat_template_kwargs`, `reasoning_budget` | `model.generate()` and `tokenizer.apply_chat_template()` |
| `vllm` | `model`, `api_key`, `base_url`, `organization`, `project`, `default_headers`, `dtype`, `start_vllm_on_init`, `vllm_server_params`, `vllm_generation_params` | the `vllm serve` command line and the OpenAI chat-completions request |
| `nvidia` | `model`, `api_key`, `base_url`, `default_headers`, `nvidia_generation_params` | the hosted OpenAI-compatible request |
| all | `function_call_timeout_secs` | the pipecat `LLMService` (defaults to `10.0` here, not pipecat's `None`) |

Notes on the individual blocks:

- **`generation_kwargs` / `apply_chat_template_kwargs` (`hf` only).** Free-form dicts forwarded verbatim.
  `apply_chat_template_kwargs.tokenize` is forced to `False` — a `True` value is dropped with a warning.
- **`vllm_server_params` (`vllm` only).** A single flag string appended to `vllm serve <model>`. If it
  does not already mention `dtype`, the server prepends `--dtype <llm.dtype>` for you. Keep
  `--max-num-seqs 1`: the agent serves one connection at a time.
- **`vllm_generation_params` / `nvidia_generation_params`.** Cast into the pipecat OpenAI settings
  object, so the keys must be ones that object accepts. The shipped set is `frequency_penalty`,
  `presence_penalty`, `seed`, `temperature`, `top_k`, `top_p`, `max_completion_tokens`, and `extra`.
  Anything model-specific goes under `extra` — for example `extra.extra_body.chat_template_kwargs` is
  where the Nemotron configs flip `enable_thinking`. `extra` must be a mapping or `null`; any other
  type raises at startup.

## Switch Models

Point both keys at the model and its config, in `default.yaml`:

```yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
  model_config: "./server_configs/llm_configs/nemotron_nano_v2.yaml"
```

Then run the server as usual:

```bash
cd examples/generic_voice_agent/server && python server.py
```

To use a config that is not `default.yaml`, set `SERVER_CONFIG_PATH` — it is resolved against the
current working directory, so `cd` first:

```bash
cd examples/generic_voice_agent/server
SERVER_CONFIG_PATH=./server_configs/default_nvidia.yaml python server.py
```

## Hosted Endpoints (`nvidia`)

`server_configs/default_nvidia.yaml` is the ready-made example: `llm.type: nvidia`,
`base_url: https://integrate.api.nvidia.com/v1`, `model: nvidia/nemotron-3-nano-30b-a3b`. It reads
`NVIDIA_API_KEY` from the environment and raises at startup if the key is missing. Pointing `base_url`
at `https://inference-api.nvidia.com/v1` switches the expected variable to `NVIDIA_INFERENCE_API_KEY`.
Unlike the local backends, this config carries its generation settings inline under
`llm.nvidia_generation_params` rather than in a `llm_configs/` sub-file. See
[NVIDIA NIM & Riva](../../../build-voice-agents/model-serving/nvidia-nim.md).

## Related Topics

Use these pages to serve models, enable advanced response behavior, and configure prompts or tools.

- [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md) and [vLLM Plugins](../../../build-voice-agents/model-serving/vllm-plugins.md)
- [Reasoning Mode](reasoning.md) for `enable_reasoning` and the `_think.yaml` configs
- [Tool Calling](../../../build-voice-agents/tools/tool-calling.md) for `enable_tool_calling` and parser requirements
- [Prompts](../../../build-voice-agents/configure/prompts.md) for `system_prompt`, `system_role`, and `system_prompt_suffix`
