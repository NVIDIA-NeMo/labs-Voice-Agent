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

# Serving with vLLM

With `llm.type: vllm`, NeMo Labs Voice Agent does not load the LLM in-process. It speaks to an
OpenAI-compatible vLLM server over `llm.base_url`. This page covers who starts that server, which flags it
needs, and how to point the agent at one that is already running.

## The model sub-YAML is the source of truth

Serving flags live in the model config that `llm.model_config` points at — not in `default.yaml`, and not in
this page. The sub-YAML **overrides** `default.yaml` for every `llm.*` key it sets, which is why the shipped
`llm.type: auto` ends up as `vllm`. Read
`examples/generic_voice_agent/server/server_configs/llm_configs/nemotron_nano_v3.yaml` before copying any
command from here; if the two disagree, the YAML wins.

Keys that matter for vLLM:

| Key | Purpose |
| --- | --- |
| `llm.type` | `auto`, `hf`, `vllm`, or `nvidia`. `auto` probes vLLM support for the model and falls back to HuggingFace. |
| `llm.model` | Sent as the `model` field on every request, so it must match the id vLLM serves. |
| `llm.base_url` | Where the agent sends completions. Shipped default: `http://localhost:8000/v1`. |
| `llm.api_key` | vLLM ignores it unless it was started with an API key; the configs ship `"EMPTY"`. |
| `llm.start_vllm_on_init` | `true` makes the agent spawn `vllm serve` at startup. `false` means you start it. |
| `llm.vllm_server_params` | A single string of CLI flags appended to `vllm serve <model>`. Only read when spawning. |
| `llm.vllm_generation_params` | Per-request sampling settings (OpenAI fields plus an `extra` dict). |
| `llm.dtype` | Prepended as `--dtype <value>` when spawning, unless the flag string already contains `dtype`. |
| `llm.device` | Read only by the `hf` backend. It is not passed to `VLLMService`, so it has no effect on the spawned vLLM process (see below). |

## Who starts vLLM

| `start_vllm_on_init` | Behavior | Ships with |
| --- | --- | --- |
| `false` | You run `vllm serve` in a separate terminal. The agent just connects to `base_url`. | `nemotron_nano_v3.yaml`, `nemotron_nano_v3_think.yaml`, both `*_omni*.yaml`, and the eval configs under `evaluation/server_configs/` |
| `true` | `VLLMService.__init__` spawns `vllm serve` and blocks until the server reports the model. | `nemotron_nano_v2.yaml`, `qwen3-8B.yaml`, `qwen2.5-7B.yaml`, `llama3.1-8B-instruct.yaml`, `hf_llm_generic.yaml` |

Because the shipped default sets `false`, running `python server.py` on its own will not produce a working
agent — start vLLM first.

## Start vLLM yourself (shipped default)

These flags are the `vllm_server_params` string from `nemotron_nano_v3.yaml`, one flag per line:

```bash
source .venv/bin/activate
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Wait until vLLM reports it is serving, then confirm the model id the agent will ask for:

```bash
curl -s http://localhost:8000/v1/models
```

Only then start the agent from `examples/generic_voice_agent/server/`.

Notes on individual flags:

- `--max-num-seqs 1` — the agent serves one client at a time (a second WebSocket connection is rejected with
  close code 1013 and the incumbent is kept), so extra sequence slots only cost memory.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder` — required for [tool calling](../tools/tool-calling.md).
  Without them the model's tool calls arrive as plain text. Some models need a parser plugin file instead; see
  [vLLM Plugins](vllm-plugins.md).
- `--reasoning-parser nemotron_v3` — strips reasoning content out of the response server-side. See
  [Reasoning Mode](../../about/core-concepts/language-models/reasoning.md).
- `--gpu-memory-utilization 0.8` leaves headroom on the same GPU for ASR, diarization, and TTS. Lower it if
  those models fail to allocate.

`nemotron_nano_v3_think.yaml` uses the **same** server flags, so one running server serves both configs. The
differences are all request-side: `enable_thinking: True` under
`vllm_generation_params.extra.extra_body.chat_template_kwargs`, `max_new_tokens` raised from 1024 to 4096, and a
new `thinking_budget: 2048` forwarded as `thinking_token_budget`.

## Let the agent start vLLM

Set `start_vllm_on_init: true` in the model sub-YAML. At startup `VLLMService` (in
`nemo_voice_agent/pipecat/services/nemo/llm.py`) does the following:

1. Takes the port from `base_url`; an explicit `--port` inside `vllm_server_params` overrides it with a warning.
2. Probes `/v1/models` on that port. If a server is already answering there with the **same** model id, it
   reuses it and skips the spawn entirely.
3. Otherwise scans upward from that port for a free one, rewrites the base URL to
   `http://localhost:<port>/v1`, and runs `vllm serve <llm.model> <vllm_server_params>`.
4. Polls `/v1/models` every few seconds until the model appears, and raises if the child process dies or the
   wait exceeds its (non-configurable) one-hour cap.
5. Terminates the child process when the pipeline stops or is cancelled.

Two constraints follow from the implementation:

- **Local only.** The resolved URL is always `localhost`, so a remote `base_url` is discarded in this mode.
  Use `start_vllm_on_init: false` for a remote server.
- **No shell quoting.** `vllm_server_params` is split on whitespace, so any flag whose value contains a space
  (such as the omni `--limit-mm-per-prompt` JSON) is mangled. Start those models manually.

`llm.device` does **not** reach the spawned vLLM process: `get_llm_service_from_config` passes `device` only on
the `hf` path, so `VLLMService` always uses its built-in `cuda` default and never sets `CUDA_VISIBLE_DEVICES`.
To pin the spawned server to a specific GPU, export `CUDA_VISIBLE_DEVICES` in the shell that starts the agent
server, or start vLLM yourself with `start_vllm_on_init: false`.

## Point at an already-running vLLM

Nothing else is needed — set `base_url` to the server and leave `start_vllm_on_init: false`:

```yaml
# llm_configs/<your-model>.yaml
type: vllm
base_url: "http://gpu-host.internal:8000/v1"
api_key: "EMPTY"
start_vllm_on_init: false
```

Set `llm.model` (top level, in `default.yaml`) to exactly the id returned by that server's `/v1/models`,
otherwise every request fails with a 404 from vLLM. Start the remote server with `--host 0.0.0.0` so it
accepts connections from outside its machine. For a hosted endpoint rather than your own vLLM, use
`llm.type: nvidia` instead — see [NVIDIA NIM & Riva](nvidia-nim.md).

## Tensor parallelism and VRAM

The shipped default is NVFP4 and fits one GPU with FP4 support at `--tensor-parallel-size 1`. Rough sizing
guidance when you switch models: about 21 GB for a 9B LLM and 13 GB for a 4B LLM, plus a few GB for the
speech models sharing the GPU. `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` needs more than 60 GB, so raise
`--tensor-parallel-size` to 2 (and set `--max-model-len 8192` to keep the KV cache in budget).

Tuning order when you hit an out-of-memory error:

1. Lower `--gpu-memory-utilization`.
2. Lower `--max-model-len`.
3. Raise `--tensor-parallel-size` to spread across GPUs, or move ASR/diarization/TTS to a different GPU with
   their own `device:` keys.

## Omni (audio-in) recipe

`nemotron_nano_v3_omni.yaml` sets `is_omni_model: true`, which inserts the user-audio buffer into the
pipeline so raw audio reaches the LLM. Its `vllm_server_params` add the multimodal flags:

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 131072 \
    --video-pruning-rate 0.5 \
    --allowed-local-media-path / \
    --limit-mm-per-prompt '{"video": 999, "image": 999, "audio": 999}' \
    --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Keep `start_vllm_on_init: false` here: the quoted JSON values would not survive the whitespace split
described above. Set `keep_only_last_audio_turn: false` against a self-hosted vLLM, which accepts multiple
audio turns. See [Multimodal / Omni](../../about/core-concepts/language-models/multimodal.md) for the rest of the omni keys.

## Related pages

- [LLM Backends](../../about/core-concepts/language-models/llm.md) — choosing between `auto`, `hf`, `vllm`, and `nvidia`.
- [Server Configuration](../configure/server-config.md) — how the sub-YAML merge works.
- [Troubleshooting](../../troubleshooting/index.md) — connection and startup failures.
