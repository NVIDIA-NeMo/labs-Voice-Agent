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

# Reasoning Mode

Reasoning (thinking) mode is **off by default** in NeMo Labs Voice Agent. A reasoning model emits a
reasoning block before its spoken answer. In a voice pipeline, no content reaches text-to-speech (TTS)
until that block closes. `default.yaml` therefore ships `llm.enable_reasoning: false` to minimize latency,
and the default large language model (LLM) sub-configuration
(`server_configs/llm_configs/nemotron_nano_v3.yaml`) sends `enable_thinking: False` to vLLM.

Enable reasoning when answer quality on multi-step or tool-heavy tasks is more important than time to first audio.

## Reasoning Components

The following settings and configuration files control whether reasoning runs and whether it reaches audio output.

| Piece | Location | Function |
| --- | --- | --- |
| `llm.enable_reasoning` | `server_configs/default.yaml` (and `default_nvidia.yaml`) | Boolean switch. Drives the `_think.yaml` swap and, in the NVIDIA configuration, is interpolated into the request body. |
| `*_think.yaml` sub-configuration | `server_configs/llm_configs/` | Sibling of each LLM configuration that changes the model's thinking flag and raises the token budget. |
| `tts.think_tokens` | `server_configs/tts_configs/*.yaml` | Delimiter pair the local TTS services use to suppress the reasoning span so the user never hears it. |
| `--reasoning-parser` | inside `llm.vllm_server_params` | Makes vLLM split reasoning into a separate response field, so it never enters the text stream at all. |

## Enable Reasoning

`llm.enable_reasoning: true` on its own does **not** always change the model's behavior. The swap in
`nemo_voice_agent/utils/config_manager.py` (`_configure_llm`) occurs only when all three conditions hold:

1. `server.use_model_registry: true` is set.
2. `llm.model_config` is **not** set. An explicit `model_config` short-circuits the registry lookup
   and marks the configuration as non-registry.
3. The model entry in `server/model_registry.yaml` has `reasoning_supported: true`.

Only then is the resolved path rewritten from `<name>.yaml` to `<name>_think.yaml`. Today
`Qwen/Qwen3-8B` is the sole registry entry with `reasoning_supported: true`.

Because `default.yaml` pins `llm.model_config` explicitly, **the swap does not occur for the shipped
default model.** Point `model_config` at the think variant explicitly.

### Route 1 — Select the Think Configuration for the Shipped Default

This route uses `model_config` to select a `_think.yaml` file explicitly.

Edit `examples/generic_voice_agent/server/server_configs/default.yaml`:

```yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
  model_config: "./server_configs/llm_configs/nemotron_nano_v3_think.yaml"
  enable_reasoning: true   # documentary here; the think config is what actually flips the model
```

The sub-YAML **overrides** `default.yaml`, so settings in `nemotron_nano_v3_think.yaml`, including
`type: vllm` and `max_new_tokens`, take precedence. For details, refer to
[Server Configuration](../../../build-voice-agents/configure/server-config.md). To avoid editing the shipped
file, copy it and select the copy at launch:

```bash
cd examples/generic_voice_agent/server
SERVER_CONFIG_PATH=./server_configs/my_think.yaml python server.py
```

### Route 2 — Use a Registry-Driven Swap

Remove `llm.model_config` and let the registry resolve the file:

```yaml
server:
  use_model_registry: true
llm:
  model: "Qwen/Qwen3-8B"
  enable_reasoning: true    # resolves qwen3-8B.yaml -> qwen3-8B_think.yaml
```

To use this route for a model you added, give it a `reasoning_supported: true` entry and a
`_think.yaml` sibling. For details, refer to
[Model Registry](../../../build-voice-agents/configure/model-registry.md).

### Route 3 — Interpolate the Flag for Hosted NVIDIA NIM Endpoints

`default_nvidia.yaml` wires the switch directly into the request body with OmegaConf interpolation,
so no file swap is needed — flipping `llm.enable_reasoning` is enough:

```yaml
llm:
  type: nvidia
  enable_reasoning: false
  nvidia_generation_params:
    extra:
      extra_body:
        chat_template_kwargs:
          enable_thinking: ${llm.enable_reasoning}
        thinking_token_budget: 3000
```

For endpoint configuration, refer to
[NVIDIA NIM Endpoints](../../../build-voice-agents/model-serving/nvidia-nim.md).

## How Think Configuration Variants Work

Each `_think.yaml` variant changes only the settings listed below.

Comparing each pair shows the complete difference. The rest of each configuration is identical.

| Pair | Difference |
| --- | --- |
| `nemotron_nano_v3.yaml` to `nemotron_nano_v3_think.yaml` | Changes `enable_thinking` to `True`, raises `max_new_tokens` from 1024 to 4096, and adds `thinking_budget: 2048`, passed on as `thinking_token_budget`. |
| `nemotron_nano_v3_omni.yaml` to `nemotron_nano_v3_omni_think.yaml` | Applies the same three changes and moves sampling from near-greedy (`temperature: 0.2`, `top_k: 1`) to `temperature: 0.6` and `top_p: 0.95`. |
| `qwen3-8B.yaml` to `qwen3-8B_think.yaml` | Changes `system_prompt_suffix` from `/no_think` to `/think` and drops the `extra_body` block that forced thinking off. |

Raising `max_new_tokens` matters: the reasoning block and the spoken answer share one completion
budget, so a think configuration left at 1024 tokens can truncate mid-thought and produce no audio.

## Keep Reasoning Out of the Audio

`tts.think_tokens` is a two-element list of delimiters. The local NeMo TTS services safely strip the
delimited content from a stream before synthesis. Text before the opening tag is spoken, chunks inside
the block are dropped, and speech resumes after the closing tag. The logic lives in
`_handle_think_tokens` in `nemo_voice_agent/pipecat/services/nemo/tts.py`.

All three shipped TTS sub-configurations (`kokoro_82M.yaml`, `nemo_fastpitch-hifigan.yaml`,
`magpie_tts_multilingual_357m.yaml`) already set it:

```yaml
think_tokens: ["<think>", "</think>"]
```

Set it to `null` if you want the model to think out loud. This setting is useful for debugging but unsuitable
for end users.

The value must be a list of exactly **two** strings (asserted at construction), and only the local
NeMo TTS services honor it. `tts.type: nvidia` (Riva or NVIDIA Cloud Functions (NVCF) Magpie) is built
without the `think_tokens` argument, so with hosted TTS, rely on the vLLM-side parser. For details, refer to
[Text-to-Speech](../speech-pipeline/tts.md).

## Filter Reasoning with vLLM

The preferred option is to let vLLM separate the reasoning. Add `--reasoning-parser` to
`llm.vllm_server_params`. vLLM then routes the thinking block to a separate response field that the
pipeline never reads, so no `<think>` delimiter ever reaches TTS and `think_tokens` becomes a
secondary safeguard.

The shipped configurations use the following reasoning parsers:

| Configuration | Reasoning Parser |
| --- | --- |
| `llm_configs/nemotron_nano_v3*.yaml` (including omni and think) | `nemotron_v3` (vLLM built-in) |
| `evaluation/server_configs/agent.yaml`, `user.yaml` | `deepseek_r1` |
| All other shipped LLM configurations | None |

Because `nemotron_nano_v3.yaml` sets `start_vllm_on_init: false`, you launch vLLM yourself with the
same flags the configuration expects:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
  --max-num-seqs 1 --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3
```

For details, refer to [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md).

## Bound Reasoning Time

Two independent mechanisms bound reasoning time:

- **`llm.thinking_budget`** — used by the think configurations, forwarded to the server as
  `thinking_token_budget` inside `vllm_generation_params.extra.extra_body`. Models or servers that implement
  that field, including Nemotron-3 Nano and hosted NIM, honor the setting.
- **`ReasoningBudgetLogitsProcessor`** — a vLLM plugin shipped in this repository
  (`nemo_voice_agent/vllm/v1/sample/logits_processor/`) that counts tokens inside the thinking block
  and forces the closing sequence when the budget is hit. It is loaded with `--logits-processors` and
  driven for each request through `vllm_xargs`, so it works for models that ignore `thinking_token_budget`.
  No shipped configuration enables it. For details, refer to
  [vLLM Plugins](../../../build-voice-agents/model-serving/vllm-plugins.md).

## Verify Reasoning Behavior

Reasoning can be misconfigured without a startup error, so check the log. With `server.log_level: DEBUG`
(the default), `bot_server.log` shows:

- `Loading LLM config from: ...` — confirms whether the `_think.yaml` path was chosen.
- `Final LLM config: ...` — the merged config, including the resolved `enable_thinking` value.
- `LLM starts thinking:` and `LLM is done thinking:` — emitted by the TTS think-token handler, so their
  presence proves both that the model is reasoning and that the span is being suppressed.

Use the following symptom-to-cause mapping:

| Symptom | Likely Cause |
| --- | --- |
| Log still loads the non-think config after setting `enable_reasoning: true` | `llm.model_config` is set (short-circuits the registry) or the registry entry lacks `reasoning_supported: true`. |
| The bot speaks its chain of thought | `tts.think_tokens` is `null`, the model uses different delimiters, or `tts.type: nvidia` is in use with no `--reasoning-parser`. |
| Long silence, then a truncated or empty reply | `max_new_tokens` too low for reasoning plus answer, or the thinking budget is unbounded. |

For more diagnostic guidance, refer to [Troubleshooting](../../../troubleshooting/index.md) and the
[Configuration Schema](../../../reference/runtime/config-schema.md).

## Related Topics

Use these pages to configure the surrounding model-serving and runtime behavior.

- [LLM Backends](llm.md) — select and configure an LLM backend.
- [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md) — deploy the backend for reasoning models.
- [Server Configuration](../../../build-voice-agents/configure/server-config.md) — understand configuration precedence.
- [Troubleshooting](../../../troubleshooting/index.md) — diagnose reasoning and latency problems.
