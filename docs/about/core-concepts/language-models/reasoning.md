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
hidden chain-of-thought block before its spoken answer, and in a voice pipeline every one of those
tokens is dead air — nothing reaches TTS until the thinking block closes. `default.yaml` therefore
ships `llm.enable_reasoning: false`, commented in-file as "it's best to turn-off reasoning for lowest
latency", and the default LLM sub-config (`server_configs/llm_configs/nemotron_nano_v3.yaml`) sends
`enable_thinking: False` to vLLM.

Turn it on when answer quality on multi-step or tool-heavy tasks matters more than time-to-first-audio.

## The four moving parts

| Piece | Where it lives | What it does |
| --- | --- | --- |
| `llm.enable_reasoning` | `server_configs/default.yaml` (and `default_nvidia.yaml`) | Boolean switch. Drives the `_think.yaml` swap and, in the NVIDIA config, is interpolated straight into the request body. |
| `*_think.yaml` sub-config | `server_configs/llm_configs/` | Sibling of each LLM config that flips the model's thinking flag and raises the token budget. |
| `tts.think_tokens` | `server_configs/tts_configs/*.yaml` | Delimiter pair the local TTS services use to suppress the reasoning span so the user never hears it. |
| `--reasoning-parser` | inside `llm.vllm_server_params` | Makes vLLM split reasoning into a separate response field, so it never enters the text stream at all. |

## Enabling reasoning

`llm.enable_reasoning: true` on its own does **not** always change the model's behavior. The swap in
`nemo_voice_agent/utils/config_manager.py` (`_configure_llm`) fires only when all three hold:

1. `server.use_model_registry: true`,
2. `llm.model_config` is **not** set — an explicit `model_config` short-circuits the registry lookup
   and marks the config as non-registry, and
3. the model's entry in `server/model_registry.yaml` has `reasoning_supported: true`.

Only then is the resolved path rewritten from `<name>.yaml` to `<name>_think.yaml`. Today
`Qwen/Qwen3-8B` is the sole registry entry with `reasoning_supported: true`.

Because `default.yaml` pins `llm.model_config` explicitly, **the swap never fires for the shipped
default model.** Point `model_config` at the think variant by hand instead.

### Route 1 — point `model_config` at the `_think.yaml` (shipped default)

Edit `examples/generic_voice_agent/server/server_configs/default.yaml`:

```yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
  model_config: "./server_configs/llm_configs/nemotron_nano_v3_think.yaml"
  enable_reasoning: true   # documentary here; the think config is what actually flips the model
```

Remember the precedence rule: the sub-YAML **overrides** `default.yaml`, so anything
`nemotron_nano_v3_think.yaml` sets (including `type: vllm` and `max_new_tokens`) wins — see
[Server Configuration](../../../build-voice-agents/configure/server-config.md). To avoid editing the shipped file, copy it and
select the copy at launch:

```bash
cd examples/generic_voice_agent/server
SERVER_CONFIG_PATH=./server_configs/my_think.yaml python server.py
```

### Route 2 — registry-driven swap

Drop `llm.model_config` entirely and let the registry resolve the file:

```yaml
server:
  use_model_registry: true
llm:
  model: "Qwen/Qwen3-8B"
  enable_reasoning: true    # resolves qwen3-8B.yaml -> qwen3-8B_think.yaml
```

To make this work for a model you added, give it a `reasoning_supported: true` entry plus a
`_think.yaml` sibling — see [Model Registry](../../../build-voice-agents/configure/model-registry.md).

### Route 3 — interpolate the flag (hosted NIM endpoints)

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

See [NVIDIA NIM Endpoints](../../../build-voice-agents/model-serving/nvidia-nim.md).

## What the `_think.yaml` variants change

Diffing each pair shows the whole delta — the rest of the config is identical.

| Pair | Difference |
| --- | --- |
| `nemotron_nano_v3.yaml` to `nemotron_nano_v3_think.yaml` | `enable_thinking` flips to `True`; `max_new_tokens` 1024 to 4096; adds `thinking_budget: 2048`, passed on as `thinking_token_budget`. |
| `nemotron_nano_v3_omni.yaml` to `nemotron_nano_v3_omni_think.yaml` | Same three changes plus sampling moves from greedy-ish (`temperature: 0.2`, `top_k: 1`) to `temperature: 0.6` / `top_p: 0.95`. |
| `qwen3-8B.yaml` to `qwen3-8B_think.yaml` | `system_prompt_suffix` flips from `/no_think` to `/think`; the `extra_body` block that forced thinking off is dropped. |

Raising `max_new_tokens` matters: the reasoning block and the spoken answer share one completion
budget, so a think config left at 1024 tokens can truncate mid-thought and produce no audio at all.

## Keeping reasoning out of the audio

`tts.think_tokens` is a two-element list of delimiters. The local NeMo TTS services strip everything
between them before synthesis, streaming-safe: text before the opening tag is spoken, chunks inside
the block are dropped, and speech resumes after the closing tag. The logic lives in
`_handle_think_tokens` in `nemo_voice_agent/pipecat/services/nemo/tts.py`.

All three shipped TTS sub-configs (`kokoro_82M.yaml`, `nemo_fastpitch-hifigan.yaml`,
`magpie_tts_multilingual_357m.yaml`) already set it:

```yaml
think_tokens: ["<think>", "</think>"]
```

Set it to `null` if you want the model to think out loud — useful for debugging, unpleasant for users.
The value must be a list of exactly **two** strings (asserted at construction), and only the local
NeMo TTS services honor it: `tts.type: nvidia` (Riva/NVCF Magpie) is built without the `think_tokens`
argument, so with a hosted TTS you must rely on the vLLM-side parser below. See
[Text-to-Speech](../speech-pipeline/tts.md).

## Filtering on the vLLM side

The more robust option is to let vLLM separate the reasoning itself. Add `--reasoning-parser` to
`llm.vllm_server_params`; vLLM then routes the thinking block to a separate response field that the
pipeline never reads, so no `<think>` delimiter ever reaches TTS and `think_tokens` becomes a
belt-and-braces fallback.

What the shipped configs use:

| Config | Reasoning parser |
| --- | --- |
| `llm_configs/nemotron_nano_v3*.yaml` (incl. omni + think) | `nemotron_v3` (vLLM built-in) |
| `evaluation/server_configs/agent.yaml`, `user.yaml` | `deepseek_r1` |
| All other shipped LLM configs | none |

Because `nemotron_nano_v3.yaml` sets `start_vllm_on_init: false`, you launch vLLM yourself with the
same flags the config expects:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
  --max-num-seqs 1 --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3
```

Details in [Serving with vLLM](../../../build-voice-agents/model-serving/vllm.md).

## Bounding how long the model thinks

Two independent mechanisms:

- **`llm.thinking_budget`** — used by the think configs, forwarded to the server as
  `thinking_token_budget` inside `vllm_generation_params.extra.extra_body`. Honored by models/servers
  that implement that field (Nemotron-3 Nano, hosted NIM).
- **`ReasoningBudgetLogitsProcessor`** — a vLLM plugin shipped in this repo
  (`nemo_voice_agent/vllm/v1/sample/logits_processor/`) that counts tokens inside the thinking block
  and forces the closing sequence when the budget is hit. It is loaded with `--logits-processors` and
  driven per-request via `vllm_xargs`, so it works for models that ignore `thinking_token_budget`. No
  shipped config enables it — see [vLLM Plugins](../../../build-voice-agents/model-serving/vllm-plugins.md).

## Verifying it took effect

Reasoning mode fails quietly, so check the log rather than guessing. With `server.log_level: DEBUG`
(the default), `bot_server.log` shows:

- `Loading LLM config from: ...` — confirms whether the `_think.yaml` path was chosen.
- `Final LLM config: ...` — the merged config, including the resolved `enable_thinking` value.
- `LLM starts thinking:` / `LLM is done thinking:` — emitted by the TTS think-token handler, so their
  presence proves both that the model is reasoning and that the span is being suppressed.

Symptom-to-cause shortcuts:

| Symptom | Likely cause |
| --- | --- |
| Log still loads the non-think config after setting `enable_reasoning: true` | `llm.model_config` is set (short-circuits the registry) or the registry entry lacks `reasoning_supported: true`. |
| The bot speaks its chain of thought | `tts.think_tokens` is `null`, the model uses different delimiters, or `tts.type: nvidia` is in use with no `--reasoning-parser`. |
| Long silence, then a truncated or empty reply | `max_new_tokens` too low for reasoning plus answer, or the thinking budget is unbounded. |

More in [Troubleshooting](../../../troubleshooting/index.md) and the
[Configuration Schema](../../../reference/runtime/config-schema.md).
