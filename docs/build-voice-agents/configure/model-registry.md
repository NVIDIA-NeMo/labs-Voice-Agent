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

# Model Registry

The model registry maps a model identifier to its sub-config YAML. You can switch models by editing
`llm.model` instead of also updating `llm.model_config`. The registry is an optional convenience layer over
the sub-config merge described in [Server Config](server-config.md). The shipped NeMo Labs Voice Agent config
does not use it for any component.

## Where It Lives

`ConfigManager` (`nemo_voice_agent/utils/config_manager.py`) reads `model_registry.yaml` from the
**server base path** — the directory containing the server script. For the example server that is
`examples/generic_voice_agent/server/model_registry.yaml`. The path is not configurable from YAML.

The file has three top-level sections, one per component:

```yaml
llm_models:
  "Qwen/Qwen3-8B":
    yaml_id: "qwen3-8B.yaml"
    reasoning_supported: true

tts_models:
  "hexgrad/Kokoro-82M":
    yaml_id: "kokoro_82M.yaml"

stt_models:
  "stt_en_fastconformer_hybrid_large_streaming_80ms":
    yaml_id: "nemo_cache_aware_streaming.yaml"
    type: "nemo"
```

`yaml_id` is a bare filename. It is resolved against the sibling directory for that component —
`server_configs/llm_configs/`, `server_configs/tts_configs/`, or `server_configs/stt_configs/`.

## Resolution Order

For each large language model (LLM), text-to-speech (TTS), and speech-to-text (STT) component,
`ConfigManager` selects a sub-config in this order:

| Step | Condition | Result |
| --- | --- | --- |
| 1 | The component block sets `model_config:` | That file is used. **The registry is never consulted.** |
| 2 | No `model_config:` and `server.use_model_registry: true` | Look up the component's `model:` value in the matching registry section. |
| 3 | Registry hit | Load `<component>_configs/<yaml_id>`. |
| 4 | Registry miss | Log a warning and merge **no** sub-config. The top-level block must be self-sufficient. |
| 5 | `server.use_model_registry: false` | Skip steps 2-4 entirely. |

`server.use_model_registry` defaults to `true` when the key is absent.

Two path-handling details affect resolution:

- **`model_config` is reduced to its basename.** Only the filename is kept; the directory part of the
  value is discarded and the file is looked up in the component's `*_configs/` directory. So
  `model_config: "./server_configs/llm_configs/qwen3-8B.yaml"` and `model_config: "qwen3-8B.yaml"`
  behave identically, and you cannot point `model_config` at a file outside that directory.
- **STT keys strip a `.nemo` path.** If `stt.model` ends in `.nemo`, the lookup key is the basename
  without the extension, so a local checkpoint path can still match a registry entry.

Whatever sub-config gets loaded, its keys **override** the top-level block — not the other way round.
For that precedence rule, refer to [Server Config](server-config.md).

## Reasoning Auto-Swap

An LLM registry entry may declare `reasoning_supported: true`. That is a promise that a sibling file
with the same name plus a `_think` suffix exists and has been tested — for example `qwen3-8B.yaml`
and `qwen3-8B_think.yaml`.

The swap fires only when **all three** are true:

1. The config was resolved **through the registry** (no explicit `llm.model_config:`).
2. That registry entry sets `reasoning_supported: true`.
3. `llm.enable_reasoning: true` in your server config.

Then `ConfigManager` rewrites the `.yaml` suffix to `_think.yaml` before loading. If the `_think`
file is missing, startup fails with a `FileNotFoundError` naming the path it tried.

Today `Qwen/Qwen3-8B` is the only registry entry with `reasoning_supported: true`.

### The Swap Does Not Fire for the Shipped Default

`server_configs/default.yaml` sets both `llm.model` and `llm.model_config`:

```yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
  model_config: "./server_configs/llm_configs/nemotron_nano_v3.yaml"
  enable_reasoning: false
```

Because `model_config` is set, the lookup short-circuits at step 1 above. Setting
`enable_reasoning: true` alone does **not** load `nemotron_nano_v3_think.yaml` — that model is not in
the registry at all. Point `model_config` at the thinking config by hand:

```yaml
llm:
  model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
  model_config: "./server_configs/llm_configs/nemotron_nano_v3_think.yaml"
  enable_reasoning: true
```

The `_think` variant raises `max_new_tokens` to 4096, adds a `thinking_budget`, and flips the vLLM
`enable_thinking` chat-template kwarg to `True`. Both variants keep `start_vllm_on_init: false`, so
you still start vLLM yourself. For runtime behavior, refer to
[Reasoning Mode](../../about/core-concepts/language-models/reasoning.md).

## The Registry Does Not Cover Every Shipped Model

The registry is a partial, hand-maintained list. Several configs under `server_configs/` have no
entry, and the models named in `default.yaml` are among them:

| Component | `default.yaml` Value | In Registry? |
| --- | --- | --- |
| LLM | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | No |
| STT | `nvidia/parakeet_realtime_eou_120m-v1` | No |
| TTS | `kokoro` | No — the registry key is `hexgrad/Kokoro-82M` |

All three work because `default.yaml` sets an explicit `model_config` for each. Config files with no
registry entry include `nemotron_nano_v3.yaml`, `nemotron_nano_v3_think.yaml`,
`nemotron_nano_v3_omni.yaml`, `nemotron_nano_v3_omni_think.yaml`, and
`tts_configs/magpie_tts_multilingual_357m.yaml` — reach them with `model_config`.

`server_configs/default_nvidia.yaml` leaves `use_model_registry: true` but sets no `model_config`
anywhere. Its hosted model IDs miss the registry, so every component falls to step 4 and uses only the
top-level blocks. Warnings in `bot_server.log` about a model not being in the registry are expected. Refer to
[NVIDIA NIM](../model-serving/nvidia-nim.md).

## Adding an Entry

Add a registry-backed model by updating both the relevant sub-config and `model_registry.yaml`.

1. Put the sub-config in the right directory, e.g. `server_configs/llm_configs/my-model.yaml`.
2. If you want a tested thinking variant, add `server_configs/llm_configs/my-model_think.yaml`.
3. Register it:

```yaml
llm_models:
  "my-org/My-Model-8B":
    yaml_id: "my-model.yaml"
    reasoning_supported: true   # only if the _think.yaml exists
```

4. In your server config, set `llm.model: "my-org/My-Model-8B"` and **remove** `llm.model_config`.

Confirm the resolution in the log — `ConfigManager` writes a `Loading LLM config from: ...` line, plus
one line per key it overrode, before the pipeline starts.

## Turning the Registry Off

Set `server.use_model_registry: false` and give every component an explicit `model_config` (or a
complete top-level block). This is what the evaluation harness does: `evaluation/server_configs/*.yaml`
all set `use_model_registry: false`.

Do this whenever your server base path has no `model_registry.yaml`. With `use_model_registry: true` and a
missing or unreadable registry file, `ConfigManager` logs an error and falls back to an empty registry. It
then raises a `ConfigAttributeError` on the first component that reaches step 2. The first error is
`Missing key stt_models` because `_initialize_config_parameters` configures STT before LLM and TTS. The error
is `Missing key llm_models` if only the STT block has an explicit `model_config`. A component with an explicit
`model_config` bypasses the lookup, which can make the failure appear inconsistent. The `evaluation/`
directory has no registry file, so those configs disable it.

## Related Pages

Use these pages to configure the files and runtime behavior that surround registry resolution:

- [Server Config](server-config.md) — the full merge order and precedence rules.
- [Configuration Model](index.md) — how the config files fit together.
- [LLM Backends](../../about/core-concepts/language-models/llm.md) — what `llm.type` (`auto`, `hf`, `vllm`, `nvidia`) selects.
- [Reasoning Mode](../../about/core-concepts/language-models/reasoning.md) — thinking-mode behavior at runtime.
