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

# Configuration Model

NeMo Labs Voice Agent is configured by YAML. One top-level server config selects the models and pipeline
behaviour; each of the STT, LLM, and TTS components may additionally pull in a per-model sub-config. All
merging happens in `ConfigManager` (`nemo_voice_agent/utils/config_manager.py`), which the example server
constructs before it builds any pipeline stage.

## Where the files live

```
examples/generic_voice_agent/server/
├── model_registry.yaml          # model id -> sub-config filename
└── server_configs/
    ├── default.yaml             # the shipped top-level config
    ├── default_nvidia.yaml      # top-level config for hosted NIM endpoints
    ├── stt_configs/             # per-ASR-model sub-configs
    ├── llm_configs/             # per-LLM sub-configs
    └── tts_configs/             # per-TTS-model sub-configs
```

`ConfigManager` takes a *server base path* — the directory holding `server_configs/` and `model_registry.yaml`
— plus an optional path to the top-level file. The example server passes its own directory as the base path and
reads the file path from the `SERVER_CONFIG_PATH` environment variable, falling back to
`<base>/server_configs/default.yaml`. A relative `SERVER_CONFIG_PATH` is resolved against the current working
directory, so `cd` first:

```bash
cd examples/generic_voice_agent/server
SERVER_CONFIG_PATH=server_configs/default_nvidia.yaml python server.py
```

## The three layers

| Layer | What it sets | How it is selected |
| --- | --- | --- |
| 1. Top-level config | Every block: `server`, `transport`, `vad`, `stt`, `diar`, `turn_taking`, `llm`, `tts` | `SERVER_CONFIG_PATH`, else `server_configs/default.yaml` |
| 2. Component sub-config | Keys inside one of `stt` / `llm` / `tts` only | That block's `model_config:` field |
| 3. Model registry | Supplies the sub-config filename when `model_config:` is absent | `server.use_model_registry: true` (the default) plus a matching `model:` id |

Layers 2 and 3 are alternatives, not additions: `model_config:` short-circuits the registry lookup entirely.

### The sub-config wins

This is the single most surprising rule. For each of STT, LLM, and TTS, the merge loop copies **every** key of
the sub-config over the corresponding key in the top-level block. The top-level file does not override the
sub-config — it is the other way round.

The shipped default demonstrates it. `default.yaml` declares `llm.type: auto`, but
`llm_configs/nemotron_nano_v3.yaml` (selected by `llm.model_config:`) declares `type: vllm`, so the effective
backend is `vllm`. `ConfigManager` logs every such replacement:

```
LLM config field `type` is overridden from `auto` to `vllm` by .../llm_configs/nemotron_nano_v3.yaml
```

Practical consequence: editing a key in `default.yaml` that the sub-config also sets has no effect. Change it in
the sub-config, or point `model_config:` at a copy you own. Keys the sub-config does not mention (`model`,
`enable_reasoning`, `function_call_timeout_secs`, `system_prompt`, …) keep their top-level values. `device` is
*not* one of them — `nemotron_nano_v3.yaml` sets `device: "cuda"` itself, so a `device` you edit in
`default.yaml` is silently replaced by the sub-config's value.

The merge is a shallow, per-key replacement within the block. If a sub-config defines a nested mapping such as
`vllm_generation_params`, the whole mapping replaces the top-level one rather than merging into it.

### Only the filename of `model_config:` matters

`ConfigManager` takes `os.path.basename()` of `model_config:` and looks the file up in
`<base>/server_configs/<component>_configs/`. The directory portion of the shipped values
(`./server_configs/llm_configs/nemotron_nano_v3.yaml`) is decorative — a sub-config must physically live in the
component's own directory next to the top-level file. A missing file raises `FileNotFoundError` at startup.

### Registry auto-resolution

When a block has no `model_config:` and `server.use_model_registry` is true, `ConfigManager` looks `model:` up in
`model_registry.yaml` and uses the entry's `yaml_id` as the sub-config filename. See
[Model Registry](model-registry.md) for the file's structure. Two follow-on behaviours:

- **Reasoning swap.** If the model was resolved *through the registry*, its entry sets
  `reasoning_supported: true`, and `llm.enable_reasoning: true`, the loader substitutes the sibling file whose
  name ends in `_think.yaml`. Because an explicit `model_config:` bypasses the registry, the swap does not fire
  for the shipped default — point `model_config:` at the `_think.yaml` by hand. See
  [Reasoning](../models/reasoning.md).
- **No match.** If the model is in neither `model_config:` nor the registry, no sub-config is loaded and a
  warning is logged; the top-level block must then be complete on its own. Nothing is silently substituted.

## OmegaConf interpolation

Configs use OmegaConf, so a value may reference another key. Resolution happens in two distinct phases, and the
difference matters:

- **Top-level file: resolved eagerly at load.** `ConfigManager` calls `OmegaConf.to_container(..., resolve=True)`
  before any sub-config is merged. An interpolation here can only reference keys present in the same file;
  referencing one that only a sub-config supplies raises `InterpolationKeyError` at startup.
- **Sub-config: resolved lazily against the merged root.** Sub-config values are copied over verbatim and
  resolved on access, against the *final merged* server config. That is why `llm_configs/nemotron_nano_v3.yaml`
  can write the following and get `0.6` and `1024` — the values its own file contributed to `llm.temperature`
  and `llm.max_new_tokens`:

```yaml
vllm_generation_params:
  temperature: ${llm.temperature}
  top_p: ${llm.top_p}
  max_completion_tokens: ${llm.max_new_tokens}
```

Paths in interpolations are absolute from the config root (`llm.temperature`), not relative to the sub-config.

## System prompt

`llm.system_prompt` is **path-or-literal**: `ConfigManager` runs `os.path.isfile()` on the value and reads the
file when it exists, otherwise treats the string as the prompt itself. Relative paths resolve against the
current working directory, not the server base path. Reusable prompts ship in
`examples/generic_voice_agent/server/example_prompts/`.

`llm.system_prompt_suffix` is appended to whichever prompt was chosen, separated by a newline. The shipped
default combines a literal prompt from `default.yaml` with a tool-usage suffix from the LLM sub-config. Omit
`system_prompt` entirely and a short built-in default is used. Details in [Prompts](prompts.md).

## Which block feeds which component

Every stage is constructed by a builder in `nemo_voice_agent/pipecat/services/nemo/builders.py`, each reading one
block of the merged config.

| Block | Sub-config directory | Read by |
| --- | --- | --- |
| `server` | — | log file/level and talk-first behaviour (see note below) |
| `transport` | — | `build_ws_transport`, `build_audio_logger` |
| `vad` | — | `build_vad_analyzer` (Silero `VADParams`) |
| `stt` | `stt_configs/` | `build_stt` |
| `diar` | — | `build_diar`; returns `None` when `diar.enabled` is false |
| `turn_taking` | — | `build_turn_taking`; returns `None` when `turn_taking.enabled` is false |
| `llm` | `llm_configs/` | `build_llm`; `llm.type` selects `auto`, `hf`, `vllm`, or `nvidia` |
| `tts` | `tts_configs/` | `build_tts` and `build_llm_text_processor` |

Note on the `server` block: `server.log_file`, `server.log_level`, `server.create_new_log`,
`server.overwrite_existing_log`, and `server.talk_first` are honoured by the evaluation bot server
(`evaluation/bot_server.py`). The example server in `examples/generic_voice_agent/server/server.py` hardcodes
talk-first and calls logging setup with its defaults, writing `bot_server.log` at `DEBUG`.

`llm.type: auto` is resolved at service-construction time, not by `ConfigManager`: the LLM factory probes whether
vLLM can load the model and falls back to the HuggingFace backend if not. See [LLM Backends](../models/llm.md).

## Inspecting the merged result

The startup log is the source of truth. `ConfigManager` emits `Final STT config:`, `Final LLM config:`, and
`Final TTS config:` lines after each merge, plus one `... is overridden from ... by ...` line per replaced key;
the example server then logs the fully resolved config as `Server config:`. Check those lines first whenever a
setting appears to be ignored — the override log names the file that won.

## Gotchas

- The shipped `llm_configs/nemotron_nano_v3.yaml` sets `start_vllm_on_init: false`, so `python server.py` alone
  will not work: start vLLM yourself first, or flip that key. See [vLLM Backend](../models/vllm.md).
- `turn_taking.backchannel_phrases_path` is tried against the working directory first, then against the server
  base path, and raises `FileNotFoundError` naming both if neither exists. An inline list or `null` is also
  accepted — `null` lets any speech interrupt the bot.
- Only one client may be connected at a time; a second connection is rejected with WebSocket close code 1013 and
  the incumbent is kept. No config key changes this.

## Next

- [Server Config Reference](server-config.md) — block-by-block walkthrough of the top-level file.
- [Model Registry](model-registry.md) — adding a model so it auto-resolves.
- [Audio Logging](audio-logging.md) — the `transport.record_audio_data` path.
- [Config Schema](../reference/config-schema.md) — full key listing.
