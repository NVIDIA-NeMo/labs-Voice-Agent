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

# Tool Calling

Tool calling lets the LLM invoke Python functions mid-conversation — either to reach outside the
process (look up the weather) or to reconfigure the agent itself (speak faster, switch accent). NeMo Labs
Voice Agent ships both kinds as working demos, so you can try tool calling with the default config before
writing any code.

## Backend support

Tool calls are produced by the LLM backend, so only backends that parse tool-call syntax can serve them.

| `llm.type` | Tool calling | Notes |
| --- | --- | --- |
| `vllm` | Yes | Requires `--enable-auto-tool-choice` plus a `--tool-call-parser` in `vllm_server_params`. |
| `nvidia` | Yes | Hosted NIM endpoint; see [NVIDIA NIM Services](../models/nvidia-nim.md). |
| `hf` | No | `HuggingFaceLLMService` streams text only — its `_process_context` never inspects `tool_calls`. |
| `auto` | Depends | Resolves to `vllm` or `hf` at startup; only the `vllm` outcome supports tools. |

`examples/generic_voice_agent/server/server.py` gates registration purely on `llm.enable_tool_calling` —
there is **no backend check**. Setting the flag under `llm.type: hf` registers the tools and advertises them
to the model, but nothing ever fires them. The shipped model sub-YAMLs handle this for you by forcing
`type: vllm` alongside `enable_tool_calling: true` (see
`server_configs/llm_configs/nemotron_nano_v3.yaml`). Remember that the model sub-YAML *overrides*
`default.yaml`, so flipping `llm.type` in `default.yaml` has no effect.

## Enabling it

The default config (`llm_configs/nemotron_nano_v3.yaml` for
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`) already has tool calling on. Because that file sets
`start_vllm_on_init: false`, you start vLLM yourself with the tool-parser flags:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
    --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Then start the server as in the [Quickstart](../get-started/quickstart.md). On startup the log line
`Tool calling enabled; registering initial tools...` confirms the flag was read.

Relevant config keys:

| Key | Default | Purpose |
| --- | --- | --- |
| `llm.enable_tool_calling` | unset in `default.yaml`, so it reads as `false`; `true` in the shipped Nemotron sub-YAMLs | Gates all tool registration. |
| `llm.function_call_timeout_secs` | `10.0` | Seconds to wait for a tool result before giving up. |
| `llm.system_prompt_suffix` | model-specific | Appended to the system prompt; the shipped text tells the model when to reach for a tool and when not to. |
| `llm.vllm_server_params` | model-specific | Must carry `--enable-auto-tool-choice` and a `--tool-call-parser` for the vLLM path. |

Different models need different parsers. `nemotron_nano_v3.yaml` uses vLLM's built-in `qwen3_coder` parser;
`nemotron_nano_v2.yaml` loads the repo's streaming parser plugin with `--tool-parser-plugin` and
`--tool-call-parser nemotron_json` (see [vLLM plugins](../models/vllm-plugins.md)).

## Shipped demo tools

Two registration mechanisms are demonstrated, both wired up in one call to `register_direct_tools_to_llm`
in `examples/generic_voice_agent/server/server.py`.

**Direct function.** `tool_get_city_weather` in `nemo_voice_agent/utils/tool_calling/basic_tools.py` fetches
live conditions through the `python_weather` package. It speaks a "please wait" filler before the network
call, and its own HTTP request timeout is 10 seconds, independent of `function_call_timeout_secs`. Pipecat
derives the LLM-visible tool name from the Python function name.

**Component-owned tools.** `KokoroTTSService` mixes in `ToolCallingMixin` and registers six voice-control
tools from its `setup_tool_calling` method (`nemo_voice_agent/pipecat/services/nemo/tts.py`).

| Tool | Effect |
| --- | --- |
| `tool_tts_speak_faster` | Multiplies the current speaking rate by 1.15 per call. |
| `tool_tts_speak_slower` | Multiplies the current speaking rate by 0.85 per call. |
| `tool_tts_set_speed` | Scales the current rate by a caller-supplied positive factor. |
| `tool_tts_reset_speed` | Restores the rate configured in the TTS YAML. |
| `tool_tts_set_voice` | Switches accent (American or British English) and/or gender; reloads the Kokoro pipeline. |
| `tool_tts_reset_voice` | Restores the original accent and voice. |

These voice tools exist only on the Kokoro service. `MagpieTTSService` and the FastPitch/HiFi-GAN service
register no tools, and the hosted `tts.type: nvidia` service is not a `ToolCallingMixin` at all — it is
skipped with a warning. Use `tts.model: kokoro` to try them.

### Phrases to try

With the default config, say:

- "What's the weather in New York City?" / "What's the weather in Paris, Texas, USA?"
- "Can you speak faster?" / "Speak twice as fast." / "Reset to the original speaking speed."
- "Speak in a British accent." / "Switch to a male voice." / "Reset to the original voice."

## Tool-call timeouts

`llm.function_call_timeout_secs` bounds how long the pipeline waits for a tool result. The service factory in
`nemo_voice_agent/pipecat/services/nemo/llm.py` reads it once and passes it to whichever backend it builds,
defaulting to `10.0` when the key is absent. Pipecat's own default is unbounded, which would let a hung tool
stall a turn forever with no error — the explicit `10.0` restores a bounded wait. Raise it if you register a
genuinely slow tool, or set it to `null` to opt back into pipecat's unbounded behavior.

## Unknown tools

Models sometimes invent tool names that were never registered. Pipecat 1.6 answers any unmatched name with a
terminal placeholder result ("the function is not currently available"), so the turn completes instead of
wedging.

`register_schema_tools_to_llm` in `nemo_voice_agent/utils/tool_calling/base.py` installs a richer catch-all
on top of that: it returns a structured error carrying `error_type: unknown_tool` **plus the list of tool
names the LLM can actually see**, so the model can self-correct on the next turn. It deliberately reads that
list from the context's tool schema rather than the Python-side function registry, because per-scenario
re-registration replaces the former without clearing the latter — reading the registry once made an agent
announce stale bootstrap tools to the user. Pass `register_unknown_tool_handler=False` if you have already
registered your own catch-all.

This richer handler ships with the schema-tool path used by the evaluation bots. The example server uses
`register_direct_tools_to_llm`, which does not install it, so unknown calls there fall through to pipecat's
generic placeholder.

## Prompting notes

Tool availability changes how the model behaves on *non*-tool questions. Two failure modes show up in
practice:

- The model announces it is using a tool without actually calling it.
- After one tool-related answer it refuses unrelated questions ("commitment bias"), or the reverse.

The shipped `system_prompt_suffix` in the Nemotron sub-YAMLs counteracts both by explicitly instructing the
model to check whether a tool fits, to skip tools for casual conversation, and to keep answering questions
outside the tool surface. Tune that string first when adapting a new model — see
[Prompts](../configure/prompts.md).

## Next steps

- [Writing Your Own Tools](./custom-tools.md) — add a direct function or a component-owned tool.
- [Server Configuration](../configure/server-config.md) — how config layering and overrides work.
- [vLLM Backend](../models/vllm.md) — parser flags and server startup options.
- [Authoring Evaluation Tools](../evaluate/authoring-tools.md) — the schema-tool path used by the eval harness.
