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

# vLLM Plugins

NeMo Labs Voice Agent ships three files that run **inside the vLLM process**, not inside the voice agent
process: a tool-call parser, a reasoning parser, and a logits processor. The `vllm serve` command-line flags
load them, so their logging, imports, and Python environment belong to vLLM — nothing in
`nemo_voice_agent/pipecat/` can import or configure them at runtime.

**Most deployments need none of them.** Current vLLM releases cover [reasoning parsing](https://docs.vllm.ai/en/latest/features/reasoning_outputs/#quickstart) and [thinking-budget
control natively](https://docs.vllm.ai/en/latest/features/reasoning_outputs/#thinking-budget-control) for Nemotron-3 and newer, so the shipped `nemotron_nano_v3*` configs use only built-in
functionality. The custom plugins predate that support and remain for the one model that still needs a plugin
and for older vLLM releases.

## What Ships

The repository includes parser plugins for tool calls and reasoning plus a deprecated logits processor.

| File | Plugin Kind | Loaded by | Status |
| --- | --- | --- | --- |
| `examples/generic_voice_agent/server/parsers/nemotron_toolcall_parser_streaming.py` | Tool-call parser, registered as `nemotron_json` | `llm_configs/nemotron_nano_v2.yaml` using `--tool-parser-plugin` | **Current** — required for Nemotron-Nano-v2 |
| `examples/generic_voice_agent/server/parsers/nano_v3_reasoning_parser.py` | Reasoning parser, registered as `nano_v3` | Nothing | **Deprecated** — use `--reasoning-parser nemotron_v3` |
| `nemo_voice_agent/vllm/v1/sample/logits_processor/reasoning_budget_logits_processor.py` | V1 logits processor, `ReasoningBudgetLogitsProcessor` | Nothing | **Deprecated** — use `thinking_token_budget` |

### Which Do You Need?

Choose plugins according to the model's output format and whether it exposes reasoning or tool calls.

| Model | Reasoning | Thinking Budget | Tool Calls |
| --- | --- | --- | --- |
| Nemotron-3 and newer (`nemotron_nano_v3*`) | built-in `--reasoning-parser nemotron_v3` | built-in `thinking_token_budget` | built-in `--tool-call-parser qwen3_coder` |
| Nemotron-Nano-v2 (`nemotron_nano_v2`) | not configured | not configured | **plugin required** — `nemotron_json` |

The two deprecated files are kept, not deleted, so that a deployment pinned to an older vLLM — one without
`nemotron_v3` or `thinking_token_budget` — still has a working path. Prefer the built-ins on any current vLLM
release; the plugins receive no further work.

## Tool-Call Parser: nemotron_json

`nemotron_toolcall_parser_streaming.py` defines `NemotronToolParser` and registers it with vLLM's
`ToolParserManager` under the name `nemotron_json`. It is a streaming parser for Nemotron-Nano-9B-v2, which
emits tool calls wrapped in `<TOOLCALL>` control tokens. The parser intercepts those tokens and buffers
ambiguous partial tags so they do not reach the user as spoken text. It reconstructs valid objects from
incomplete JSON with `partial_json_parser` and emits monotonic deltas. This buffering matters for voice because
a leaked partial tag goes directly to text-to-speech (TTS).

It is the only plugin any shipped config loads. `nemotron_nano_v2.yaml` sets, inside `vllm_server_params`:

```
--enable-auto-tool-choice
--tool-parser-plugin server/parsers/nemotron_toolcall_parser_streaming.py
--tool-call-parser nemotron_json
```

All three flags are required together: the plugin flag imports the module (which runs the registration
decorator), and `--tool-call-parser` selects the now-registered name.

### The Plugin Path Is Relative

`server/parsers/...` is written relative to `examples/generic_voice_agent/`, and vLLM resolves it against its
own working directory. `nemotron_nano_v2.yaml` also sets `start_vllm_on_init: true`, and the spawn in
`nemo_voice_agent/pipecat/services/nemo/llm.py` calls `subprocess.Popen` without a `cwd` override — so vLLM
inherits the *voice agent server's* working directory.

Launch the server from `examples/generic_voice_agent/`, not from `server/`:

```bash
cd examples/generic_voice_agent
# export SERVER_CONFIG_PATH=server/server_configs/my_config.yaml  # optional; defaults to server_configs/default.yaml
python server/server.py
```

Config files still resolve correctly from there, because `server.py` anchors `ConfigManager` to its own
directory rather than to the working directory. If you prefer to run from elsewhere, edit
`vllm_server_params` and give `--tool-parser-plugin` an absolute path.

If you start vLLM yourself in a separate terminal (the pattern used by the default Nemotron-3 config), the
same rule applies to that shell's working directory:

```bash
cd examples/generic_voice_agent
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --trust-remote-code --enable-prefix-caching --max-num-seqs 1 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --mamba_ssm_cache_dtype float32 \
    --enable-auto-tool-choice \
    --tool-parser-plugin server/parsers/nemotron_toolcall_parser_streaming.py \
    --tool-call-parser nemotron_json
```

## Reasoning Parsers

The shipped Nemotron-3 configs do **not** use a plugin for reasoning. Every `nemotron_nano_v3*.yaml` passes
`--reasoning-parser nemotron_v3`, which is a parser built into vLLM. The evaluation configs under
`evaluation/server_configs/` pass `--reasoning-parser deepseek_r1`, also built in. Configs for other models
set no reasoning parser at all.

`examples/generic_voice_agent/server/parsers/nano_v3_reasoning_parser.py` is **deprecated**. It subclasses
vLLM's DeepSeek-R1 parser and registers itself as `nano_v3`, and it predates vLLM shipping `nemotron_v3`
natively. It is superseded: use `--reasoning-parser nemotron_v3` for Nemotron-3 and newer.

No shipped config loads it — none passes `--reasoning-parser nano_v3`, and none passes
`--reasoning-parser-plugin` to import the file, so its registration never runs. Do not mistake it for the
mechanism behind the default model's reasoning support; that is the built-in `nemotron_v3`. It is retained
only for deployments pinned to a vLLM release old enough to lack `nemotron_v3`, where both flags are needed:

```bash
cd examples/generic_voice_agent
vllm serve <model> \
    --reasoning-parser-plugin server/parsers/nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3
```

A reasoning parser strips the thinking span out of the OpenAI response entirely, so the voice agent does not receive
it. That is a different mechanism from `tts.think_tokens`, which lets the text reach the pipeline but keeps it
out of TTS. Refer to [Reasoning](../../about/core-concepts/language-models/reasoning.md).

## ReasoningBudgetLogitsProcessor (Deprecated)

> **Use `thinking_token_budget` instead.** Current vLLM releases cap thinking natively for Nemotron-3 and
> newer, and that is what the shipped configs use. `llm_configs/nemotron_nano_v3_think.yaml` sets a top-level
> `thinking_budget` and passes it through as a request parameter:
>
> ```yaml
> vllm_generation_params:
>   extra:
>     extra_body:
>       chat_template_kwargs:
>         enable_thinking: True
>       thinking_token_budget: ${llm.thinking_budget}
> ```
>
> No plugin, no `--logits-processors` flag, and it works over the plain OpenAI-compatible API. The logits
> processor below is retained only for vLLM releases that predate `thinking_token_budget`.

`nemo_voice_agent/vllm/v1/sample/logits_processor/` holds a vLLM V1 `LogitsProcessor` that caps how many
tokens a request may spend inside a thinking block. Long reasoning is the dominant latency cost in a voice
turn, so a hard budget bounds time-to-first-audio even when the model wants to keep thinking.

It tracks `<think>` and `</think>` boundaries in each request's output stream. As the count approaches the
budget, it boosts the logits of newline and end tokens. At the hard limit, it masks every other token so the
forced end sequence is emitted. It reports `is_argmax_invariant()` as `False` because forcing tokens changes
the argmax outcome.

No shipped config loads it. Enable it per the module docstring by adding `--logits-processors` to
`vllm_server_params` (or to your manual `vllm serve` command), pointing at the importable path:

```
--logits-processors '["nemo_voice_agent.vllm.v1.sample.logits_processor.reasoning_budget_logits_processor:ReasoningBudgetLogitsProcessor"]'
```

vLLM parses each entry as a fully qualified class name in `module:Type` form and imports it with `importlib`.
The `nemo_voice_agent` package must be installed in the vLLM environment. It is installed when both processes
share this repository's virtual environment.

The processor remains inactive until a request supplies a budget. Use `vllm_xargs` over the OpenAI API. For
offline use with `vllm.LLM`, set `SamplingParams(extra_args=...)`.

| Request Parameter | Type | Meaning |
| --- | --- | --- |
| `thinking_budget` | int | Max thinking tokens before the end sequence is forced. Required — omitting it, or passing `0`, leaves the request unprocessed. |
| `thinking_budget_grace_period` | int | How many tokens before the budget the logit boost starts. Defaults to 10% of `thinking_budget`. |
| `think_start_tokens` | str | Text marking the start of a thinking block. Defaults to `<think>`. |
| `think_end_tokens` | str | Text forced at cutoff. Defaults to a newline followed by `</think>`; can be a custom closing sentence. |

Values are validated when the request is admitted: a negative or non-integer `thinking_budget`, a negative
grace period, or a non-string delimiter raises `ValueError`.

To send `vllm_xargs` from the voice agent, put it under `llm.vllm_generation_params.extra` in your model
config — Pipecat merges that dict into the chat-completion request body verbatim.

Two suites cover the processor: `tests/unit/test_reasoning_budget_processor_coverage.py` (CPU-only, runs
in the default unit pass) and `tests/functional/vllm/test_reasoning_budget_logits_processor.py`.

```bash
uv run pytest tests/unit/test_reasoning_budget_processor_coverage.py
```

Note that the `llm.reasoning_budget` config key is unrelated to this plugin: it is read only on the
Hugging Face backend path in `nemo_voice_agent/pipecat/services/nemo/llm.py` and is never forwarded to vLLM.

## Related Pages

Use these pages for the server configuration and runtime behavior that load the plugins:

- [vLLM Backend](vllm.md) — starting the server, `vllm_server_params`, `start_vllm_on_init`.
- [Reasoning](../../about/core-concepts/language-models/reasoning.md) — thinking mode, `tts.think_tokens`, the `*_think.yaml` configs.
- [Tool Calling](../tools/tool-calling.md) — what the parser feeds into.
- [Testing](../../resources/contribute/testing.md) — markers and how to run the functional suites.
