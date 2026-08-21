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

# System Prompts

The system prompt controls how a NeMo Labs Voice Agent bot responds. `ConfigManager`
(`nemo_voice_agent/utils/config_manager.py`) assembles it from the `llm.*` keys. The
`build_context_and_aggregators` function in `nemo_voice_agent/pipecat/services/nemo/builders.py` then adds it
to the large language model (LLM) context.

## Keys

Five LLM keys control the system prompt and how it is applied to a conversation:

| Key | Type | Effect |
| --- | --- | --- |
| `llm.system_prompt` | string | The prompt itself, **or** a path to a text file containing it. If unset or `null`, a built-in three-sentence fallback ("You are a helpful AI agent named Lisa…") is used. |
| `llm.system_prompt_suffix` | string or `null` | Appended to `system_prompt` after a newline. Intended for model-specific boilerplate (thinking toggles, tool-use policy). |
| `llm.system_role` | string | The chat role the prompt is sent under. Defaults to `system`. Set it to `user` for models with no system-role support. |
| `llm.inject_dummy_user_message` | bool | When true, a synthetic user turn is appended after the system message so the model has something to answer on the first turn. Defaults to false. |
| `llm.dummy_user_message` | string | Text of that synthetic turn. Defaults to `Hello.` |

The final message list the LLM sees at startup is therefore:

```text
[{role: <system_role>, content: <system_prompt> + "\n" + <system_prompt_suffix>},
 {role: "user",        content: <dummy_user_message>}]   # only if inject_dummy_user_message
```

## Path or Literal

`system_prompt` is checked with `os.path.isfile()` against the raw string. If it names an existing file, the
file's full contents become the prompt; otherwise, the string is used verbatim. The path resolves against the
**process working directory**, not the config file's directory. Run the server from
`examples/generic_voice_agent/server/` and use paths relative to that directory, or use an absolute path.

```yaml
llm:
  # literal
  system_prompt: "You are a terse assistant. Answer in one sentence."
  # ...or a file
  system_prompt: "./example_prompts/fast-bite.txt"
```

```bash
cd examples/generic_voice_agent/server
python server.py
```

One practical difference between the two forms: the YAML is loaded with OmegaConf resolution on, so a literal
prompt containing `${...}` is interpolated as a config reference. A prompt loaded from a text file is read with
a plain `open()` and never interpolated — prefer the file form for anything long or containing braces.

## Shipped Example Prompts

`examples/generic_voice_agent/server/example_prompts/` contains three starting points.

| File | Purpose |
| --- | --- |
| `simple_chatbot.txt` | Minimal named assistant that greets at startup and stays terse. |
| `simple_chatbot_diar.txt` | Same, plus instructions for reading `speaker_0` / `speaker_1` tags out of the transcript and never echoing them back. Pair it with diarization. |
| `fast-bite.txt` | A worked domain prompt: a full lunch menu inlined above the behavioral rules, with per-speaker order tracking. |

All three end with `/no_think`, which is a Qwen-family thinking toggle. It is inert for the default
Nemotron model. Refer to [Reasoning](../../about/core-concepts/language-models/reasoning.md) for reasoning
controls. Drop that line if you copy one of these files for a different model.

## The Suffix and the Override Gotcha

`system_prompt_suffix` is set by the **model sub-YAML**, not by `default.yaml`. Because the sub-YAML overrides
the top-level config, as described in [Server configuration](server-config.md), adding `llm.system_prompt_suffix` to
`default.yaml` has no effect whenever the selected `llm.model_config` also defines it — and every shipped
`llm_configs/*.yaml` does. Edit the sub-YAML, or point `model_config` at your own copy.

`system_prompt` itself is not set by any shipped sub-YAML, so editing it in `default.yaml` works as expected.

What each family puts in the suffix:

| Config | Suffix Content |
| --- | --- |
| `nemotron_nano_v3.yaml` (default), `nemotron_nano_v3_think.yaml`, the `_omni` variants, `nemotron_nano_v2.yaml` | The tool-use policy paragraph described below. |
| `qwen3-8B.yaml` / `qwen3-8B_think.yaml` | `/no_think` / `/think`. |
| `hf_llm_generic.yaml` | `/no_think`. |
| `qwen2.5-7B.yaml`, `llama3.1-8B-instruct.yaml` | `null`. |

## Prompt Patterns for Tool Calling

When tools are registered, LLMs show two failure modes that a plain task prompt does not fix:

- **Tunnel vision** — after tools are attached, the model refuses anything outside the tool surface, or claims
  it called a tool without emitting a call.
- **Commitment bias** — after one tool-backed answer, the model keeps routing every later turn through tools
  (or, after answering from its own knowledge, stops using tools for the rest of the session).

The `system_prompt_suffix` in the Nemotron configs gives explicit instructions for both directions. Check
whether the request matches a tool before answering, and call the tool when it matches. Answer from internal
knowledge when it does not match. Do not treat the tool list as a capability limit or let earlier turns
constrain the current tool decision. If you write a domain prompt, retain these clauses. They are the reason
the suffix exists. Refer to [Tool Calling](../tools/tool-calling.md).

For a stricter agent, extend the pattern by naming the tools you expect to be called and, separately, naming
the categories of question that must be answered without a tool. Enumerating the callable tool names is
particularly effective when the same conversation also involves capabilities the model cannot invoke.

## Voice-Realization Patterns

Prompt text is spoken verbatim by TTS, so anything the model writes for the eye is read aloud as characters.
The reusable fragments live in `nemo_voice_agent/utils/voice_prompts.py`:

| Constant | What It Instructs |
| --- | --- |
| `GENERAL_PROMPT` | Plain spoken prose only; no Markdown emphasis, headings, backticks, or line-start list markers; enumerate in prose instead; standard punctuation; no emoji; no fabrication. |
| `VOICE_ALPHANUMERIC_RULE` | Spell alphanumeric identifiers one character at a time: letters as letters, digits as words, and punctuation as "dash", "at", "dot", or "hash". Speak *only* the spelled form, never the canonical string alongside it. |

Import these into your own prompt builder rather than re-writing the rules; the evaluation harness uses the
same constants, so agents stay consistent between the server and eval runs.

If diarization is on, add the speaker-tag handling from `simple_chatbot_diar.txt`. The transcript carries
`speaker_0`-style tags in angle brackets. The model must use them for attribution without repeating them in
its reply. Refer to [Diarization](../../about/core-concepts/speech-pipeline/diarization.md).

## The First Turn

`server.py` starts the conversation itself: it passes `talk_first=True` and an `LLMRunFrame` factory to
`run_bot_websocket_server`, so the model is prompted as soon as the client is ready. That first response is
generated from only the system prompt and the dummy user message, if enabled. The shipped prompts therefore
open with an explicit greeting instruction. If you remove that instruction, the bot still speaks first with a
model-generated response.

## Verifying What Was Loaded

`ConfigManager` logs the fully assembled prompt and every sub-YAML override at startup.

```bash
grep -E "System prompt:|Adding system prompt suffix:|is overridden from" \
  examples/generic_voice_agent/server/bot_server.log
```

If the prompt in the log is the built-in Lisa fallback, `llm.system_prompt` was `null` or absent. If your file
contents are missing and the literal path string appears instead, `os.path.isfile()` did not find the file —
check the working directory you launched from.

## Changing the Prompt at Runtime

The example server registers only the `reset` client message, which restores the context to the startup
messages. Live prompt replacement is an evaluation-harness feature. The `update_system_prompt` RTVI client
message swaps the system message and, optionally, the tool surface mid-session. It re-appends the configured
`system_prompt_suffix` unless the caller passes `add_suffix: false`. Refer to
[RTVI actions](../extend/protocols/rtvi-actions.md) and [RTVI messages](../../reference/runtime/rtvi-messages.md).
